"""Filters the raw TWCS export down to reconstructed SpotifyCares threads.

The TWCS CSV is a flat table of individual tweets linked only by
`in_response_to_tweet_id` (one parent) and `response_tweet_id` (a
comma-separated list of children -- the inverse of the former, and
redundant for our purposes). We only need the parent pointer: for every
SpotifyCares reply, walk upward to the customer's root complaint, group all
SpotifyCares replies that share a root into one conversation, and union
their ancestor chains. Unioning chains (rather than greedily re-walking
back down from the root) sidesteps branch-selection ambiguity when a root
tweet has multiple reply branches -- we know exactly which tweets belong to
the conversation because we arrived at each one by walking up from a real
SpotifyCares reply.
"""
from __future__ import annotations

import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from . import config
from .schemas import ConversationThread, Speaker, SpeakerTurn
from .utils import parse_twitter_datetime, set_global_seed, write_jsonl

logger = logging.getLogger(__name__)

_EXPECTED_HEADER = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


@dataclass(slots=True, frozen=True)
class _RawRow:
    author_id: str
    inbound: bool
    created_at: str
    text: str
    in_response_to_tweet_id: str | None


def _load_raw_index(csv_path: Path) -> dict[str, _RawRow]:
    """Read the full TWCS export into an in-memory tweet_id -> row index.

    A single pass over ~3M rows costs roughly 6s and 1.8GB RSS on this
    dataset (measured locally) -- acceptable for a one-time data-prep run
    that produces the small cached JSONL the rest of the pipeline reads.
    `response_tweet_id` is parsed for validation but not stored, since
    thread reconstruction only ever walks upward via `in_response_to_tweet_id`.
    """
    index: dict[str, _RawRow] = {}
    skipped = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != _EXPECTED_HEADER:
            raise ValueError(f"Unexpected TWCS columns: {header!r}")
        for row in reader:
            if len(row) != len(_EXPECTED_HEADER):
                # A handful of rows in the Kaggle export can have stray
                # delimiters inside unescaped text; skip rather than crash.
                skipped += 1
                continue
            tweet_id, author_id, inbound, created_at, text, _response_ids, parent_id = row
            index[tweet_id] = _RawRow(
                author_id=author_id,
                inbound=inbound == "True",
                created_at=created_at,
                text=text,
                in_response_to_tweet_id=parent_id or None,
            )
    if skipped:
        logger.warning("Skipped %d malformed rows while indexing", skipped)
    return index


def _walk_ancestors(start_id: str, index: dict[str, _RawRow]) -> list[tuple[str, _RawRow]]:
    """Walk from `start_id` up to the thread root via `in_response_to_tweet_id`.

    Stops without error at the first of: no parent recorded, the parent id
    is missing from the index (a dangling reference -- the export does not
    always capture a tweet's full ancestor chain), a repeated id (cycle
    guard), or `MAX_THREAD_WALK_DEPTH` hops. A missing link is never
    fabricated; it just yields a shorter thread anchored lower down.
    """
    chain: list[tuple[str, _RawRow]] = []
    visited: set[str] = set()
    current_id: str | None = start_id
    for _ in range(config.MAX_THREAD_WALK_DEPTH):
        if current_id is None or current_id in visited:
            break
        visited.add(current_id)
        row = index.get(current_id)
        if row is None:
            break
        chain.append((current_id, row))
        current_id = row.in_response_to_tweet_id
    return chain


def _role(row: _RawRow) -> Speaker | None:
    """Classify a row as customer/brand, or None if it belongs to neither.

    `None` covers third parties dragged into a chain -- a different brand
    the customer also tagged, or an unrelated reply -- which are dropped
    rather than mislabeled, since assigning them a role would misrepresent
    who said what in this brand's conversation.
    """
    if row.inbound:
        return Speaker.CUSTOMER
    if row.author_id == config.BRAND_HANDLE:
        return Speaker.BRAND
    return None


def _sort_key(raw_created_at: str) -> float:
    try:
        return parse_twitter_datetime(raw_created_at).timestamp()
    except ValueError:
        return 0.0


def build_conversation_threads(csv_path: Path) -> list[ConversationThread]:
    """Reconstruct every usable SpotifyCares conversation thread in `csv_path`.

    Returns one `ConversationThread` per distinct root customer tweet that
    has at least one customer turn and at least one SpotifyCares turn.
    """
    logger.info("Indexing raw TWCS export from %s", csv_path)
    index = _load_raw_index(csv_path)
    logger.info("Indexed %d rows", len(index))

    brand_tweet_ids = [
        tweet_id
        for tweet_id, row in index.items()
        if row.author_id == config.BRAND_HANDLE and not row.inbound
    ]
    logger.info("Found %d %s tweets", len(brand_tweet_ids), config.BRAND_HANDLE)

    root_to_rows: dict[str, dict[str, _RawRow]] = {}
    for brand_tweet_id in brand_tweet_ids:
        chain = _walk_ancestors(brand_tweet_id, index)
        if not chain:
            continue
        root_id = chain[-1][0]
        bucket = root_to_rows.setdefault(root_id, {})
        for tweet_id, row in chain:
            bucket[tweet_id] = row

    threads: list[ConversationThread] = []
    oversized_roots = 0
    for root_id, rows_by_id in root_to_rows.items():
        speaker_turns: list[SpeakerTurn] = []
        for tweet_id, row in rows_by_id.items():
            speaker = _role(row)
            if speaker is None:
                continue
            speaker_turns.append(
                SpeakerTurn(speaker=speaker, tweet_id=tweet_id, text=row.text, created_at=row.created_at)
            )
        # Chronological order; ties fall back to numeric tweet_id, which is
        # assigned in file order and is a reasonable recency proxy.
        speaker_turns.sort(key=lambda t: (_sort_key(t.created_at), int(t.tweet_id)))

        has_customer = any(t.speaker == Speaker.CUSTOMER for t in speaker_turns)
        has_brand = any(t.speaker == Speaker.BRAND for t in speaker_turns)
        if not (has_customer and has_brand):
            continue
        if len(speaker_turns) > config.MAX_THREAD_TURNS:
            # Almost certainly a viral broadcast tweet with many unrelated
            # repliers sharing one root, not a real support conversation.
            oversized_roots += 1
            continue

        threads.append(
            ConversationThread(thread_id=root_id, created_at=speaker_turns[0].created_at, turns=speaker_turns)
        )

    if oversized_roots:
        logger.info(
            "Dropped %d oversized roots (> %d turns, likely viral broadcast fan-out)",
            oversized_roots,
            config.MAX_THREAD_TURNS,
        )
    logger.info("Reconstructed %d usable threads", len(threads))
    return threads


def sample_threads(
    threads: list[ConversationThread], target_size: int, seed: int
) -> list[ConversationThread]:
    """Deterministically subsample threads for a runnable, gradeable corpus."""
    if len(threads) <= target_size:
        return threads
    rng = random.Random(seed)
    return rng.sample(threads, target_size)


def _self_check(threads: list[ConversationThread]) -> None:
    """Cheap invariant checks -- not a full test suite, just enough to catch
    a broken reconstruction before it silently corrupts every downstream step.
    """
    assert threads, "no threads produced"
    sample = threads[: min(50, len(threads))]
    for thread in sample:
        assert thread.customer_turns, f"thread {thread.thread_id} has no customer turns"
        assert thread.brand_turns, f"thread {thread.thread_id} has no brand turns"
        assert len(thread.turns) <= config.MAX_THREAD_TURNS, f"thread {thread.thread_id} exceeds size cap"
        timestamps = [_sort_key(t.created_at) for t in thread.turns]
        assert timestamps == sorted(timestamps), f"thread {thread.thread_id} turns are out of order"
    logger.info("Self-check passed on %d sampled threads", len(sample))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_global_seed(config.RANDOM_SEED)

    threads = build_conversation_threads(config.DATA_PATH)
    sampled = sample_threads(threads, config.TARGET_THREAD_SAMPLE_SIZE, config.RANDOM_SEED)
    _self_check(sampled)

    count = write_jsonl(config.PROCESSED_DATA_PATH, sampled)
    logger.info("Wrote %d threads to %s", count, config.PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()
