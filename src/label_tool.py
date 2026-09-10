"""Terminal tool for building the golden evaluation set via hybrid labeling.

Workflow (see golden/labeling_rubric.md and report/DECISIONS.md):
1. A human labels a seed set (config.GOLDEN_SEED_LABEL_COUNT threads) with
   no LLM suggestion shown -- this anchors the taxonomy in the labeler's
   own judgment and supplies few-shot examples for step 2.
2. For the remaining threads, an LLM drafts a suggested (intent, action,
   reason) using the seed labels + rubric as context. The human reviews
   every suggestion and explicitly accepts or edits it before it is
   written -- a suggestion is never accepted silently (see
   `LabelSource` in schemas.py, which records exactly what happened).

Resumable: already-labeled thread_ids (read back from golden/golden_set.csv)
are skipped on restart, and each example is appended to disk immediately,
so an interrupted session loses at most the example in progress.
"""
from __future__ import annotations

import csv
import logging
import random
from pathlib import Path

from pydantic import BaseModel

from . import config
from .llm_client import NvidiaClient, NvidiaRequestError
from .schemas import AgentAction, ConversationThread, GoldenExample, IntentDefinition, LabelSource
from .utils import load_intents, read_jsonl, set_global_seed

logger = logging.getLogger(__name__)

_CSV_FIELDS = list(GoldenExample.model_fields.keys())
_ACTION_KEYS = {"a": AgentAction.AUTO_HANDLE, "e": AgentAction.ESCALATE}
_ACTION_LETTERS = {AgentAction.AUTO_HANDLE: "a", AgentAction.ESCALATE: "e"}


class _QuitLabeling(Exception):
    """Raised when the user types 'quit' at any prompt."""


class _SkipExample(Exception):
    """Raised when the user types 'skip' at any prompt."""


class _LabelSuggestion(BaseModel):
    intent: str
    action: AgentAction
    reason: str


def sample_candidates(
    threads: list[ConversationThread], target_size: int, seed: int
) -> list[ConversationThread]:
    """Uniform random sample.

    Stratifying by intent isn't attempted here -- the intents don't exist
    for these threads yet, that's what this tool produces. See
    golden/sampling_notes.md for the resulting intent distribution and why
    plain random sampling was judged sufficient.
    """
    rng = random.Random(seed)
    pool = threads[:]
    rng.shuffle(pool)
    return pool[:target_size]


_NULLABLE_FIELDS = ("llm_suggested_intent", "llm_suggested_action", "split")


def _load_existing(path: Path) -> list[GoldenExample]:
    """Read golden_set.csv back into GoldenExample rows.

    csv.DictWriter serializes a Python `None` as an empty string, but
    csv.DictReader reads it back as `""`, not `None` -- pydantic then
    rejects `""` for the Optional[AgentAction]/Literal fields, since
    neither accepts an empty string. Restoring `None` for the known
    nullable columns before validation makes the round trip lossless.
    """
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            for field in _NULLABLE_FIELDS:
                if row.get(field) == "":
                    row[field] = None
            rows.append(GoldenExample.model_validate(row))
        return rows


def _append(path: Path, example: GoldenExample, write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(example.model_dump(mode="json"))


def _print_thread(thread: ConversationThread) -> None:
    print("\n" + "=" * 70)
    for turn in thread.turns:
        print(f"[{turn.speaker.value:>8}] {turn.text}")
    print("=" * 70)


def _check_control_word(raw: str) -> None:
    if raw == "quit":
        raise _QuitLabeling
    if raw == "skip":
        raise _SkipExample


def _prompt_intent(intents: list[IntentDefinition], default: str | None = None) -> str:
    for i, intent in enumerate(intents, start=1):
        print(f"  {i}. {intent.name}")
    prompt = f"Intent number{f' [{default}]' if default else ''} (or 'skip'/'quit'): "
    while True:
        raw = input(prompt).strip()
        _check_control_word(raw)
        if not raw and default:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(intents):
            return intents[int(raw) - 1].name
        print("Invalid choice, try again.")


def _prompt_action(default: AgentAction | None = None) -> AgentAction:
    default_letter = _ACTION_LETTERS[default] if default else None
    prompt = f"Action a=auto_handle / e=escalate{f' [{default_letter}]' if default_letter else ''}: "
    while True:
        raw = input(prompt).strip().lower()
        _check_control_word(raw)
        if not raw and default:
            return default
        if raw in _ACTION_KEYS:
            return _ACTION_KEYS[raw]
        print("Invalid choice, try again.")


def _prompt_reason(default: str | None = None) -> str:
    raw = input(f"Reason{f' [{default}]' if default else ''}: ").strip()
    _check_control_word(raw)
    return raw or default or ""


def label_seed_example(thread: ConversationThread, intents: list[IntentDefinition]) -> GoldenExample:
    """Pure human labeling, no LLM suggestion -- this becomes both a golden
    row and few-shot context for the assisted phase."""
    _print_thread(thread)
    intent = _prompt_intent(intents)
    action = _prompt_action()
    reason = _prompt_reason()
    return GoldenExample(
        thread_id=thread.thread_id,
        customer_text=thread.customer_turns[0].text,
        intent=intent,
        action=action,
        reason=reason,
        label_source=LabelSource.HUMAN,
    )


def suggest_label(
    thread: ConversationThread,
    intents: list[IntentDefinition],
    seed_examples: list[GoldenExample],
    client: NvidiaClient,
) -> _LabelSuggestion:
    """Ask the LLM to draft a label using the rubric and seed examples as context."""
    intent_lines = "\n".join(f"- {i.name}: {i.description.strip()}" for i in intents)
    few_shot = "\n".join(
        f"Message: {e.customer_text!r}\n"
        f"-> intent={e.intent!r}, action={e.action.value!r}, reason={e.reason!r}"
        for e in seed_examples[: min(8, len(seed_examples))]
    )
    prompt = (
        "You are drafting a label for a support-agent evaluation set. Choose exactly one intent "
        f"from this list:\n{intent_lines}\n\n"
        "Decide auto_handle vs escalate: escalate if it needs identity/account verification, a "
        "specific promise (refund amount, timeline), or the customer is clearly angry/urgent; "
        "otherwise auto_handle.\n\n"
        f"Examples of already-labeled messages:\n{few_shot}\n\n"
        f"Now label this message: {thread.customer_turns[0].text!r}"
    )
    return client.generate_json(prompt, _LabelSuggestion, system_instruction="You are a careful data labeler.")


def label_assisted_example(
    thread: ConversationThread,
    intents: list[IntentDefinition],
    seed_examples: list[GoldenExample],
    client: NvidiaClient,
) -> GoldenExample:
    """LLM drafts, human reviews and explicitly accepts or edits every field."""
    _print_thread(thread)
    valid_intent_names = {i.name for i in intents}
    default_intent: str | None
    default_action: AgentAction | None
    default_reason: str | None
    try:
        suggestion = suggest_label(thread, intents, seed_examples, client)
        print(
            f"LLM suggests: intent={suggestion.intent!r}, "
            f"action={suggestion.action.value!r}, reason={suggestion.reason!r}"
        )
        # A suggestion outside the taxonomy is shown for transparency but
        # never offered as a one-keystroke default -- the human must pick a
        # real entry from the menu (spec section 15: never silently accept
        # an invalid label, LLM-assisted or not).
        default_intent = suggestion.intent if suggestion.intent in valid_intent_names else None
        default_action = suggestion.action
        default_reason = suggestion.reason
    except NvidiaRequestError as exc:
        print(f"(LLM suggestion failed: {exc}; labeling from scratch)")
        default_intent, default_action, default_reason = None, None, None

    intent = _prompt_intent(intents, default=default_intent)
    action = _prompt_action(default=default_action)
    reason = _prompt_reason(default=default_reason)

    if default_intent is None:
        source = LabelSource.HUMAN
    elif (intent, action, reason) == (default_intent, default_action, default_reason):
        source = LabelSource.LLM_SUGGESTED_HUMAN_CONFIRMED
    else:
        source = LabelSource.LLM_SUGGESTED_HUMAN_EDITED

    return GoldenExample(
        thread_id=thread.thread_id,
        customer_text=thread.customer_turns[0].text,
        intent=intent,
        action=action,
        reason=reason,
        label_source=source,
        llm_suggested_intent=default_intent,
        llm_suggested_action=default_action,
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    set_global_seed(config.RANDOM_SEED)

    intents = load_intents(config.INTENTS_PATH)
    threads = list(read_jsonl(config.PROCESSED_DATA_PATH, ConversationThread))
    candidates = sample_candidates(threads, config.GOLDEN_SET_TARGET_SIZE, config.RANDOM_SEED)

    existing = _load_existing(config.GOLDEN_SET_PATH)
    labeled_ids = {e.thread_id for e in existing}
    seed_examples = [e for e in existing if e.label_source == LabelSource.HUMAN]
    remaining = [t for t in candidates if t.thread_id not in labeled_ids]

    print(f"{len(existing)} already labeled, {len(remaining)} remaining ({len(candidates)} total target).")
    print("Type 'skip' to drop an example, 'quit' to save and exit; progress saves after every example.\n")

    client = NvidiaClient()
    write_header = not config.GOLDEN_SET_PATH.exists()

    for i, thread in enumerate(remaining):
        total_done = len(existing) + i
        is_seed_phase = total_done < config.GOLDEN_SEED_LABEL_COUNT
        print(f"\n--- Example {total_done + 1}/{len(candidates)} ({'seed' if is_seed_phase else 'assisted'}) ---")
        try:
            example = (
                label_seed_example(thread, intents)
                if is_seed_phase
                else label_assisted_example(thread, intents, seed_examples, client)
            )
        except _SkipExample:
            print("Skipped.")
            continue
        except (_QuitLabeling, EOFError, KeyboardInterrupt):
            print("\nStopping; progress saved.")
            return

        if is_seed_phase:
            seed_examples.append(example)
        _append(config.GOLDEN_SET_PATH, example, write_header)
        write_header = False

    print(f"\nDone labeling this batch -- see {config.GOLDEN_SET_PATH}")


if __name__ == "__main__":
    main()
