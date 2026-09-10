"""Historical resolution retrieval: finds how SpotifyCares actually handled
similar issues before, to ground the pipeline's drafted replies.

A flat, L2-normalized NumPy matrix is the whole "index" -- the corpus is a
few thousand vectors, well within the size where an actual vector database
would be pure overhead (see DECISIONS.md). Every result traces back to a
real row in the processed corpus via `thread_id`; nothing here invents or
paraphrases evidence.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import config
from .llm_client import NvidiaClient
from .schemas import ConversationThread, RetrievalCase, RetrievedExample
from .utils import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

_EMBED_BATCH_SIZE = 200
_EMBEDDINGS_SUFFIX = ".npy"


def build_cases(threads: list[ConversationThread], intents_by_thread_id: dict[str, str]) -> list[RetrievalCase]:
    """Extract one retrieval case per thread: the opening complaint and the
    brand's first reply to it (the initial resolution attempt -- later
    turns are follow-up troubleshooting on the same issue, a noisier
    grounding signal than the first response).

    `intents_by_thread_id` is supplied by the caller rather than computed
    here: retrieval doesn't own how corpus-wide intent tags are produced
    (see src/pipeline.py's `tag_corpus_intents`, which uses the trained
    simple baseline classifier rather than an LLM call per historical
    thread -- classifying ~4000 threads live would be slow and unnecessary
    when a cheap local classifier does the job for indexing purposes).
    Threads with no tag fall back to "unknown" rather than being dropped.
    """
    return [
        RetrievalCase(
            thread_id=thread.thread_id,
            customer_problem=thread.customer_turns[0].text,
            brand_reply=thread.brand_turns[0].text,
            intent=intents_by_thread_id.get(thread.thread_id, "unknown"),
        )
        for thread in threads
    ]


class HistoricalIndex:
    """Searchable, cached embedding index over `RetrievalCase` records."""

    def __init__(self, cases: list[RetrievalCase], embeddings: np.ndarray) -> None:
        if len(cases) != embeddings.shape[0]:
            raise ValueError("cases and embeddings must have the same length")
        self._cases = cases
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # guard a theoretical all-zero embedding
        self._embeddings = embeddings / norms

    @classmethod
    def build(cls, cases: list[RetrievalCase], client: NvidiaClient) -> HistoricalIndex:
        """Embed every case's customer-problem text and build the index.

        Embedded as "passage" (not "query"): these are the documents being
        searched, not the search term -- using the wrong asymmetric mode
        measurably hurts retrieval quality for QA-style embedding models.
        """
        vectors: list[list[float]] = []
        for i in range(0, len(cases), _EMBED_BATCH_SIZE):
            batch = cases[i : i + _EMBED_BATCH_SIZE]
            texts = [c.customer_problem for c in batch]
            vectors.extend(client.embed(texts, input_type="passage"))
            logger.info("Embedded %d/%d retrieval cases", min(i + _EMBED_BATCH_SIZE, len(cases)), len(cases))
        return cls(cases, np.array(vectors, dtype=np.float32))

    def search(
        self, query: str, client: NvidiaClient, top_k: int, intent_filter: str | None = None
    ) -> list[RetrievedExample]:
        """Return the top_k historical cases most similar to `query`.

        If `intent_filter` matches nothing (an intent with zero indexed
        cases), falls back to searching the whole corpus rather than
        returning nothing -- a weaker match is more useful to a reply
        drafter than no grounding at all, and the similarity score still
        honestly reflects how weak it is.
        """
        query_vec = np.array(client.embed([query], input_type="query")[0], dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        candidate_indices = list(range(len(self._cases)))
        if intent_filter is not None:
            filtered = [i for i in candidate_indices if self._cases[i].intent == intent_filter]
            if filtered:
                candidate_indices = filtered

        similarities = self._embeddings[candidate_indices] @ query_vec
        ranked_local = np.argsort(-similarities)[:top_k]

        return [
            RetrievedExample(
                thread_id=self._cases[candidate_indices[local_i]].thread_id,
                similarity=float(similarities[local_i]),
                customer_problem=self._cases[candidate_indices[local_i]].customer_problem,
                brand_reply=self._cases[candidate_indices[local_i]].brand_reply,
                intent=self._cases[candidate_indices[local_i]].intent,
            )
            for local_i in ranked_local
        ]

    def save(self, base_path: Path) -> None:
        """Persist to `<base_path>.jsonl` (case metadata) + `<base_path>.npy` (vectors)."""
        write_jsonl(base_path.with_suffix(".jsonl"), self._cases)
        np.save(base_path.with_suffix(_EMBEDDINGS_SUFFIX), self._embeddings)

    @classmethod
    def load(cls, base_path: Path) -> HistoricalIndex:
        cases = list(read_jsonl(base_path.with_suffix(".jsonl"), RetrievalCase))
        embeddings = np.load(base_path.with_suffix(_EMBEDDINGS_SUFFIX))
        return cls(cases, embeddings)


def main() -> None:
    """Build and cache the retrieval index from the processed corpus.

    Run once (or whenever the processed corpus / intent tags change) --
    the pipeline and eval harness load the cached index rather than
    rebuilding it on every run.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    threads = list(read_jsonl(config.PROCESSED_DATA_PATH, ConversationThread))

    try:
        from .baselines import load_or_train_intent_classifier

        classifier = load_or_train_intent_classifier()
        intents_by_thread_id = {
            thread.thread_id: classifier.predict(thread.customer_turns[0].text) for thread in threads
        }
        logger.info("Tagged %d retrieval cases with the trained baseline classifier", len(intents_by_thread_id))
    except FileNotFoundError:
        # The golden set (needed to train the baseline classifier) doesn't
        # exist yet. The index still builds and is fully searchable -- it
        # just can't be filtered by intent until the classifier exists.
        logger.warning("No golden set found; building the retrieval index without intent tags")
        intents_by_thread_id = {}

    cases = build_cases(threads, intents_by_thread_id)
    client = NvidiaClient()
    index = HistoricalIndex.build(cases, client)
    index_path = config.PROCESSED_DATA_PATH.parent / "retrieval_index"
    index.save(index_path)
    logger.info("Saved retrieval index (%d cases) to %s.{jsonl,npy}", len(cases), index_path)


if __name__ == "__main__":
    main()
