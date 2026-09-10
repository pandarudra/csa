"""Exploratory tool: cluster customer messages to discover the intent taxonomy.

Dev-time only -- not imported by the runtime pipeline. Run it, read the
printed cluster samples, and use what you see to write intents.yaml by
hand. Clustering never writes intents.yaml itself: naming and merging
clusters into a small, evaluable taxonomy is a judgment call, not
something to automate away.
"""
from __future__ import annotations

import argparse
import logging
import random

import numpy as np
from sklearn.cluster import KMeans

from . import config
from .llm_client import NvidiaClient
from .schemas import ConversationThread
from .utils import read_jsonl, set_global_seed

logger = logging.getLogger(__name__)

_EMBED_BATCH_SIZE = 200


def _first_customer_message(thread: ConversationThread) -> str | None:
    """The customer's opening message is the strongest single signal of
    intent -- later turns are back-and-forth about that same issue."""
    customer_turns = thread.customer_turns
    return customer_turns[0].text if customer_turns else None


def sample_customer_messages(
    threads: list[ConversationThread], sample_size: int, seed: int
) -> list[str]:
    messages = [m for t in threads if (m := _first_customer_message(t)) is not None]
    if len(messages) <= sample_size:
        return messages
    rng = random.Random(seed)
    return rng.sample(messages, sample_size)


def embed_messages(messages: list[str], client: NvidiaClient) -> np.ndarray:
    """Embed messages in batches and stack into a matrix.

    Uses input_type="passage": we're clustering messages against each
    other (symmetric similarity), not searching one against the rest, so
    the "query" mode of this asymmetric embedding model doesn't apply.
    """
    vectors: list[list[float]] = []
    for i in range(0, len(messages), _EMBED_BATCH_SIZE):
        batch = messages[i : i + _EMBED_BATCH_SIZE]
        vectors.extend(client.embed(batch, input_type="passage"))
        logger.info("Embedded %d/%d messages", min(i + _EMBED_BATCH_SIZE, len(messages)), len(messages))
    return np.array(vectors)


def cluster_messages(matrix: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return kmeans.fit_predict(matrix)


def print_cluster_samples(messages: list[str], labels: np.ndarray, examples_per_cluster: int) -> None:
    for cluster_id in sorted(set(labels.tolist())):
        cluster_messages = [m for m, label in zip(messages, labels) if label == cluster_id]
        print(f"\n=== Cluster {cluster_id} ({len(cluster_messages)} messages) ===")
        for message in cluster_messages[:examples_per_cluster]:
            print(f"  - {message[:160]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--n-clusters", type=int, default=9)
    parser.add_argument("--examples-per-cluster", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_global_seed(config.RANDOM_SEED)

    threads = list(read_jsonl(config.PROCESSED_DATA_PATH, ConversationThread))
    messages = sample_customer_messages(threads, args.sample_size, config.RANDOM_SEED)
    logger.info("Sampled %d customer opening messages from %d threads", len(messages), len(threads))

    client = NvidiaClient()
    matrix = embed_messages(messages, client)
    labels = cluster_messages(matrix, args.n_clusters, config.RANDOM_SEED)
    print_cluster_samples(messages, labels, args.examples_per_cluster)


if __name__ == "__main__":
    main()
