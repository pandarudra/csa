"""Two non-LLM baselines the full agent has to beat.

Trivial: majority-class intent, always-escalate, one canned reply.
Simple: TF-IDF + Logistic Regression intent classifier, escalation from
        intents.yaml's static `always_escalate` flags only (no review
        step), and nearest-historical-reply retrieval by TF-IDF cosine
        similarity (not embeddings). Neither baseline calls an LLM.

Both baselines' classifiers train on the golden set's *dev* split only --
the same split used to tune the full system's prompts/thresholds, so all
three systems (trivial, simple, full agent) are evaluated on the same
untouched *test* split. See report/DECISIONS.md for why 45-ish training
examples make the simple baseline intentionally weak, and why that's a
fair rather than rigged comparison.
"""
from __future__ import annotations

import logging
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from . import config
from .schemas import AgentAction, AgentResult, ConversationThread, GoldenExample, IntentDefinition, RetrievedExample
from .utils import load_intents, read_golden_csv

logger = logging.getLogger(__name__)

_CLASSIFIER_CACHE_PATH = config.RESULTS_DIR / "simple_baseline_classifier.pkl"
_CANNED_REPLY = "Thanks for reaching out! Please DM us with more details so our team can take a closer look."


def load_golden_examples(path: Path = config.GOLDEN_SET_PATH) -> list[GoldenExample]:
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found at {path}; run `python -m src.label_tool` first")
    return read_golden_csv(path)


def split_dev_test(
    examples: list[GoldenExample], dev_fraction: float, seed: int
) -> tuple[list[GoldenExample], list[GoldenExample]]:
    """Stratified by intent where possible.

    With ~200 examples across 6 intents, stratification mostly holds; an
    intent with only a single example can't be split (sklearn requires
    >= 2 per class to stratify) and falls back to an unstratified split for
    that call -- disclosed in DECISIONS.md rather than silently ignored.
    """
    labels = [e.intent for e in examples]
    counts = Counter(labels)
    stratify = labels if min(counts.values()) >= 2 else None
    if stratify is None:
        logger.warning("At least one intent has < 2 golden examples; dev/test split is not stratified")
    dev, test = train_test_split(examples, train_size=dev_fraction, random_state=seed, stratify=stratify)
    return dev, test


class MajorityIntentClassifier:
    """Trivial baseline: always predicts the single most common training intent."""

    def __init__(self, majority_intent: str) -> None:
        self.majority_intent = majority_intent

    def predict(self, _text: str) -> str:
        return self.majority_intent

    @classmethod
    def train(cls, examples: list[GoldenExample]) -> MajorityIntentClassifier:
        counts = Counter(e.intent for e in examples)
        return cls(counts.most_common(1)[0][0])


class TfidfIntentClassifier:
    """Simple baseline: TF-IDF features + Logistic Regression, no LLM."""

    def __init__(self, vectorizer: TfidfVectorizer, model: LogisticRegression) -> None:
        self._vectorizer = vectorizer
        self._model = model

    def predict(self, text: str) -> str:
        return str(self._model.predict(self._vectorizer.transform([text]))[0])

    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        vec = self._vectorizer.transform([text])
        probabilities = self._model.predict_proba(vec)[0]
        best_index = int(np.argmax(probabilities))
        return str(self._model.classes_[best_index]), float(probabilities[best_index])

    @classmethod
    def train(cls, examples: list[GoldenExample], seed: int) -> TfidfIntentClassifier:
        texts = [e.customer_text for e in examples]
        labels = [e.intent for e in examples]
        # min_df=1 because the training set (the dev split, ~45-50 rows) is
        # too small for a higher document-frequency floor to leave any
        # vocabulary at all.
        vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=1)
        features = vectorizer.fit_transform(texts)
        model = LogisticRegression(max_iter=1000, random_state=seed)
        model.fit(features, labels)
        return cls(vectorizer, model)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump((self._vectorizer, self._model), f)

    @classmethod
    def load(cls, path: Path) -> TfidfIntentClassifier:
        with path.open("rb") as f:
            vectorizer, model = pickle.load(f)
        return cls(vectorizer, model)


def load_or_train_intent_classifier() -> TfidfIntentClassifier:
    """Load the cached simple-baseline classifier, training it on the golden
    dev split if no cache exists yet.

    Raises FileNotFoundError (propagated from `load_golden_examples`) if
    the golden set itself doesn't exist yet -- callers such as
    `retrieve.py` decide how to degrade in that case rather than this
    function silently returning something misleading.
    """
    if _CLASSIFIER_CACHE_PATH.exists():
        return TfidfIntentClassifier.load(_CLASSIFIER_CACHE_PATH)
    examples = load_golden_examples()
    dev, _test = split_dev_test(examples, config.GOLDEN_DEV_FRACTION, config.RANDOM_SEED)
    classifier = TfidfIntentClassifier.train(dev, config.RANDOM_SEED)
    classifier.save(_CLASSIFIER_CACHE_PATH)
    return classifier


class RuleBasedEscalation:
    """Simple baseline escalation: only intents.yaml's static default, no review step."""

    def __init__(self, intents: list[IntentDefinition]) -> None:
        self._always_escalate = {intent.name: intent.always_escalate for intent in intents}

    def decide(self, intent: str) -> tuple[AgentAction, str]:
        # An intent name outside the taxonomy (a classifier bug, or output
        # drift) fails safe to escalate rather than defaulting to
        # auto-handle -- silently auto-handling something we don't
        # recognize is the more expensive mistake.
        if intent not in self._always_escalate:
            return AgentAction.ESCALATE, f"unrecognized intent {intent!r}, escalating to be safe"
        escalate = self._always_escalate[intent]
        action = AgentAction.ESCALATE if escalate else AgentAction.AUTO_HANDLE
        reason = f"intents.yaml default for {intent!r}: always_escalate={escalate}"
        return action, reason


class TfidfReplyRetriever:
    """Simple baseline grounding: nearest historical brand reply by TF-IDF
    cosine similarity over the same processed corpus the full system's
    semantic index uses -- same source material, a cheaper matching method."""

    def __init__(
        self, vectorizer: TfidfVectorizer, matrix: object, cases: list[tuple[str, str, str]]
    ) -> None:
        self._vectorizer = vectorizer
        self._matrix = matrix
        self._cases = cases  # (thread_id, customer_problem, brand_reply)

    def retrieve(self, text: str) -> RetrievedExample:
        vector = self._vectorizer.transform([text])
        similarities = cosine_similarity(vector, self._matrix)[0]
        best_index = int(np.argmax(similarities))
        thread_id, customer_problem, brand_reply = self._cases[best_index]
        return RetrievedExample(
            thread_id=thread_id,
            similarity=float(similarities[best_index]),
            customer_problem=customer_problem,
            brand_reply=brand_reply,
            intent="unknown",  # the TF-IDF baseline does not tag corpus intent
        )

    @classmethod
    def build(cls, threads: list[ConversationThread]) -> TfidfReplyRetriever:
        cases = [(t.thread_id, t.customer_turns[0].text, t.brand_turns[0].text) for t in threads]
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([c[1] for c in cases])
        return cls(vectorizer, matrix, cases)


class TrivialBaseline:
    """Majority-class intent, always escalate, one canned reply."""

    def __init__(self, classifier: MajorityIntentClassifier) -> None:
        self._classifier = classifier

    def run(self, text: str) -> AgentResult:
        intent = self._classifier.predict(text)
        return AgentResult(
            intent=intent,
            intent_confidence=1.0,  # deterministic by construction, not a real confidence signal
            reply=_CANNED_REPLY,
            action=AgentAction.ESCALATE,
            escalation_reason="trivial baseline always escalates",
            retrieved_examples=[],
        )

    @classmethod
    def train(cls, dev_examples: list[GoldenExample]) -> TrivialBaseline:
        return cls(MajorityIntentClassifier.train(dev_examples))


class SimpleBaseline:
    """TF-IDF + Logistic Regression intent, static rule escalation, TF-IDF nearest-reply."""

    def __init__(
        self, classifier: TfidfIntentClassifier, escalation: RuleBasedEscalation, retriever: TfidfReplyRetriever
    ) -> None:
        self._classifier = classifier
        self._escalation = escalation
        self._retriever = retriever

    def run(self, text: str) -> AgentResult:
        intent, confidence = self._classifier.predict_with_confidence(text)
        action, reason = self._escalation.decide(intent)
        retrieved = self._retriever.retrieve(text)
        return AgentResult(
            intent=intent,
            intent_confidence=confidence,
            reply=retrieved.brand_reply,
            action=action,
            escalation_reason=reason,
            retrieved_examples=[retrieved],
        )

    @classmethod
    def train(cls, dev_examples: list[GoldenExample], threads: list[ConversationThread], seed: int) -> SimpleBaseline:
        classifier = TfidfIntentClassifier.train(dev_examples, seed)
        escalation = RuleBasedEscalation(load_intents(config.INTENTS_PATH))
        retriever = TfidfReplyRetriever.build(threads)
        return cls(classifier, escalation, retriever)


def _self_check() -> None:
    """Escalation logic self-check -- not a full test suite, just enough to
    catch a broken rule mapping before it silently corrupts every baseline
    escalation decision."""
    from .schemas import IntentDefinition

    intents = [
        IntentDefinition(name="Billing & Subscription", description="", always_escalate=True),
        IntentDefinition(name="Playback / App Technical Issue", description="", always_escalate=False),
    ]
    escalation = RuleBasedEscalation(intents)
    action, _ = escalation.decide("Billing & Subscription")
    assert action == AgentAction.ESCALATE
    action, _ = escalation.decide("Playback / App Technical Issue")
    assert action == AgentAction.AUTO_HANDLE
    action, _ = escalation.decide("Some Unknown Intent")
    assert action == AgentAction.ESCALATE  # unrecognized intents fail safe to escalate
    print("baselines self-check passed")


if __name__ == "__main__":
    _self_check()
