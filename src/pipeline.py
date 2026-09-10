"""The end-to-end SpotifyCares support agent.

Input -> intent classification -> historical retrieval -> grounded reply
drafting -> escalation decision -> structured AgentResult. See
src/schemas.py for the AgentResult shape and src/retrieve.py for how
historical grounding is found.
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from pydantic import BaseModel

from . import config
from .llm_client import NvidiaClient, NvidiaRequestError
from .retrieve import HistoricalIndex
from .schemas import AgentAction, AgentResult, EscalationDecision, IntentClassification, IntentDefinition, RetrievedExample
from .utils import load_intents

logger = logging.getLogger(__name__)

_FALLBACK_INTENT = "General Complaint / Praise / Other"


class _ReplyDraft(BaseModel):
    reply: str


def _format_context_block(label: str, thread_context: str) -> str:
    return f"{label}:\n{thread_context}\n\n" if thread_context else ""


_MENTION_RE = re.compile(r"@\w+")
_NON_WORD_RE = re.compile(r"[^\w\s]")
_ECHO_SIMILARITY_THRESHOLD = 0.8


def _looks_like_echo(reply: str, customer_message: str) -> bool:
    """True if `reply` is essentially the customer's own message restated.

    The JSON-schema-constrained drafting call (see `_draft_reply`) made
    this failure mode rare but did not eliminate it -- a short, simple
    customer message (e.g. "can you add the new Drake album please") can
    still get echoed back verbatim with only an "@SpotifyCares" prefix
    added. Comparing normalized text (mentions and punctuation stripped)
    catches that case without being thrown off by the prefix itself.
    """

    def normalize(text: str) -> str:
        return _NON_WORD_RE.sub("", _MENTION_RE.sub("", text)).lower().strip()

    normalized_reply, normalized_message = normalize(reply), normalize(customer_message)
    if not normalized_reply or not normalized_message:
        return False
    return SequenceMatcher(None, normalized_reply, normalized_message).ratio() > _ECHO_SIMILARITY_THRESHOLD


class SupportAgent:
    """Holds config (taxonomy, retrieval index, LLM client); each `handle()`
    call is independent -- conversation memory is whatever the caller
    passes in as `thread_context`, not held internally.
    """

    def __init__(self, intents: list[IntentDefinition], index: HistoricalIndex, client: NvidiaClient) -> None:
        self._intents = intents
        self._intents_by_name = {intent.name: intent for intent in intents}
        self._index = index
        self._client = client

    @classmethod
    def load(cls) -> SupportAgent:
        intents = load_intents(config.INTENTS_PATH)
        index = HistoricalIndex.load(config.PROCESSED_DATA_PATH.parent / "retrieval_index")
        return cls(intents, index, NvidiaClient())

    def handle(self, customer_message: str, thread_context: str = "") -> AgentResult:
        """Run the full pipeline on one customer message."""
        classification = self._classify(customer_message, thread_context)
        retrieved = self._index.search(
            customer_message, self._client, config.RETRIEVAL_TOP_K, intent_filter=classification.intent
        )
        reply = self._draft_reply(customer_message, thread_context, classification.intent, retrieved)
        action, reason = self._decide_escalation(customer_message, classification, retrieved)
        return AgentResult(
            intent=classification.intent,
            intent_confidence=classification.confidence,
            reply=reply,
            action=action,
            escalation_reason=reason,
            retrieved_examples=retrieved,
        )

    def _classify(self, message: str, thread_context: str) -> IntentClassification:
        intent_lines = "\n".join(f"- {intent.name}: {intent.description.strip()}" for intent in self._intents)
        prompt = (
            f"Classify this SpotifyCares customer message into exactly one of these intents:\n{intent_lines}\n\n"
            f"{_format_context_block('Conversation so far', thread_context)}"
            f"Customer message: {message!r}\n\n"
            "Give a confidence between 0 and 1 for how sure you are."
        )
        try:
            result = self._client.generate_json(prompt, IntentClassification, thinking=False)
        except NvidiaRequestError as exc:
            logger.warning("Classification request failed (%s); falling back to %r", exc, _FALLBACK_INTENT)
            return IntentClassification(intent=_FALLBACK_INTENT, confidence=0.0)
        if result.intent not in self._intents_by_name:
            # Spec requires explicit handling of an out-of-taxonomy label,
            # not silent acceptance: fall back to a named default and mark
            # confidence 0 so downstream escalation logic treats it as
            # untrustworthy rather than as a real classification.
            logger.warning("Classifier returned out-of-taxonomy intent %r; falling back", result.intent)
            return IntentClassification(intent=_FALLBACK_INTENT, confidence=0.0)
        return result

    def _draft_reply(
        self, message: str, thread_context: str, intent: str, retrieved: list[RetrievedExample]
    ) -> str:
        # Numbered, prose-style grounding rather than a repeated
        # "Customer: X / SpotifyCares replied: Y" block: the latter reliably
        # made the model pattern-complete the template with the *new*
        # customer's own message instead of answering it (see
        # DECISIONS.md) -- four repetitions of a two-line dialogue format
        # right before "now write the next one" is exactly the shape that
        # invites continuation instead of a fresh response.
        examples_block = "\n".join(
            f"{i}. A customer had a similar issue ({example.customer_problem!r}) and SpotifyCares told "
            f"them: {example.brand_reply!r}"
            for i, example in enumerate(retrieved, start=1)
        ) or "No closely similar historical case was found."
        base_prompt = (
            f"You are drafting a public Twitter reply as SpotifyCares support, for a message classified as "
            f"{intent!r}.\n\nHow similar issues were handled before:\n{examples_block}\n\n"
            f"{_format_context_block('Conversation so far', thread_context)}"
            f"Customer message: {message!r}\n\n"
            "Write a concise, on-brand reply grounded in how the similar cases above were actually handled. "
            "Do not invent policies, refund amounts, timelines, or account-specific facts you don't know. "
            "Do not claim to have taken an action you have not taken. If the historical examples don't give "
            "you enough to safely resolve this, ask a clarifying question or say you'll need to look into "
            "their account rather than guessing. The reply must respond to the customer, not restate their "
            "own message back to them."
        )
        correction = (
            "\n\nYour previous attempt just repeated the customer's own message back to them -- that is not "
            "a reply. Write an actual response: acknowledge the issue and say what you or the customer "
            "should do next."
        )
        # Structured JSON output (not free text): a free-text version of this
        # prompt occasionally echoed the customer's own message verbatim
        # instead of answering it -- forcing a {"reply": ...} JSON shape
        # made that rare but did not eliminate it (see DECISIONS.md), so a
        # runtime echo check backs up the prompt with one corrective retry
        # rather than trusting instruction-following alone. thinking=False
        # matches classify/escalate: a reasoning pass here measurably
        # increases latency/timeout risk against the eval harness's
        # 15-minute reproduction budget without a clear quality win in
        # testing.
        prompt = base_prompt
        for attempt in range(2):
            try:
                reply = self._client.generate_json(prompt, _ReplyDraft, temperature=0.4, thinking=False).reply
            except NvidiaRequestError as exc:
                logger.warning("Reply drafting failed (%s); falling back to a generic acknowledgement", exc)
                return "Thanks for reaching out -- we're looking into this and will follow up with more details."
            if not _looks_like_echo(reply, message):
                return reply
            logger.warning("Drafted reply echoed the customer's message (attempt %d); retrying", attempt + 1)
            prompt = base_prompt + correction
        return "Thanks for reaching out -- we're looking into this and will follow up with more details."

    def _decide_escalation(
        self, message: str, classification: IntentClassification, retrieved: list[RetrievedExample]
    ) -> tuple[AgentAction, str]:
        intent_def = self._intents_by_name.get(classification.intent)
        default_escalate = intent_def.always_escalate if intent_def else True
        best_similarity = max((example.similarity for example in retrieved), default=0.0)
        prompt = (
            f"An intent classifier labeled this message {classification.intent!r} (confidence "
            f"{classification.confidence:.2f}); that intent's default policy is "
            f"{'escalate' if default_escalate else 'auto-handle'}. The best matching historical case had "
            f"similarity {best_similarity:.2f} on a 0-1 scale.\n\n"
            f"Customer message: {message!r}\n\n"
            "Decide auto_handle or escalate. Override the default policy if the message shows real "
            "anger or urgency, asks for a specific promise (a refund amount, a timeline, compensation), or "
            "the historical evidence is too weak (similarity well below 0.3) to safely resolve without "
            "guessing. State one concise, concrete reason."
        )
        try:
            decision = self._client.generate_json(prompt, EscalationDecision, thinking=False)
            return decision.action, decision.reason
        except NvidiaRequestError as exc:
            logger.warning("Escalation review failed (%s); falling back to the intent's static default", exc)
            action = AgentAction.ESCALATE if default_escalate else AgentAction.AUTO_HANDLE
            return action, f"LLM review unavailable; used intents.yaml default (always_escalate={default_escalate})"
