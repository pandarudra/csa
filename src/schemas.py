"""Typed data contracts shared across the pipeline.

Keeping these in one place means data_prep, retrieve, pipeline, and
eval_harness agree on field names and validation rules without importing
each other's internals. Pydantic (already a transitive dependency of
google-genai) gives us free JSON (de)serialization plus validation of LLM
output, which matters most for `IntentClassification` and `EscalationDecision`
-- a malformed or out-of-taxonomy value from the model must fail loudly
rather than be silently coerced into something plausible-looking.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class Speaker(str, Enum):
    CUSTOMER = "customer"
    BRAND = "brand"


class Turn(BaseModel):
    """One tweet in a reconstructed conversation."""

    tweet_id: str
    text: str
    created_at: str


class SpeakerTurn(BaseModel):
    """A `Turn` tagged with who said it, in conversation order."""

    speaker: Speaker
    tweet_id: str
    text: str
    created_at: str


class ConversationThread(BaseModel):
    """A single customer<->brand conversation reconstructed from TWCS.

    `turns` is the ordered, speaker-tagged view -- the source of truth, used
    for context (an LLM prompt needs to know who said what, and in what
    order). `customer_turns` / `brand_turns` are the flat views the
    assignment asks for; they are computed from `turns` rather than stored
    independently, so the two representations can never disagree.
    """

    thread_id: str
    created_at: str
    turns: list[SpeakerTurn]

    @computed_field  # type: ignore[misc]
    @property
    def customer_turns(self) -> list[Turn]:
        return [Turn(**t.model_dump(exclude={"speaker"})) for t in self.turns if t.speaker == Speaker.CUSTOMER]

    @computed_field  # type: ignore[misc]
    @property
    def brand_turns(self) -> list[Turn]:
        return [Turn(**t.model_dump(exclude={"speaker"})) for t in self.turns if t.speaker == Speaker.BRAND]


class IntentDefinition(BaseModel):
    """One entry in intents.yaml."""

    name: str
    description: str
    examples: list[str] = Field(default_factory=list)
    always_escalate: bool = False


class IntentClassification(BaseModel):
    """Structured output of the intent classifier (LLM or baseline)."""

    intent: str
    confidence: float = Field(ge=0.0, le=1.0)


class RetrievalCase(BaseModel):
    """One (customer problem, brand reply) pair persisted in the retrieval
    index -- the on-disk record `RetrievedExample` search results are drawn
    from. Kept separate from `RetrievedExample` because a case has no
    similarity score until it's matched against a specific query.
    """

    thread_id: str
    customer_problem: str
    brand_reply: str
    intent: str


class RetrievedExample(BaseModel):
    """One historical (customer problem, brand reply) pair used as grounding.

    Every instance must trace back to a real row in the processed corpus --
    `thread_id` is what makes that traceable, and retrieval code must never
    construct one of these from anything but an actual indexed example.
    """

    thread_id: str
    similarity: float
    customer_problem: str
    brand_reply: str
    intent: str


class AgentAction(str, Enum):
    AUTO_HANDLE = "auto_handle"
    ESCALATE = "escalate"


class EscalationDecision(BaseModel):
    action: AgentAction
    reason: str


class AgentResult(BaseModel):
    """Full structured output of one pipeline run."""

    intent: str
    intent_confidence: float = Field(ge=0.0, le=1.0)
    reply: str
    action: AgentAction
    escalation_reason: str
    retrieved_examples: list[RetrievedExample] = Field(default_factory=list)


class LabelSource(str, Enum):
    """Provenance of a golden-set label, for the hybrid seed+assisted workflow."""

    HUMAN = "human"
    LLM_SUGGESTED_HUMAN_CONFIRMED = "llm_suggested_human_confirmed"
    LLM_SUGGESTED_HUMAN_EDITED = "llm_suggested_human_edited"


class GoldenExample(BaseModel):
    """One row of the golden evaluation set."""

    thread_id: str
    customer_text: str
    intent: str
    action: AgentAction
    reason: str
    label_source: LabelSource
    llm_suggested_intent: str | None = None
    llm_suggested_action: AgentAction | None = None
    split: Literal["dev", "test"] | None = None


class JudgeScore(BaseModel):
    """LLM-as-judge (or human) score on the reply-quality rubric, 1-5 each."""

    groundedness: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    helpfulness: int = Field(ge=1, le=5)
    brand_voice: int = Field(ge=1, le=5)
    actionability: int = Field(ge=1, le=5)
    no_unsupported_claims: int = Field(ge=1, le=5)

    @computed_field  # type: ignore[misc]
    @property
    def overall(self) -> float:
        values = [
            self.groundedness,
            self.correctness,
            self.helpfulness,
            self.brand_voice,
            self.actionability,
            self.no_unsupported_claims,
        ]
        return sum(values) / len(values)
