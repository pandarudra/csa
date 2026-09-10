"""Thin, reusable wrapper around NVIDIA NIM's OpenAI-compatible chat/embedding API.

This module knows nothing about intents, escalation rules, or evaluation --
it only knows how to send text/JSON/embedding requests reliably. Business
logic (prompts, taxonomies, decision rules) lives in the modules that call
this one (pipeline.py, baselines.py, eval_harness.py, etc.).

Provider note: this project originally targeted Gemini, but Gemini's
free-tier generateContent quota turned out to be 5 requests/minute for the
configured key -- unworkable for an eval harness needing a few hundred
chat calls per run. NVIDIA's build.nvidia.com NIM catalog exposes hosted
open models through a standard OpenAI-compatible API with a much more
usable free tier (see DECISIONS.md), so `openai.OpenAI` pointed at NIM's
base URL is the client here rather than a NIM-specific SDK.
"""
from __future__ import annotations

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Callable, Literal, TypeVar, get_args, get_origin

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

from . import config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
I = TypeVar("I")
R = TypeVar("R")

_MAX_RETRIES = 4
_BACKOFF_BASE_SECONDS = 2.0
_REQUEST_TIMEOUT_SECONDS = 45.0


def _example_instance(model: type[BaseModel]) -> dict[str, object]:
    """Build a placeholder-filled example dict for `model`, for prompting.

    Only needs to cover the field types our LLM-generated schemas actually
    use (str, float, int, bool, Enum, Literal) -- not a general-purpose
    JSON Schema example generator.
    """
    return {name: _placeholder_for(field.annotation) for name, field in model.model_fields.items()}


def _placeholder_for(annotation: object) -> object:
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return next(iter(annotation)).value
    if get_origin(annotation) is Literal:
        return get_args(annotation)[0]
    if annotation is bool:
        return True
    if annotation is int:
        return 3
    if annotation is float:
        return 0.9
    return "..."


class NvidiaConfigError(RuntimeError):
    """Raised when NVIDIA_API_KEY is missing -- fails at first use, not at import."""


class NvidiaRequestError(RuntimeError):
    """Raised when a request exhausts retries or returns unusable output."""


class NvidiaClient:
    """Lazily-initialized NIM client with retry/backoff on transient errors.

    Construction never touches the network, so importing this module (or
    code that imports it) works fine without NVIDIA_API_KEY set -- only
    methods that actually call the API need the key, and they raise a clear
    error naming the missing variable instead of a confusing SDK trace.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or config.NVIDIA_API_KEY
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self._api_key:
                raise NvidiaConfigError(
                    "NVIDIA_API_KEY is not set. Copy .env.example to .env and fill in your key."
                )
            self._client = OpenAI(
                base_url=config.NVIDIA_BASE_URL,
                api_key=self._api_key,
                timeout=_REQUEST_TIMEOUT_SECONDS,
                max_retries=0,  # we handle retries ourselves, see _with_retries
            )
        return self._client

    def generate_text(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
        thinking: bool = False,
    ) -> str:
        """Free-form text generation. Returns the response text, never None."""
        content = self._chat(prompt, system_instruction=system_instruction, temperature=temperature, thinking=thinking)
        if not content:
            raise NvidiaRequestError("NIM returned an empty text response")
        return content

    def generate_json(
        self,
        prompt: str,
        response_model: type[T],
        *,
        system_instruction: str | None = None,
        temperature: float = 0.0,
        thinking: bool = False,
    ) -> T:
        """Structured generation validated against `response_model`.

        NIM's JSON mode (response_format={"type": "json_object"}) guarantees
        syntactically valid JSON but not conformance to a specific schema --
        unlike Gemini's response_schema, there is no server-side shape
        enforcement here. The result is validated with pydantic; a
        malformed or out-of-taxonomy response raises NvidiaRequestError
        rather than being silently accepted (the assignment's requirement,
        not just a nicety -- see spec section 15).

        The model is shown a concrete example *instance* rather than the
        formal JSON Schema document: passing model_json_schema() directly
        (with its "properties"/"required"/"title" wrapper) tends to make
        this model echo the schema's own structure back instead of
        producing a value that conforms to it.
        """
        example = json.dumps(_example_instance(response_model))
        full_prompt = (
            f"{prompt}\n\n"
            f"Respond with a single JSON object shaped exactly like this example "
            f"(same keys, your own values), and nothing else -- no markdown fences, "
            f"no commentary:\n{example}"
        )
        content = self._chat(
            full_prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            thinking=thinking,
            json_mode=True,
        )
        try:
            payload = json.loads(content)
            return response_model.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise NvidiaRequestError(
                f"NIM response did not match {response_model.__name__}: {content!r}"
            ) from exc

    def embed(self, texts: list[str], *, input_type: str = "query") -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input, same order.

        `input_type` selects the embedding model's asymmetric mode: "query"
        for a live customer message being searched with, "passage" for
        historical text being indexed -- using the wrong one measurably
        hurts retrieval quality for QA-style embedding models.
        """
        if not texts:
            return []
        client = self._get_client()

        def call() -> list[list[float]]:
            response = client.embeddings.create(
                model=config.NVIDIA_EMBEDDING_MODEL,
                input=texts,
                extra_body={"input_type": input_type},
            )
            return [list(item.embedding) for item in response.data]

        vectors = self._with_retries(call, operation="embed")
        if len(vectors) != len(texts):
            raise NvidiaRequestError(f"Expected {len(texts)} embeddings, got {len(vectors)}")
        return vectors

    def _chat(
        self,
        prompt: str,
        *,
        system_instruction: str | None,
        temperature: float,
        thinking: bool,
        json_mode: bool = False,
    ) -> str:
        client = self._get_client()
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        def call() -> str:
            response = client.chat.completions.create(
                model=config.NVIDIA_CHAT_MODEL,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"} if json_mode else None,
                extra_body={"chat_template_kwargs": {"thinking": thinking}},
            )
            return response.choices[0].message.content or ""

        return self._with_retries(call, operation="chat")

    def _with_retries(self, call: Callable[[], R], *, operation: str) -> R:
        """Retry transient failures (429, timeout, connection) with backoff.

        4xx errors other than 429 (bad request, auth, not-found model) are
        never retried -- they will fail identically every time and just
        waste the retry budget.
        """
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return call()
            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                last_exc = exc
                if attempt >= _MAX_RETRIES:
                    break
                delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 1)
                logger.warning(
                    "NIM %s attempt %d/%d failed (%s); retrying in %.1fs",
                    operation, attempt, _MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
            except APIStatusError as exc:
                raise NvidiaRequestError(f"NIM {operation} request rejected: {exc}") from exc
        raise NvidiaRequestError(f"NIM {operation} failed after {_MAX_RETRIES} attempts: {last_exc}") from last_exc


def run_concurrently(fn: Callable[[I], R], items: list[I]) -> list[R]:
    """Run `fn` over `items` with bounded concurrency, preserving input order.

    Bounded by `config.NVIDIA_MAX_CONCURRENT_REQUESTS` -- empirically this
    account tolerates a handful of simultaneous requests but starts
    throwing 429s/timeouts well before 8-way concurrency, so higher-level
    code (eval_harness, retrieval index building) should route batches of
    independent API calls through this rather than a raw thread pool.
    """
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=config.NVIDIA_MAX_CONCURRENT_REQUESTS) as executor:
        return list(executor.map(fn, items))
