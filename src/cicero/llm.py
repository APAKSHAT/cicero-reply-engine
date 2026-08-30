"""Thin wrapper around the Anthropic SDK.

Everything the pipeline asks of a model goes through here so that retries,
timeouts, refusals, and token accounting are handled in exactly one place --
and so the whole system can be run offline against `StubLLM` in tests.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Type, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when a call fails in a way the caller must treat as 'no answer'.

    The pipeline turns this into an escalation, never into a guess.
    """


class LLMBackend(ABC):
    """What the pipeline needs from a model: one validated object, or one
    string. Three implementations satisfy it -- Anthropic, OpenRouter, and an
    offline stub -- and they share no behaviour, only this shape.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @abstractmethod
    def structured(self, *, model: str, system: str, user: str,
                   schema: Type[T], max_tokens: int = 4000) -> T:
        """One call, one schema-valid object. Anything else raises LLMError."""

    @abstractmethod
    def text(self, *, model: str, system: str, user: str,
             max_tokens: int = 2000, effort: str = "medium") -> str:
        """One call, one non-empty string. Anything else raises LLMError."""


class AnthropicLLM(LLMBackend):
    def __init__(self, timeout: float = 60.0, max_retries: int = 3):
        # Zero-arg client: resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or
        # an `ant auth login` profile.
        self.client = anthropic.Anthropic(timeout=timeout, max_retries=max_retries)
        self.input_tokens = 0
        self.output_tokens = 0

    def _account(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0

    def structured(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: Type[T],
        max_tokens: int = 4000,
    ) -> T:
        """One call, one validated object. Anything else raises LLMError."""
        try:
            response = self.client.messages.parse(
                model=model,
                max_tokens=max_tokens,
                system=system,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
        except anthropic.RateLimitError as e:
            raise LLMError(f"rate limited after retries: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"api error {e.status_code}: {e}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"connection error: {e}") from e
        except Exception as e:  # noqa: BLE001 - auth, config, serialization
            # Anything else -- a missing key, a bad schema, a serialization
            # failure -- must still surface as "no answer" so the policy layer
            # escalates rather than the pipeline crashing on an unhandled type.
            raise LLMError(f"{type(e).__name__}: {e}") from e

        if response.stop_reason == "refusal":
            raise LLMError(f"model refused: {getattr(response, 'stop_details', None)}")
        if response.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens")

        parsed = response.parsed_output
        if parsed is None:
            raise LLMError("model returned no parseable structured output")
        self._account(response)
        return parsed

    def text(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 2000,
        effort: str = "medium",
    ) -> str:
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.RateLimitError as e:
            raise LLMError(f"rate limited after retries: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"api error {e.status_code}: {e}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"connection error: {e}") from e
        except Exception as e:  # noqa: BLE001 - auth, config, serialization
            # Anything else -- a missing key, a bad schema, a serialization
            # failure -- must still surface as "no answer" so the policy layer
            # escalates rather than the pipeline crashing on an unhandled type.
            raise LLMError(f"{type(e).__name__}: {e}") from e

        if response.stop_reason == "refusal":
            raise LLMError(f"model refused: {getattr(response, 'stop_details', None)}")
        if response.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens")

        self._account(response)
        parts = [b.text for b in response.content if b.type == "text"]
        out = "\n".join(parts).strip()
        if not out:
            raise LLMError("model returned an empty response")
        return out


class StubLLM(LLMBackend):
    """Offline double. Lets the policy layer, scheduler, guardrails, and the
    whole pipeline be tested without a network or a key -- which matters,
    because those are the parts that must be deterministic.

    Responses are returned in the order queued. Every call is recorded on
    `calls`, so a test can assert what the pipeline actually asked for.
    """

    def __init__(self, structured_responses: Optional[list[BaseModel]] = None,
                 text_responses: Optional[list[str]] = None):
        self._structured = list(structured_responses or [])
        self._text = list(text_responses or [])
        self.input_tokens = self.output_tokens = 0
        self.calls: list[tuple[str, str]] = []

    def structured(self, *, model: str, system: str, user: str,
                   schema: Type[T], max_tokens: int = 4000) -> T:
        self.calls.append(("structured", schema.__name__))
        if not self._structured:
            raise LLMError(f"StubLLM: no {schema.__name__} response queued")
        return self._structured.pop(0)  # type: ignore[return-value]

    def text(self, *, model: str, system: str, user: str,
             max_tokens: int = 2000, effort: str = "medium") -> str:
        self.calls.append(("text", model))
        if not self._text:
            raise LLMError("StubLLM: no text response queued")
        return self._text.pop(0)


# ---------------------------------------------------------------------------
# OpenRouter backend
# ---------------------------------------------------------------------------

# Keywords OpenAI-style strict json_schema does not accept. Pydantic emits them
# from Field(ge=..., le=...) and friends. We strip them from the wire schema and
# keep enforcing them locally by validating the response with the same model --
# so a model that returns confidence=1.5 fails validation and escalates, rather
# than being silently accepted.
_UNSUPPORTED_SCHEMA_KEYS = (
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength", "minItems", "maxItems", "pattern", "format",
    "default",
)


def _strictify(node: Any) -> Any:
    """Pydantic JSON schema -> the strict subset OpenAI-compatible APIs accept.

    Strict mode requires every object to declare `additionalProperties: false`
    and to list *all* of its properties in `required` (nullability is expressed
    in the type, not by omission)."""
    if isinstance(node, dict):
        for key in _UNSUPPORTED_SCHEMA_KEYS:
            node.pop(key, None)
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        for value in node.values():
            _strictify(value)
    elif isinstance(node, list):
        for value in node:
            _strictify(value)
    return node


class OpenRouterLLM(LLMBackend):
    """OpenAI-compatible backend, for running the pipeline through OpenRouter.

    Same two methods as `LLM`, so nothing downstream changes. Anthropic-only
    features (adaptive thinking, effort, refusal stop reasons) have no
    equivalent here and are simply not sent -- `effort` is accepted and ignored
    so the call sites stay identical.
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, timeout: float = 90.0, max_retries: int = 3):
        from openai import OpenAI

        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        self.client = OpenAI(base_url=self.BASE_URL, api_key=key,
                             timeout=timeout, max_retries=max_retries)
        self.input_tokens = 0
        self.output_tokens = 0
        # The autorouter serves a different model per call; record which one
        # actually answered so a run stays attributable after the fact.
        self.models_used: dict[str, int] = {}

    def _account(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        # The autorouter picks a different model per call; record what actually
        # answered so the run is attributable after the fact.
        served = getattr(response, "model", None)
        if served:
            self.models_used[served] = self.models_used.get(served, 0) + 1

    def _call(self, *, model: str, system: str, user: str, max_tokens: int,
              response_format: Optional[dict] = None) -> str:
        import openai as _openai

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        if response_format:
            kwargs["response_format"] = response_format
        try:
            response = self.client.chat.completions.create(**kwargs)
        except _openai.RateLimitError as e:
            raise LLMError(f"rate limited after retries: {e}") from e
        except _openai.APIStatusError as e:
            raise LLMError(f"api error {e.status_code}: {e}") from e
        except _openai.APIConnectionError as e:
            raise LLMError(f"connection error: {e}") from e
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"{type(e).__name__}: {e}") from e

        if not response.choices:
            raise LLMError("provider returned no choices")
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMError(f"model refused: {choice.message.refusal}")
        if choice.finish_reason == "length":
            raise LLMError("response truncated at max_tokens")

        self._account(response)
        content = choice.message.content or ""
        if not content.strip():
            raise LLMError("model returned an empty response")
        return content.strip()

    def structured(self, *, model: str, system: str, user: str,
                   schema: Type[T], max_tokens: int = 4000) -> T:
        wire = _strictify(schema.model_json_schema())
        raw = self._call(
            model=model, system=system, user=user, max_tokens=max_tokens,
            response_format={"type": "json_schema", "json_schema": {
                "name": schema.__name__, "strict": True, "schema": wire}})
        try:
            return schema.model_validate_json(raw)
        except ValidationError as e:
            # Local validation is the real gate -- see _strictify.
            raise LLMError(f"response failed schema validation: {e}") from e
        except Exception as e:  # noqa: BLE001 - malformed JSON
            raise LLMError(f"could not parse structured output: {e}") from e

    def text(self, *, model: str, system: str, user: str,
             max_tokens: int = 2000, effort: str = "medium") -> str:
        return self._call(model=model, system=system, user=user,
                          max_tokens=max_tokens)


def default_llm() -> LLMBackend:
    """Pick a backend. OpenRouter wins when its key is present."""
    if os.getenv("CICERO_OFFLINE"):
        return StubLLM()
    if os.getenv("OPENROUTER_API_KEY"):
        return OpenRouterLLM()
    return AnthropicLLM()
