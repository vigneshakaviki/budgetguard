"""Universal usage extraction: sniff tokens + model from any LLM response.

The goal is "no lambda required" -- ``@gov.guard(agent="x")`` with no ``usage=``
should still account cost for common providers. We recognise, in order:

1. OpenAI chat/responses: ``resp.usage.prompt_tokens`` / ``.completion_tokens``
2. Anthropic messages: ``resp.usage.input_tokens`` / ``.output_tokens``
3. OpenTelemetry GenAI semconv dict/span attrs: ``gen_ai.usage.input_tokens`` ...
4. LangChain ``LLMResult.llm_output["token_usage"]``
5. Generic dicts/objects with ``input_tokens`` / ``output_tokens`` / ``cost`` / ``usd``

Returns an :class:`Usage` (model + token counts), or ``None`` if nothing is
recognised -- in which case the call still counts toward call/rate/loop caps but
records zero USD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# OpenTelemetry GenAI semantic-convention attribute names (stable since v1.37).
OTEL_MODEL_KEYS = ("gen_ai.request.model", "gen_ai.response.model")
OTEL_INPUT_KEYS = ("gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens")
OTEL_OUTPUT_KEYS = ("gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens")


@dataclass(frozen=True)
class Usage:
    """Normalised usage pulled from a response."""

    model: str | None
    input_tokens: int
    output_tokens: int
    usd: float | None = None  # set when the source already knows the cost


def _get(obj: Any, name: str) -> Any:
    """Read ``name`` from an attribute or a mapping key; None if absent."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _first(obj: Any, keys: tuple[str, ...]) -> Any:
    for k in keys:
        v = _get(obj, k)
        if v is not None:
            return v
    return None


def _int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def extract_usage(result: Any) -> Usage | None:
    """Best-effort normalise any LLM response into a :class:`Usage`."""
    if result is None:
        return None

    # (3) OTel semconv attrs on a dict or span-like object.
    otel_in = _first(result, OTEL_INPUT_KEYS)
    otel_out = _first(result, OTEL_OUTPUT_KEYS)
    if otel_in is not None or otel_out is not None:
        return Usage(
            model=_first(result, OTEL_MODEL_KEYS),
            input_tokens=_int(otel_in),
            output_tokens=_int(otel_out),
        )

    model = _get(result, "model")

    # (1)/(2) Provider SDK objects expose a ``usage`` sub-object.
    usage = _get(result, "usage")
    if usage is not None:
        # OpenAI uses prompt/completion; Anthropic uses input/output.
        inp = _first(usage, ("prompt_tokens", "input_tokens"))
        out = _first(usage, ("completion_tokens", "output_tokens"))
        if inp is not None or out is not None:
            return Usage(model=model, input_tokens=_int(inp), output_tokens=_int(out))

    # (4) LangChain LLMResult: llm_output["token_usage"].
    llm_output = _get(result, "llm_output")
    if llm_output is not None:
        tu = _get(llm_output, "token_usage") or {}
        inp = _first(tu, ("prompt_tokens", "input_tokens"))
        out = _first(tu, ("completion_tokens", "output_tokens"))
        if inp is not None or out is not None:
            return Usage(
                model=model or _get(llm_output, "model_name"),
                input_tokens=_int(inp),
                output_tokens=_int(out),
            )

    # (5) Generic flat dict/object.
    cost = _first(result, ("cost", "usd", "response_cost"))
    inp = _first(result, ("input_tokens", "prompt_tokens"))
    out = _first(result, ("output_tokens", "completion_tokens"))
    if cost is not None:
        return Usage(model=model, input_tokens=_int(inp), output_tokens=_int(out), usd=float(cost))
    if inp is not None or out is not None:
        return Usage(model=model, input_tokens=_int(inp), output_tokens=_int(out))

    return None
