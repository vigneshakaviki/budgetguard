"""Usage autodetection across provider/framework response shapes."""

from __future__ import annotations

import pytest

from budgetguard import Budget, Governor, extract_usage


class Obj:
    """Tiny attribute bag to fake SDK response objects."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_openai_shape():
    resp = Obj(model="gpt-4o", usage=Obj(prompt_tokens=100, completion_tokens=50))
    u = extract_usage(resp)
    assert (u.model, u.input_tokens, u.output_tokens) == ("gpt-4o", 100, 50)


def test_anthropic_shape():
    resp = Obj(model="claude-sonnet-4", usage=Obj(input_tokens=200, output_tokens=80))
    u = extract_usage(resp)
    assert (u.input_tokens, u.output_tokens) == (200, 80)


def test_otel_semconv_dict():
    span = {
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.usage.input_tokens": 10,
        "gen_ai.usage.output_tokens": 5,
    }
    u = extract_usage(span)
    assert (u.model, u.input_tokens, u.output_tokens) == ("gpt-4o-mini", 10, 5)


def test_langchain_llmresult_shape():
    resp = Obj(llm_output={"model_name": "gpt-4o", "token_usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    u = extract_usage(resp)
    assert (u.model, u.input_tokens, u.output_tokens) == ("gpt-4o", 7, 3)


def test_generic_cost_dict():
    u = extract_usage({"cost": 0.25})
    assert u.usd == 0.25


def test_unknown_shape_returns_none():
    assert extract_usage("just a string") is None
    assert extract_usage(None) is None


def test_decorator_autodetects_without_usage_lambda():
    gov = Governor(Budget(usd_total=1.00))

    @gov.guard(agent="w")  # no usage= -> autodetect
    def call():
        return Obj(model="gpt-4o-mini", usage=Obj(prompt_tokens=1000, completion_tokens=1000))

    call()
    assert gov.spent("w") == pytest.approx(0.00075)  # gpt-4o-mini: 0.15 + 0.60 /Mtok
