"""Adapter logic, exercised via the framework-free mixin classes.

We avoid importing real litellm/langchain by testing ``_LiteLLMGuardImpl`` and
``_LangChainGuardImpl`` directly -- their hook methods need only a Governor.
"""

from __future__ import annotations

import pytest

from budgetguard import Budget, BudgetExceeded, Governor, LoopDetected
from budgetguard.adapters.langchain import _LangChainGuardImpl
from budgetguard.adapters.litellm import _LiteLLMGuardImpl


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# -- LiteLLM --------------------------------------------------------------


def test_litellm_check_and_record_flow():
    gov = Governor(Budget(usd_total=1.00, max_calls=2))
    h = _LiteLLMGuardImpl(gov, default_agent="svc")

    kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    h.log_pre_api_call("gpt-4o", kwargs["messages"], kwargs)
    # LiteLLM attaches response_cost on kwargs -> we prefer it
    h.log_success_event({**kwargs, "response_cost": 0.40}, Obj(), 0, 1)
    assert gov.spent("svc") == pytest.approx(0.40)


def test_litellm_agent_from_metadata():
    gov = Governor(Budget(max_calls=1))
    h = _LiteLLMGuardImpl(gov, default_agent="svc")
    kwargs = {"model": "gpt-4o", "messages": [], "metadata": {"budgetguard_agent": "alice"}}
    h.log_pre_api_call("gpt-4o", [], kwargs)
    h.log_success_event(kwargs, Obj(), 0, 1)
    assert gov.spent("alice") >= 0.0
    with pytest.raises(BudgetExceeded):
        h.log_pre_api_call("gpt-4o", [], kwargs)  # alice hit max_calls=1


def test_litellm_records_via_autodetect_when_no_cost():
    gov = Governor(Budget(usd_total=10))
    h = _LiteLLMGuardImpl(gov, default_agent="svc")
    resp = Obj(model="gpt-4o-mini", usage=Obj(prompt_tokens=1000, completion_tokens=1000))
    h.log_success_event({"model": "gpt-4o-mini", "messages": []}, resp, 0, 1)
    assert gov.spent("svc") == pytest.approx(0.00075)


# -- LangChain ------------------------------------------------------------


def test_langchain_check_and_record_flow():
    gov = Governor(Budget(usd_total=1.00))
    h = _LangChainGuardImpl(gov, agent="researcher")
    h.on_chat_model_start({"name": "ChatOpenAI"}, [[{"content": "hi"}]])
    resp = Obj(llm_output={"model_name": "gpt-4o", "token_usage": {"prompt_tokens": 1000, "completion_tokens": 0}})
    h.on_llm_end(resp)
    assert gov.spent("researcher") == pytest.approx(0.0025)  # 1000 in * 2.5/Mtok


def test_langchain_loop_detection_same_prompt():
    gov = Governor(Budget(max_loop=2))
    h = _LangChainGuardImpl(gov, agent="r")
    for _ in range(2):
        h.on_llm_start({"name": "LLM"}, ["same prompt"])
        h.on_llm_end(Obj(llm_output={}))
    with pytest.raises(LoopDetected):
        h.on_llm_start({"name": "LLM"}, ["same prompt"])
