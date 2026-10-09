"""LiteLLM adapter -- one hook, 100+ providers.

LiteLLM is the gateway under CrewAI, DSPy, GraphRAG and many more, so a single
``CustomLogger`` subclass enforces budgets across every model and framework that
routes through it.

SDK usage (``import litellm``)::

    import litellm
    from budgetguard import Budget, Governor
    from budgetguard.adapters import LiteLLMGuard

    gov = Governor(Budget(usd_per_day=5.00, max_loop=10))
    litellm.callbacks = [LiteLLMGuard(gov)]

    litellm.completion(model="gpt-4o", messages=[...])  # checked + recorded

Proxy usage: set ``callbacks: budgetguard_litellm.handler`` in ``config.yaml``
pointing at a module-level ``handler = LiteLLMGuard(gov)``. The proxy
``async_pre_call_hook`` rejects the request *before* it is sent once a cap trips.

Agent key resolution per call: ``metadata["budgetguard_agent"]`` ->
``metadata["user"]`` -> the governor's configured ``default_agent``.
"""

from __future__ import annotations

from typing import Any

from ..core import Governor
from ..errors import GuardTripped


def _load_custom_logger():
    try:
        from litellm.integrations.custom_logger import CustomLogger
    except Exception as exc:  # pragma: no cover - import-environment dependent
        raise ImportError(
            "LiteLLMGuard requires litellm. Install with: pip install litellm"
        ) from exc
    return CustomLogger


class _LiteLLMGuardImpl:
    """Mixin holding the budgetguard logic, mixed onto CustomLogger below."""

    def __init__(self, governor: Governor, *, default_agent: str = "default") -> None:
        self.governor = governor
        self.default_agent = default_agent

    # -- agent + signature helpers --------------------------------------

    def _agent(self, kwargs: dict[str, Any]) -> str:
        meta = (kwargs or {}).get("litellm_params", {}).get("metadata") or kwargs.get(
            "metadata"
        ) or {}
        return meta.get("budgetguard_agent") or meta.get("user") or self.default_agent

    @staticmethod
    def _signature(model: Any, messages: Any) -> str:
        # Loop signature = model + the last user message (cheap, stable enough
        # to catch an agent re-asking the identical thing in a loop).
        last = ""
        if isinstance(messages, list) and messages:
            last = str(messages[-1])
        return f"litellm:{model}:{last}"

    # -- SDK callback hooks ---------------------------------------------

    def log_pre_api_call(self, model, messages, kwargs):
        self.governor.check(self._agent(kwargs), signature=self._signature(model, messages))

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj)

    def _record(self, kwargs, response_obj):
        agent = self._agent(kwargs)
        sig = self._signature(kwargs.get("model"), kwargs.get("messages"))
        # LiteLLM computes cost and attaches it; prefer that over re-pricing.
        cost = (kwargs or {}).get("response_cost")
        if cost is None:
            hidden = (kwargs or {}).get("standard_logging_object") or {}
            cost = hidden.get("response_cost")
        if cost is not None:
            self.governor.record(agent, usd=float(cost), signature=sig)
        else:
            self.governor.record_result(agent, result=response_obj, signature=sig)

    # -- Proxy pre-call reject hook -------------------------------------

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        agent = self._agent(data or {})
        sig = self._signature((data or {}).get("model"), (data or {}).get("messages"))
        try:
            self.governor.check(agent, signature=sig)
        except GuardTripped as e:
            # Reject the request at the proxy with HTTP 429.
            try:
                from fastapi import HTTPException

                raise HTTPException(status_code=429, detail=str(e)) from e
            except ImportError:
                raise
        return data


def LiteLLMGuard(governor: Governor, *, default_agent: str = "default"):
    """Build a LiteLLM ``CustomLogger`` that enforces ``governor``'s budget."""
    CustomLogger = _load_custom_logger()

    class _LiteLLMGuard(_LiteLLMGuardImpl, CustomLogger):
        def __init__(self, gov, agent):
            CustomLogger.__init__(self)
            _LiteLLMGuardImpl.__init__(self, gov, default_agent=agent)

    return _LiteLLMGuard(governor, default_agent)
