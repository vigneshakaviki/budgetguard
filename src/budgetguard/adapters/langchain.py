"""LangChain adapter -- a callback handler that checks then records.

    from langchain_openai import ChatOpenAI
    from budgetguard import Budget, Governor
    from budgetguard.adapters import LangChainGuard

    gov = Governor(Budget(usd_per_day=5.00, max_loop=10))
    guard = LangChainGuard(gov, agent="researcher")

    llm = ChatOpenAI(model="gpt-4o", callbacks=[guard])
    llm.invoke("hi")   # on_llm_start -> check (may raise); on_llm_end -> record

``check`` runs in ``on_(chat_model|llm)_start``; raising there aborts the call
before the provider is hit. ``on_llm_end`` records usage from the ``LLMResult``.
"""

from __future__ import annotations

from typing import Any

from ..core import Governor


def _load_base_callback_handler():
    for path in (
        "langchain_core.callbacks.base",
        "langchain.callbacks.base",
    ):
        try:
            mod = __import__(path, fromlist=["BaseCallbackHandler"])
            return mod.BaseCallbackHandler
        except Exception:  # pragma: no cover - import-environment dependent
            continue
    raise ImportError(
        "LangChainGuard requires langchain-core. Install with: pip install langchain-core"
    )


class _LangChainGuardImpl:
    def __init__(self, governor: Governor, *, agent: str = "default") -> None:
        self.governor = governor
        self.agent = agent
        # run_id -> signature, so on_llm_end records under the same signature
        # on_*_start checked with (LLMResult carries no prompt to re-derive it).
        self._sig_by_run: dict[Any, str] = {}

    @staticmethod
    def _signature(serialized: Any, prompts: Any) -> str:
        model = ""
        if isinstance(serialized, dict):
            model = serialized.get("name") or serialized.get("id") or ""
        return f"langchain:{model}:{prompts!r}"

    def _start(self, serialized, payload, run_id):
        sig = self._signature(serialized, payload)
        self._sig_by_run[run_id] = sig
        self.governor.check(self.agent, signature=sig)

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        self._start(serialized, prompts, run_id)

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        self._start(serialized, messages, run_id)

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        # response is a LangChain LLMResult; extract_usage reads
        # llm_output["token_usage"]. Record under the start's signature so loop
        # detection counts repeats of the same call.
        sig = self._sig_by_run.pop(run_id, "")
        self.governor.record_result(self.agent, result=response, signature=sig)


def LangChainGuard(governor: Governor, *, agent: str = "default"):
    """Build a LangChain ``BaseCallbackHandler`` enforcing ``governor``'s budget."""
    BaseCallbackHandler = _load_base_callback_handler()

    class _LangChainGuard(_LangChainGuardImpl, BaseCallbackHandler):
        pass

    return _LangChainGuard(governor, agent=agent)
