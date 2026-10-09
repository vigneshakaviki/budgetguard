"""Framework adapters. Each imports its framework lazily, so importing
``budgetguard`` never requires LiteLLM or LangChain to be installed.

    from budgetguard.adapters import LiteLLMGuard, LangChainGuard
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers only; no runtime framework import
    from .langchain import LangChainGuard
    from .litellm import LiteLLMGuard

__all__ = ["LiteLLMGuard", "LangChainGuard"]


def __getattr__(name: str):
    # PEP 562 lazy export: resolve the adapter (and its framework import) only
    # when actually accessed.
    if name == "LiteLLMGuard":
        from .litellm import LiteLLMGuard

        return LiteLLMGuard
    if name == "LangChainGuard":
        from .langchain import LangChainGuard

        return LangChainGuard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
