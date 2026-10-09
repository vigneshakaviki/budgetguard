"""budgetguard -- drop-in budget, rate, and loop guard for AI agents.

Quick start::

    from budgetguard import Budget, Governor

    gov = Governor(Budget(usd_per_day=5.00, max_calls=200, max_loop=10))

    @gov.guard(agent="researcher", usage=lambda r: ("gpt-4o", r.in_tok, r.out_tok))
    def call_llm(messages): ...

    # call_llm raises BudgetExceeded / RateLimited / LoopDetected before
    # spending once a cap is hit.

See :class:`Budget` for the caps and :class:`Governor` for enforcement.
"""

from __future__ import annotations

from .budget import Budget
from .core import Governor
from .errors import (
    AgentExpired,
    BudgetExceeded,
    GuardTripped,
    LoopDetected,
    RateLimited,
)
from .pricing import ModelPrice, cost_of, load_prices, price_for
from .store import AgentState, MemoryStore, RedisStore, Store
from .usage import Usage, extract_usage

__version__ = "0.2.0"

__all__ = [
    "Budget",
    "Governor",
    "GuardTripped",
    "BudgetExceeded",
    "RateLimited",
    "LoopDetected",
    "AgentExpired",
    "ModelPrice",
    "cost_of",
    "price_for",
    "load_prices",
    "Store",
    "MemoryStore",
    "RedisStore",
    "AgentState",
    "Usage",
    "extract_usage",
    "__version__",
]
