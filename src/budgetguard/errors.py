"""Exceptions raised by budgetguard when a guard trips.

All inherit :class:`GuardTripped` so callers can catch the whole family with
one ``except``. Each carries the ``agent`` it fired for and a human ``reason``.
"""

from __future__ import annotations


class GuardTripped(Exception):
    """Base for every guard violation. Raised *before* the guarded call runs."""

    def __init__(self, reason: str, *, agent: str | None = None) -> None:
        self.reason = reason
        self.agent = agent
        prefix = f"[{agent}] " if agent else ""
        super().__init__(f"{prefix}{reason}")


class BudgetExceeded(GuardTripped):
    """A spend cap (USD) or call-count cap was reached."""

    def __init__(
        self,
        reason: str,
        *,
        agent: str | None = None,
        spent: float | None = None,
        limit: float | None = None,
    ) -> None:
        self.spent = spent
        self.limit = limit
        super().__init__(reason, agent=agent)


class RateLimited(GuardTripped):
    """A per-window call-rate cap was reached (retryable after ``retry_after``)."""

    def __init__(
        self,
        reason: str,
        *,
        agent: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(reason, agent=agent)


class LoopDetected(GuardTripped):
    """The same (tool, args) call repeated past ``max_loop`` -- a runaway loop."""

    def __init__(
        self,
        reason: str,
        *,
        agent: str | None = None,
        signature: str | None = None,
        count: int | None = None,
    ) -> None:
        self.signature = signature
        self.count = count
        super().__init__(reason, agent=agent)


class AgentExpired(GuardTripped):
    """The agent's TTL elapsed; it may no longer make calls."""
