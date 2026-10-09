"""The :class:`Budget` value object: the caps a guard enforces.

A Budget is plain configuration -- no state. State lives in a
:class:`~budgetguard.store.Store`. Any cap left ``None`` is not enforced.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Budget:
    """Caps applied to an agent's guarded calls.

    Args:
        usd_per_day: Max USD spend in a rolling 24h window.
        usd_total: Max USD spend over the agent's lifetime.
        max_calls: Max total guarded calls over the agent's lifetime.
        calls_per_min: Max guarded calls in a rolling 60s window (rate limit).
        max_loop: Trip if the same (tool, args) signature repeats this many
            times within the recent-call window. ``None`` disables loop detection.
        ttl_seconds: Agent may not call after this many seconds since first seen.
    """

    usd_per_day: float | None = None
    usd_total: float | None = None
    max_calls: int | None = None
    calls_per_min: int | None = None
    max_loop: int | None = 10
    ttl_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("usd_per_day", "usd_total"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"{name} must be >= 0, got {v}")
        for name in ("max_calls", "calls_per_min", "max_loop"):
            v = getattr(self, name)
            if v is not None and v < 1:
                raise ValueError(f"{name} must be >= 1, got {v}")
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {self.ttl_seconds}")
