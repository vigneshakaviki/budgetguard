"""Pluggable state backend for accounting.

A store holds, per ``(agent)`` key, the accumulated spend and a timestamped log
of recent calls (for rate windows and loop detection). The default
:class:`MemoryStore` is process-local; swap in :class:`RedisStore` (extra) to
share state across workers.

Stores deal only in primitives -- the policy lives in :mod:`budgetguard.core`.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class AgentState:
    """Mutable accounting for one agent."""

    spent_usd: float = 0.0
    total_calls: int = 0
    first_seen: float = field(default_factory=time.time)
    # (timestamp, signature, usd) for recent calls; bounded by the store. The
    # per-call cost lets rolling USD windows (e.g. usd_per_day) be exact.
    recent: deque[tuple[float, str, float]] = field(
        default_factory=lambda: deque(maxlen=512)
    )


class Store(Protocol):
    """Backend interface. Implementations must be safe for concurrent use."""

    def record(self, agent: str, usd: float, signature: str, at: float) -> AgentState:
        """Add a call's cost and signature, returning the updated state."""
        ...

    def get(self, agent: str) -> AgentState:
        """Return current state (a copy or snapshot) without mutating it."""
        ...

    def reset(self, agent: str | None = None) -> None:
        """Clear one agent, or all agents when ``agent`` is None."""
        ...


class MemoryStore:
    """Thread-safe, process-local store. The zero-config default."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._agents: dict[str, AgentState] = defaultdict(AgentState)

    def record(self, agent: str, usd: float, signature: str, at: float) -> AgentState:
        with self._lock:
            st = self._agents[agent]
            st.spent_usd += usd
            st.total_calls += 1
            st.recent.append((at, signature, usd))
            return self._snapshot(st)

    def get(self, agent: str) -> AgentState:
        with self._lock:
            return self._snapshot(self._agents[agent])

    def reset(self, agent: str | None = None) -> None:
        with self._lock:
            if agent is None:
                self._agents.clear()
            else:
                self._agents.pop(agent, None)

    @staticmethod
    def _snapshot(st: AgentState) -> AgentState:
        # Return a shallow copy so callers can read without holding the lock.
        return AgentState(
            spent_usd=st.spent_usd,
            total_calls=st.total_calls,
            first_seen=st.first_seen,
            recent=deque(st.recent, maxlen=st.recent.maxlen),
        )
