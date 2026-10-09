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


class RedisStore:
    """Store backed by Redis, so spend is shared across processes/workers.

    State per agent lives under ``{prefix}:{agent}``:

    - a hash with ``spent`` and ``calls`` counters
    - a capped list ``{prefix}:{agent}:recent`` of ``"ts|usd|signature"`` rows
      (newest first), trimmed to ``maxlen``

    Pass either a ready ``redis.Redis`` client (``client=...``) or a
    ``url=`` to connect to. ``redis`` is an optional dependency.
    """

    def __init__(
        self,
        *,
        client: object | None = None,
        url: str | None = None,
        prefix: str = "budgetguard",
        maxlen: int = 512,
    ) -> None:
        if client is None:
            try:
                import redis
            except Exception as exc:  # pragma: no cover - import-env dependent
                raise ImportError(
                    "RedisStore needs redis. Install with: pip install 'budgetguard[redis]'"
                ) from exc
            client = redis.Redis.from_url(url or "redis://localhost:6379/0")
        self._r = client
        self._prefix = prefix
        self._maxlen = maxlen

    def _keys(self, agent: str) -> tuple[str, str]:
        base = f"{self._prefix}:{agent}"
        return base, f"{base}:recent"

    def record(self, agent: str, usd: float, signature: str, at: float) -> AgentState:
        hkey, lkey = self._keys(agent)
        # Signatures can contain '|', so keep it last and split with a bound.
        row = f"{at}|{usd}|{signature}"
        pipe = self._r.pipeline()
        pipe.hincrbyfloat(hkey, "spent", usd)
        pipe.hincrby(hkey, "calls", 1)
        pipe.hsetnx(hkey, "first_seen", at)
        pipe.lpush(lkey, row)
        pipe.ltrim(lkey, 0, self._maxlen - 1)
        pipe.execute()
        return self.get(agent)

    def get(self, agent: str) -> AgentState:
        hkey, lkey = self._keys(agent)
        pipe = self._r.pipeline()
        pipe.hgetall(hkey)
        pipe.lrange(lkey, 0, self._maxlen - 1)
        h, rows = pipe.execute()
        h = {_s(k): _s(v) for k, v in (h or {}).items()}
        recent: deque[tuple[float, str, float]] = deque(maxlen=self._maxlen)
        # Redis list is newest-first; reverse to oldest-first like MemoryStore.
        for raw in reversed(rows or []):
            ts, usd, sig = _s(raw).split("|", 2)
            recent.append((float(ts), sig, float(usd)))
        return AgentState(
            spent_usd=float(h.get("spent", 0.0) or 0.0),
            total_calls=int(float(h.get("calls", 0) or 0)),
            first_seen=float(h.get("first_seen", time.time())),
            recent=recent,
        )

    def reset(self, agent: str | None = None) -> None:
        if agent is None:
            keys = list(self._r.scan_iter(f"{self._prefix}:*"))
            if keys:
                self._r.delete(*keys)
        else:
            hkey, lkey = self._keys(agent)
            self._r.delete(hkey, lkey)


def _s(v: object) -> str:
    """Decode a Redis value (bytes or str) to str."""
    return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
