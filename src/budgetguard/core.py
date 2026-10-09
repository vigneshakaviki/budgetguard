"""The governor: enforce a :class:`Budget` against a :class:`Store`.

Flow for every guarded call:

1. :meth:`Governor.check` runs *before* the call and raises a
   :class:`~budgetguard.errors.GuardTripped` subclass if any cap is already
   met (spend, call count, rate window, TTL) or if the call signature is
   looping.
2. The call runs.
3. :meth:`Governor.record` runs *after*, adding the call's USD cost to the
   agent's running total.

Because cost is only known after a call returns, a single call may push spend
slightly past a USD cap; the *next* call is then refused. This "trip on the
call after the breach" behaviour is intentional and documented.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

from .budget import Budget
from .errors import AgentExpired, BudgetExceeded, LoopDetected, RateLimited
from .pricing import cost_of
from .store import MemoryStore, Store
from .usage import extract_usage

# A usage extractor maps a guarded function's return value to a cost input,
# either (model, input_tokens, output_tokens) or a direct USD float.
UsageExtractor = Callable[[Any], "tuple[str, int, int] | float | None"]

_DAY = 86_400.0
_MIN = 60.0


class Governor:
    """Binds a budget to a store and enforces it per agent."""

    def __init__(
        self,
        budget: Budget,
        *,
        store: Store | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.budget = budget
        self.store = store or MemoryStore()
        self._now = now
        # First-seen is tracked here (not in the store) so it uses this
        # governor's clock, keeping TTL correct under an injected/test clock
        # and independent of when the store lazily created the agent row.
        self._first_seen: dict[str, float] = {}

    # -- enforcement -----------------------------------------------------

    def check(self, agent: str = "default", *, signature: str | None = None) -> None:
        """Raise if the agent may not make another call right now."""
        b = self.budget
        st = self.store.get(agent)
        now = self._now()
        first_seen = self._first_seen.setdefault(agent, now)

        if b.ttl_seconds is not None and now - first_seen > b.ttl_seconds:
            raise AgentExpired(
                f"agent TTL of {b.ttl_seconds}s elapsed", agent=agent
            )

        if b.max_calls is not None and st.total_calls >= b.max_calls:
            raise BudgetExceeded(
                f"call cap reached ({st.total_calls}/{b.max_calls})",
                agent=agent,
                spent=st.total_calls,
                limit=b.max_calls,
            )

        if b.usd_total is not None and st.spent_usd >= b.usd_total:
            raise BudgetExceeded(
                f"lifetime spend cap reached (${st.spent_usd:.4f}/${b.usd_total:.2f})",
                agent=agent,
                spent=st.spent_usd,
                limit=b.usd_total,
            )

        if b.usd_per_day is not None:
            spent_today = self._windowed_spend(st, now, _DAY)
            if spent_today >= b.usd_per_day:
                raise BudgetExceeded(
                    f"daily spend cap reached (${spent_today:.4f}/${b.usd_per_day:.2f})",
                    agent=agent,
                    spent=spent_today,
                    limit=b.usd_per_day,
                )

        if b.calls_per_min is not None:
            recent = [t for t, _, _ in st.recent if now - t < _MIN]
            if len(recent) >= b.calls_per_min:
                oldest = min(recent)
                raise RateLimited(
                    f"rate cap reached ({len(recent)}/{b.calls_per_min} per min)",
                    agent=agent,
                    retry_after=max(0.0, _MIN - (now - oldest)),
                )

        if b.max_loop is not None and signature is not None:
            repeats = sum(1 for _, s, _ in st.recent if s == signature)
            if repeats >= b.max_loop:
                raise LoopDetected(
                    f"call signature repeated {repeats} times (max {b.max_loop})",
                    agent=agent,
                    signature=signature,
                    count=repeats,
                )

    def record(
        self,
        agent: str = "default",
        *,
        usd: float | None = None,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        signature: str = "",
    ) -> None:
        """Account for a completed call's cost and signature.

        Provide either ``usd`` directly, or ``model`` + token counts to have the
        cost computed from the price table.
        """
        if usd is None:
            usd = cost_of(model, input_tokens, output_tokens) if model else 0.0
        self.store.record(agent, max(0.0, usd), signature, self._now())

    # -- introspection ---------------------------------------------------

    def spent(self, agent: str = "default") -> float:
        """Lifetime USD spent by the agent."""
        return self.store.get(agent).spent_usd

    def remaining(self, agent: str = "default") -> float | None:
        """USD left under the tightest USD cap, or None if no USD cap set."""
        b = self.budget
        st = self.store.get(agent)
        caps: list[float] = []
        if b.usd_total is not None:
            caps.append(b.usd_total - st.spent_usd)
        if b.usd_per_day is not None:
            caps.append(b.usd_per_day - self._windowed_spend(st, self._now(), _DAY))
        return max(0.0, min(caps)) if caps else None

    def reset(self, agent: str | None = None) -> None:
        """Clear accounting for one agent or all agents."""
        self.store.reset(agent)
        if agent is None:
            self._first_seen.clear()
        else:
            self._first_seen.pop(agent, None)

    # -- decorator / context manager ------------------------------------

    def guard(
        self,
        agent: str = "default",
        *,
        usage: UsageExtractor | None = None,
        signature: Callable[..., str] | None = None,
    ) -> Callable:
        """Decorate a callable so each invocation is checked and recorded.

        Args:
            agent: Accounting key.
            usage: Maps the return value to cost -- either
                ``(model, input_tokens, output_tokens)`` or a USD float. If it
                returns None (or is omitted) the call records zero cost but still
                counts toward call/rate/loop caps.
            signature: Maps call args to a loop-detection signature. Defaults to
                the function name plus a repr of its arguments.
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                sig = (
                    signature(*args, **kwargs)
                    if signature
                    else _default_signature(func, args, kwargs)
                )
                self.check(agent, signature=sig)
                result = func(*args, **kwargs)
                self._record_from_usage(agent, sig, usage, result)
                return result

            return wrapper

        return decorator

    def record_result(
        self, agent: str = "default", *, result: Any = None, signature: str = ""
    ) -> None:
        """Account for a call by auto-detecting usage from its return value.

        Recognises OpenAI / Anthropic SDK responses, LangChain ``LLMResult``,
        OpenTelemetry ``gen_ai.*`` dicts, and generic token/cost dicts (see
        :func:`budgetguard.usage.extract_usage`). Unknown shapes record zero USD
        but still count toward call/rate/loop caps.
        """
        u = extract_usage(result)
        if u is None:
            self.record(agent, usd=0.0, signature=signature)
        elif u.usd is not None:
            self.record(agent, usd=u.usd, signature=signature)
        else:
            self.record(
                agent,
                model=u.model,
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                signature=signature,
            )

    def aguard(
        self,
        agent: str = "default",
        *,
        usage: UsageExtractor | None = None,
        signature: Callable[..., str] | None = None,
    ) -> Callable:
        """Async counterpart of :meth:`guard` for coroutine functions.

        Enforcement is identical: :meth:`check` runs before the awaited call,
        usage is recorded after it returns. The store calls remain synchronous
        (fast for :class:`~budgetguard.store.MemoryStore`; a brief blocking hop
        for :class:`~budgetguard.store.RedisStore`).
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                sig = (
                    signature(*args, **kwargs)
                    if signature
                    else _default_signature(func, args, kwargs)
                )
                self.check(agent, signature=sig)
                result = await func(*args, **kwargs)
                self._record_from_usage(agent, sig, usage, result)
                return result

            return wrapper

        return decorator

    def _record_from_usage(
        self, agent: str, sig: str, usage: UsageExtractor | None, result: Any
    ) -> None:
        # No explicit extractor -> auto-detect from the response shape.
        if usage is None:
            self.record_result(agent, result=result, signature=sig)
            return
        extracted = usage(result)
        if extracted is None:
            self.record(agent, usd=0.0, signature=sig)
        elif isinstance(extracted, (int, float)):
            self.record(agent, usd=float(extracted), signature=sig)
        else:
            model, inp, out = extracted
            self.record(
                agent,
                model=model,
                input_tokens=inp,
                output_tokens=out,
                signature=sig,
            )

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _windowed_spend(st, now: float, window: float) -> float:
        # Exact rolling-window spend, summed from per-call costs in the log.
        # Caveat: the log is bounded (deque maxlen), so a window holding more
        # calls than the store keeps will under-count the oldest ones. Raise
        # the store's recent-log size if you need long windows at high volume.
        return sum(usd for t, _, usd in st.recent if now - t < window)


def _default_signature(func: Callable, args: tuple, kwargs: dict) -> str:
    name = getattr(func, "__qualname__", getattr(func, "__name__", "call"))
    return f"{name}({args!r},{kwargs!r})"
