"""Core governor behaviour: caps trip before spend, decorator accounts cost."""

from __future__ import annotations

import pytest

from budgetguard import (
    AgentExpired,
    Budget,
    BudgetExceeded,
    Governor,
    LoopDetected,
    RateLimited,
)


class Clock:
    """Manually advanced clock injected as Governor's time source."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float) -> None:
        self.t += dt


def test_budget_validation_rejects_bad_caps():
    with pytest.raises(ValueError):
        Budget(usd_total=-1)
    with pytest.raises(ValueError):
        Budget(max_calls=0)
    with pytest.raises(ValueError):
        Budget(ttl_seconds=0)


def test_call_cap_trips_on_next_call():
    gov = Governor(Budget(max_calls=2))
    for _ in range(2):
        gov.check("a")
        gov.record("a")
    with pytest.raises(BudgetExceeded) as exc:
        gov.check("a")
    assert exc.value.agent == "a"
    assert exc.value.limit == 2


def test_lifetime_usd_cap():
    gov = Governor(Budget(usd_total=1.00))
    gov.check("a")
    gov.record("a", usd=0.60)
    gov.check("a")  # 0.60 < 1.00, still allowed
    gov.record("a", usd=0.60)  # now 1.20, over
    with pytest.raises(BudgetExceeded):
        gov.check("a")
    assert gov.spent("a") == pytest.approx(1.20)


def test_remaining_reports_tightest_cap():
    gov = Governor(Budget(usd_total=10.0, usd_per_day=3.0))
    gov.record("a", usd=1.0)
    # day window: 3 - 1 = 2 ; total: 10 - 1 = 9 ; tightest = 2
    assert gov.remaining("a") == pytest.approx(2.0)


def test_rolling_daily_window_forgets_old_spend():
    clock = Clock()
    gov = Governor(Budget(usd_per_day=1.00), now=clock)
    gov.record("a", usd=0.90)
    with pytest.raises(BudgetExceeded):
        gov.record("a", usd=0.20)  # recording is fine; check enforces
        gov.check("a")
    # advance 25h: yesterday's spend leaves the window
    clock.tick(25 * 3600)
    gov.check("a")  # no longer over the daily cap
    assert gov.remaining("a") == pytest.approx(1.00)


def test_rate_limit_per_minute():
    clock = Clock()
    gov = Governor(Budget(calls_per_min=3), now=clock)
    for _ in range(3):
        gov.check("a")
        gov.record("a")
    with pytest.raises(RateLimited) as exc:
        gov.check("a")
    assert exc.value.retry_after is not None and exc.value.retry_after > 0
    clock.tick(61)
    gov.check("a")  # window cleared


def test_loop_detection():
    gov = Governor(Budget(max_loop=3))
    for _ in range(3):
        gov.check("a", signature="same-call")
        gov.record("a", signature="same-call")
    with pytest.raises(LoopDetected) as exc:
        gov.check("a", signature="same-call")
    assert exc.value.count == 3
    # a different signature is unaffected
    gov.check("a", signature="other-call")


def test_ttl_expiry():
    clock = Clock()
    gov = Governor(Budget(ttl_seconds=100), now=clock)
    gov.check("a")
    gov.record("a")
    clock.tick(101)
    with pytest.raises(AgentExpired):
        gov.check("a")


def test_reset_clears_state():
    gov = Governor(Budget(max_calls=1))
    gov.check("a")
    gov.record("a")
    gov.reset("a")
    gov.check("a")  # fresh again
