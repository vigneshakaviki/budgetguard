"""The @guard decorator wiring and the pricing table."""

from __future__ import annotations

import pytest

from budgetguard import Budget, BudgetExceeded, Governor, cost_of, load_prices, price_for


def test_decorator_records_usd_from_token_usage():
    gov = Governor(Budget(usd_total=1.00))

    class Resp:
        def __init__(self, i, o):
            self.i, self.o = i, o

    @gov.guard(agent="w", usage=lambda r: ("gpt-4o-mini", r.i, r.o))
    def call(_msg):
        return Resp(1000, 1000)  # 0.15 + 0.60 per Mtok -> 0.00075 USD

    call("hi")
    assert gov.spent("w") == pytest.approx(0.00075)


def test_decorator_blocks_before_running_when_capped():
    gov = Governor(Budget(max_calls=1))
    calls = {"n": 0}

    @gov.guard(agent="w")
    def call():
        calls["n"] += 1
        return 1

    call()
    with pytest.raises(BudgetExceeded):
        call()
    assert calls["n"] == 1  # second body never executed


def test_decorator_usd_extractor():
    gov = Governor(Budget(usd_total=5.0))

    @gov.guard(agent="w", usage=lambda r: r["cost"])
    def call():
        return {"cost": 2.5}

    call()
    assert gov.spent("w") == pytest.approx(2.5)


def test_decorator_auto_loop_signature_trips():
    gov = Governor(Budget(max_loop=2))

    @gov.guard(agent="w")
    def call(x):
        return x

    call(1)
    call(1)
    with pytest.raises(Exception) as exc:  # LoopDetected
        call(1)
    assert "repeated" in str(exc.value)
    call(2)  # different args -> different signature -> ok


def test_pricing_prefix_match_and_override():
    assert price_for("gpt-4o-2024-08-06") is price_for("gpt-4o")
    assert cost_of("unknown-model", 1000, 1000) == 0.0
    load_prices({"my-model": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}})
    assert cost_of("my-model", 1_000_000, 1_000_000) == pytest.approx(3.0)
