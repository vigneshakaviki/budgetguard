"""RedisStore (via fakeredis) and the async @aguard decorator."""

from __future__ import annotations

import pytest

from budgetguard import Budget, BudgetExceeded, Governor, LoopDetected, RedisStore


# -- RedisStore -----------------------------------------------------------


@pytest.fixture
def redis_gov():
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeStrictRedis()
    store = RedisStore(client=client, maxlen=16)
    return Governor(Budget(usd_total=1.00, max_calls=3, max_loop=2), store=store)


def test_redis_tracks_spend_and_calls(redis_gov):
    redis_gov.check("a")
    redis_gov.record("a", usd=0.40, signature="s1")
    redis_gov.check("a")
    redis_gov.record("a", usd=0.40, signature="s2")
    assert redis_gov.spent("a") == pytest.approx(0.80)
    redis_gov.record("a", usd=0.40, signature="s3")  # 1.20, over
    with pytest.raises(BudgetExceeded):
        redis_gov.check("a")


def test_redis_call_cap(redis_gov):
    for _ in range(3):
        redis_gov.check("b")
        redis_gov.record("b", signature="x")
    with pytest.raises(BudgetExceeded):
        redis_gov.check("b")


def test_redis_loop_detection_and_signature_roundtrip(redis_gov):
    # signature containing '|' must survive the encode/decode round-trip
    sig = "model|with|pipes"
    for _ in range(2):
        redis_gov.check("c", signature=sig)
        redis_gov.record("c", signature=sig)
    with pytest.raises(LoopDetected):
        redis_gov.check("c", signature=sig)


def test_redis_reset(redis_gov):
    redis_gov.record("d", usd=0.5, signature="s")
    redis_gov.reset("d")
    assert redis_gov.spent("d") == 0.0


# -- async @aguard --------------------------------------------------------


async def test_aguard_records_and_blocks():
    gov = Governor(Budget(max_calls=2))
    calls = {"n": 0}

    @gov.aguard(agent="w")
    async def call():
        calls["n"] += 1
        return {"cost": 0.1}

    await call()
    await call()
    with pytest.raises(BudgetExceeded):
        await call()
    assert calls["n"] == 2  # third body never ran


async def test_aguard_autodetects_usage():
    gov = Governor(Budget(usd_total=5.0))

    @gov.aguard(agent="w")
    async def call():
        return {"cost": 2.5}

    await call()
    assert gov.spent("w") == pytest.approx(2.5)
