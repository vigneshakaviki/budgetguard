# budgetguard

**Drop-in budget, rate, and loop guard for autonomous AI agents.** Framework-agnostic — works with CrewAI, LangGraph, the OpenAI / Anthropic SDKs, or any callable.

Autonomous agents that wake themselves, call each other, and loop on tools have no natural stopping point. Left unguarded they run up surprise bills (one runaway agent famously racked up a $50k cloud bill), spin in infinite tool-call loops, and exhaust token budgets with unbounded agent-to-agent history. `budgetguard` is a small library that sits in front of every LLM/tool call and **refuses the call before the spend** once a cap is hit.

```python
from budgetguard import Budget, Governor

gov = Governor(Budget(usd_per_day=5.00, max_calls=200, max_loop=10))

@gov.guard(agent="researcher", usage=lambda r: ("gpt-4o", r.usage.input, r.usage.output))
def call_llm(messages):
    return client.chat(messages)

# call_llm() now raises BudgetExceeded / RateLimited / LoopDetected
# *before* making the call once a cap is reached.
```

## Why

The "AI employee / agent crew" platforms (Clawith, Paperclip, …) and the agent frameworks all hit the same three failure modes, and each reinvents a half-baked guard internally:

- **Runaway cost** — self-waking triggers + agent-to-agent chatter with no human in the loop.
- **Infinite tool loops** — an agent calls the same tool with the same args forever.
- **No per-agent spend visibility** — you find out from the invoice.

`budgetguard` solves exactly this, once, as a dependency you can drop into any of them.

## Install

```bash
pip install budgetguard            # core, zero dependencies
pip install "budgetguard[redis]"   # shared state across workers/processes
```

## The caps

A `Budget` is plain config; any cap left `None` is not enforced.

| Cap | Meaning |
|---|---|
| `usd_per_day` | Max USD in a rolling 24h window |
| `usd_total` | Max USD over the agent's lifetime |
| `max_calls` | Max total guarded calls |
| `calls_per_min` | Max guarded calls in a rolling 60s window |
| `max_loop` | Trip if the same `(tool, args)` signature repeats this many times (default 10) |
| `ttl_seconds` | Agent may not call after this long since first seen |

## Three ways to use it

**Decorator** (shown above) — wrap the function that makes the call.

**Explicit check / record** — for when you control the call site:

```python
gov.check("researcher")                 # raises if a cap is already hit
resp = client.chat(messages)
gov.record("researcher", model="gpt-4o",
           input_tokens=resp.usage.input, output_tokens=resp.usage.output)
```

**Introspection:**

```python
gov.spent("researcher")       # 1.24  (USD lifetime)
gov.remaining("researcher")   # 3.76  (USD left under the tightest USD cap)
gov.reset("researcher")       # clear accounting (e.g. a nightly job)
```

## Pricing

Token counts become USD via a built-in table (`gpt-4o`, `claude-*`, `gemini-*`, …), matched by prefix so dated model ids resolve. Override anything:

```python
from budgetguard import load_prices
load_prices({"my-local-model": {"input_per_mtok": 0.0, "output_per_mtok": 0.0}})
```

A model with no registered price records **zero cost** but still counts toward call/rate/loop caps.

## State backends

Default is an in-process, thread-safe `MemoryStore`. For spend shared across
workers, use the Redis backend (ships in `budgetguard[redis]`). The `Store`
protocol is three methods — write your own for Postgres, etc.

## Known limits (v0.1)

- A single call may push spend slightly past a USD cap, since cost is only known *after* the call returns; the **next** call is then refused. This "trip on the call after the breach" behaviour is intentional.
- Rolling USD windows are summed from a bounded per-agent call log (deque). Very long windows at very high call volume can under-count the oldest calls — raise the store's log size if needed.
- No async wrapper yet (use `check`/`record` manually in async code). Async decorator tracked for v0.2.

## License

Apache-2.0.
