<div align="center">

# 🛡️ budgetguard

**A circuit breaker for AI agent spend.**
Stop runaway costs and infinite loops — before the money is gone.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-26%20passing-brightgreen.svg)](tests/)
[![Zero deps](https://img.shields.io/badge/dependencies-0-success.svg)](pyproject.toml)

</div>

---

Autonomous agents wake themselves, call each other, and loop on tools with **no human watching each call**. One bad prompt can run up a $50k bill overnight or spin forever on the same request. `budgetguard` sits in front of every LLM/tool call and **refuses it the moment a cap is hit** — so the breach never happens.

```python
from budgetguard import Budget, Governor

gov = Governor(Budget(usd_per_day=5.00, max_loop=10))

@gov.guard(agent="researcher")          # that's it — no config, no lambda
def ask(messages):
    return client.chat.completions.create(model="gpt-4o", messages=messages)

ask(...)   # raises BudgetExceeded / LoopDetected once a cap trips
```

That's the whole thing. Wrap the function, set the caps, done.

## Why

Every agent framework and "AI employee" platform hits the same three failures — and each reinvents a shaky guard for them:

| Problem | What happens | budgetguard |
|---|---|---|
| 💸 **Runaway cost** | self-waking agents burn tokens unattended | `usd_per_day` / `usd_total` caps |
| 🔁 **Infinite loops** | agent calls the same tool forever | `max_loop` kills repeats |
| 🙈 **No visibility** | you find out from the invoice | `gov.spent()` / `gov.remaining()` |

budgetguard solves this once, as a tiny dependency you can drop into any of them.

## Install

```bash
pip install git+https://github.com/vigneshakaviki/budgetguard
```

Zero dependencies. Python 3.10+. *(PyPI release coming.)*

## Quickstart

```python
from budgetguard import Budget, Governor, BudgetExceeded

# 1. Set the caps you care about (leave the rest None = not enforced)
gov = Governor(Budget(
    usd_per_day = 5.00,   # stop at $5/day
    max_calls   = 200,    # ...or 200 calls
    max_loop    = 10,     # ...or 10 identical calls in a row
))

# 2. Wrap the call
@gov.guard(agent="researcher")
def ask(messages):
    return openai_client.chat.completions.create(model="gpt-4o", messages=messages)

# 3. Use it normally — it blocks itself when a cap is hit
try:
    ask([{"role": "user", "content": "hello"}])
except BudgetExceeded as e:
    print("stopped before spending:", e)

print(gov.spent("researcher"))      # 0.0123  (USD used)
print(gov.remaining("researcher"))  # 4.9877  (USD left today)
```

## Works with your stack

budgetguard auto-detects token usage from the response — **OpenAI, Anthropic, LangChain, and OpenTelemetry `gen_ai.*` all work with no `usage=` lambda.** For frameworks, drop in an adapter:

<details open>
<summary><b>LiteLLM</b> — one hook covers 100+ providers (and CrewAI / DSPy / GraphRAG)</summary>

```python
import litellm
from budgetguard.adapters import LiteLLMGuard

litellm.callbacks = [LiteLLMGuard(gov)]
litellm.completion(model="gpt-4o", messages=[...])   # checked + recorded
```
On the LiteLLM **proxy**, the pre-call hook rejects with HTTP 429 *before* the request is sent.
</details>

<details>
<summary><b>LangChain</b> — a callback handler</summary>

```python
from langchain_openai import ChatOpenAI
from budgetguard.adapters import LangChainGuard

llm = ChatOpenAI(model="gpt-4o", callbacks=[LangChainGuard(gov, agent="researcher")])
llm.invoke("hi")
```
</details>

<details>
<summary><b>Any other SDK</b> — manual check / record</summary>

```python
gov.check("researcher")                      # raises if a cap is already hit
resp = any_client.generate(...)
gov.record("researcher", model="gpt-4o",
           input_tokens=resp.in_tok, output_tokens=resp.out_tok)
```
</details>

Adapters import their framework lazily — installing budgetguard never pulls in LiteLLM or LangChain.

## The caps

```python
Budget(
    usd_per_day   = 5.00,   # rolling 24h spend ceiling
    usd_total     = 50.00,  # lifetime spend ceiling
    max_calls     = 1000,   # lifetime call ceiling
    calls_per_min = 20,     # rate limit (avoids provider 429s)
    max_loop      = 10,      # trip on N identical (tool, args) calls
    ttl_seconds   = 3600,   # agent expires after 1h
)                           # any field left None = not enforced
```

Each `agent="..."` name is tracked independently — give every member of a crew its own budget against the same `Governor`.

## How it works

```
call → gov.check()  ──cap hit?──►  raise  (money never spent)
          │ ok
          ▼
       your LLM call runs
          │
          ▼
     gov.record()  ── adds cost from tokens × price table
```

Cost is known only *after* a call returns, so a single call may cross a USD cap by one call — the **next** one is refused. This "trip on the call after the breach" behaviour is intentional and documented.

## Pricing table

Token counts become USD via a built-in table (`gpt-4o`, `claude-*`, `gemini-*`, …), matched by prefix so dated model ids resolve. Override anything:

```python
from budgetguard import load_prices
load_prices({"my-model": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}})
```

A model with no registered price records **zero cost** but still counts toward call/rate/loop caps.

## State backends

Default is an in-process, thread-safe store. Need spend shared across workers? The `Store` protocol is just three methods (`record` / `get` / `reset`), so you can plug in Redis, Postgres, or anything else today. A built-in Redis store ships in v0.2.

## Limitations (v0.1)

- A call can overshoot a USD cap by one call (cost is post-call); the next is refused.
- Rolling USD windows read a bounded per-agent log — raise its size for long windows at high volume.
- No async decorator yet; use `check` / `record` in async code. *(v0.2.)*

## Run the demo

```bash
git clone https://github.com/vigneshakaviki/budgetguard && cd budgetguard
pip install -e ".[dev]"
python examples/runaway_agent.py      # watch a budget cap + loop both trip
pytest                                 # 26 tests
```

## Contributing

Issues and PRs welcome. Keep the core dependency-free; new integrations go in `adapters/` with a lazy framework import.

## License

[Apache-2.0](LICENSE).
