"""Demo: budgetguard catches the two classic autonomous-agent failures.

Run:  python examples/runaway_agent.py
"""

from budgetguard import Budget, BudgetExceeded, Governor, LoopDetected

# A tight budget: $0.05/day, stop any tool that repeats 3x.
gov = Governor(Budget(usd_per_day=0.05, max_loop=3))


class FakeLLMResponse:
    def __init__(self, in_tok, out_tok):
        self.in_tok, self.out_tok = in_tok, out_tok


@gov.guard(agent="worker", usage=lambda r: ("gpt-4o", r.in_tok, r.out_tok))
def ask_llm(_prompt):
    # pretend each call burns ~4k in / 1k out tokens ≈ $0.02
    return FakeLLMResponse(4000, 1000)


@gov.guard(agent="worker")
def search_web(_query):
    return "results..."


print("1) Budget cap stops a spend loop:")
try:
    for i in range(10):
        ask_llm("think harder")
        print(f"   call {i+1}: spent ${gov.spent('worker'):.4f}")
except BudgetExceeded as e:
    print(f"   STOPPED -> {e}\n")

gov.reset("worker")

print("2) Loop detector stops an identical-call loop:")
try:
    for i in range(10):
        search_web("same query over and over")
        print(f"   search {i+1}: ok")
except LoopDetected as e:
    print(f"   STOPPED -> {e}")
