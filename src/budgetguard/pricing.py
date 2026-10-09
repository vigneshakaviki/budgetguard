"""Model price table: token counts -> USD.

Prices are USD per 1,000,000 tokens (input, output). Values are approximate and
change often; override them with :func:`load_prices` from a YAML/JSON file, or
by mutating :data:`PRICES` at runtime. A missing model costs 0 and logs nothing
special -- callers should register prices for models they care about.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """Price for one model, in USD per 1,000,000 tokens."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """USD cost for the given token counts."""
        return (
            input_tokens * self.input_per_mtok / 1_000_000
            + output_tokens * self.output_per_mtok / 1_000_000
        )


# Seed table. Not exhaustive and not authoritative -- override for anything
# you bill against. Keys are matched case-insensitively and by prefix (see
# ``price_for``), so "gpt-4o-2024-08-06" resolves to the "gpt-4o" entry.
PRICES: dict[str, ModelPrice] = {
    "gpt-4o": ModelPrice(2.50, 10.00),
    "gpt-4o-mini": ModelPrice(0.15, 0.60),
    "gpt-4.1": ModelPrice(2.00, 8.00),
    "o3": ModelPrice(2.00, 8.00),
    "claude-opus-4": ModelPrice(15.00, 75.00),
    "claude-sonnet-4": ModelPrice(3.00, 15.00),
    "claude-3-5-sonnet": ModelPrice(3.00, 15.00),
    "claude-3-5-haiku": ModelPrice(0.80, 4.00),
    "gemini-2.5-pro": ModelPrice(1.25, 10.00),
    "gemini-2.5-flash": ModelPrice(0.30, 2.50),
}


def price_for(model: str) -> ModelPrice | None:
    """Resolve a model id to a :class:`ModelPrice`.

    Matches exact id first (case-insensitive), then the longest registered key
    that is a prefix of the model id. Returns ``None`` if nothing matches.
    """
    key = model.lower()
    exact = PRICES.get(key)
    if exact is not None:
        return exact
    best: tuple[int, ModelPrice] | None = None
    for name, price in PRICES.items():
        if key.startswith(name) and (best is None or len(name) > best[0]):
            best = (len(name), price)
    return best[1] if best else None


def cost_of(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for a call, or 0.0 if the model has no registered price."""
    price = price_for(model)
    return price.cost(input_tokens, output_tokens) if price else 0.0


def load_prices(mapping: dict[str, dict], *, replace: bool = False) -> None:
    """Register prices from a plain dict (e.g. parsed YAML/JSON).

    Each value must have ``input_per_mtok`` and ``output_per_mtok``. With
    ``replace=True`` the built-in table is cleared first.
    """
    if replace:
        PRICES.clear()
    for name, spec in mapping.items():
        PRICES[name.lower()] = ModelPrice(
            float(spec["input_per_mtok"]), float(spec["output_per_mtok"])
        )
