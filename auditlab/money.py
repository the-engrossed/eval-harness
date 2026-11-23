"""Money. Amounts are integer cents; model cost is integer USD micros. There is no float dollar.

Rates are never defined here. They come from `config/models.toml`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from pydantic import Field

Cents = Annotated[int, Field(strict=True)]
Micros = Annotated[int, Field(strict=True)]


def cost_micros(
    tokens_in: int,
    tokens_out: int,
    *,
    input_micros_per_million: int,
    output_micros_per_million: int,
) -> int:
    if min(tokens_in, tokens_out, input_micros_per_million, output_micros_per_million) < 0:
        raise ValueError("token counts and rates must be non-negative")
    return (
        tokens_in * input_micros_per_million + tokens_out * output_micros_per_million
    ) // 1_000_000


def format_usd(micros: int) -> str:
    sign = "-" if micros < 0 else ""
    dollars, rem = divmod(abs(micros), 1_000_000)
    return f"{sign}${dollars}.{rem:06d}"


def format_cents(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    dollars, rem = divmod(abs(cents), 100)
    return f"{sign}${dollars:,}.{rem:02d}"


def mean_micros(values: Sequence[int]) -> int:
    if not values:
        raise RuntimeError("cannot average an empty cost series")
    return sum(values) // len(values)
