from __future__ import annotations

from dataclasses import dataclass

from evaluateur.generators import TupleStrategy


@dataclass(frozen=True)
class TupleConfig:
    """Configuration for tuple generation."""

    strategy: TupleStrategy = TupleStrategy.CROSS_PRODUCT
    count: int = 20
    seed: int = 0
