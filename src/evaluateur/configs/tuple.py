from __future__ import annotations

from dataclasses import dataclass

from evaluateur.generators import TupleStrategy


@dataclass(frozen=True)
class TupleConfig:
    """Configuration for tuple generation."""

    strategy: TupleStrategy = TupleStrategy.CROSS_PRODUCT
    count: int = 20
    seed: int = 0
    #: Optional instructions that guide *option generation* (dimension values).
    #:
    #: Used only when `Evaluator` needs to generate options because none were
    #: provided (e.g. "Focus on common US payers.").
    instructions: str | None = None
    #: Target number of options to generate per dimension when options are not provided.
    options_per_field: int = 5


