from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptionsConfig:
    """Configuration for options generation.

    Options are the discrete values generated for each dimension of the query
    model. They are used to create tuples via cartesian product or sampling.
    """

    #: Optional instructions that guide option generation (dimension values).
    #:
    #: Used when the evaluator generates options because none were provided
    #: (e.g. "Focus on common US payers.").
    instructions: str | None = None
    #: Target number of options to generate per dimension.
    count_per_field: int = 5
