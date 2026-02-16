"""Configuration module for evaluateur.

This module centralizes all default configuration values, separating policy
(what defaults to use) from mechanism (how to execute operations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluateur.goals.models import GoalMode
    from evaluateur.queries.mode import QueryMode
    from evaluateur.tuples.strategies.base import TupleStrategy


@dataclass(frozen=True)
class EvaluatorConfig:
    """Configuration policy for the Evaluator.

    This dataclass holds all default values used by the Evaluator. Users can
    override any of these by passing explicit values to Evaluator methods,
    or by creating a custom config instance.

    Attributes:
        instructions: Default instructions shared across options, tuples, and
            queries.  Method-level ``instructions`` override this value.
        options_count_per_field: Default number of options to generate per field.
        tuples_count: Default number of tuples to generate.
        tuples_seed: Default random seed for tuple sampling.
        tuples_temperature: Default temperature for AI tuple generation.
        tuples_strategy: Default tuple generation strategy.
        goal_mode: Default goal guidance mode.
        query_mode: Default query generator mode.
    """

    instructions: str | None = None
    options_count_per_field: int = 5
    tuples_count: int = 20
    tuples_seed: int = 0
    tuples_temperature: float = 0.5
    tuples_strategy: str = "cross_product"
    goal_mode: str = "cycle"
    query_mode: str = "instructor"

    def get_tuple_strategy(self) -> TupleStrategy:
        """Return the TupleStrategy enum value."""
        from evaluateur.tuples.strategies.base import TupleStrategy

        return TupleStrategy(self.tuples_strategy)

    def get_goal_mode(self) -> GoalMode:
        """Return the GoalMode literal value."""
        return self.goal_mode  # type: ignore[return-value]

    def get_query_mode(self) -> QueryMode:
        """Return the QueryMode enum value."""
        from evaluateur.queries.mode import QueryMode

        return QueryMode(self.query_mode)


# The default configuration instance used when no config is provided.
DEFAULT_CONFIG = EvaluatorConfig()
