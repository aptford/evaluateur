from __future__ import annotations

from .config import DEFAULT_CONFIG, EvaluatorConfig
from .evaluator import Evaluator
from .goals import Goal, GoalSpec
from .queries import QueryMode, build_query_generator
from .queries.merge import merge_query_metadata
from .queries.models import GeneratedQuery
from .tuples import GeneratedTuple, TupleStrategy, build_tuple_generator

__all__ = [
    # Core
    "Evaluator",
    # Configuration (policy)
    "EvaluatorConfig",
    "DEFAULT_CONFIG",
    # Query generation
    "QueryMode",
    "GeneratedQuery",
    "build_query_generator",
    "merge_query_metadata",
    # Tuple generation
    "GeneratedTuple",
    "TupleStrategy",
    "build_tuple_generator",
    # Goals
    "Goal",
    "GoalSpec",
]
