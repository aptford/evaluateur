from __future__ import annotations

"""DSPy integration (optional dependency).

This package isolates all DSPy imports and DSPy-specific logic behind a stable,
typed interface. Importing Evaluateur should not import DSPy; DSPy is only
required when you use DSPy-backed query generation or optimization.
"""

from .backend import build_refiner_module, build_tuple_to_query_module, configure_lm
from .imports import import_dspy, require_dspy
from .query_generator import DSpyQueryGenerator
from .types import DSpyOptimizer

__all__ = [
    "DSpyOptimizer",
    "DSpyQueryGenerator",
    "build_refiner_module",
    "build_tuple_to_query_module",
    "configure_lm",
    "import_dspy",
    "require_dspy",
]


