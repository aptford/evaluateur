from __future__ import annotations

from typing import Any

from evaluateur.generators.query.mode import QueryMode
from evaluateur.generators.query.protocols import DSpyOptimizer, QueryGenerator
from evaluateur.generators.query.instructor import InstructorQueryGenerator

__all__ = [
    "DSpyOptimizer",
    "DSpyQueryGenerator",
    "HybridQueryGenerator",
    "InstructorQueryGenerator",
    "QueryGenerator",
    "QueryMode",
]


def __getattr__(name: str) -> Any:
    """Lazy attribute access to keep optional backends truly optional.

    In particular: importing this module should not import DSPy.
    """

    if name == "DSpyQueryGenerator":
        from evaluateur.generators.query.dspy import DSpyQueryGenerator as _DSpyQueryGenerator

        return _DSpyQueryGenerator
    if name == "HybridQueryGenerator":
        from evaluateur.generators.query.hybrid import HybridQueryGenerator as _HybridQueryGenerator

        return _HybridQueryGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
