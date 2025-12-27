from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class HasQuery(Protocol):
    """Protocol for DSPy predictions that expose a `query` field."""

    query: str


class HasRefinedQuery(Protocol):
    """Protocol for DSPy predictions that expose a `refined_query` field."""

    refined_query: str


class TupleToQueryModule(Protocol):
    """Callable DSPy module used for tuple -> query generation."""

    def __call__(self, *, context: str, tuple_json: str) -> HasQuery: ...


class RefinerModule(Protocol):
    """Callable DSPy module used for query refinement."""

    def __call__(self, *, context: str, original_query: str) -> HasRefinedQuery: ...


class DSpyOptimizer(Protocol):
    """Minimal protocol for DSPy optimizers/teleprompters.

    Any object with a ``compile`` method matching this interface can be used.
    """

    def compile(
        self,
        student: object,
        trainset: Sequence[object] | None = None,
        valset: Sequence[object] | None = None,
        **kwargs: Any,
    ) -> object: ...


