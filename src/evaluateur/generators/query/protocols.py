from __future__ import annotations

from typing import Any, Protocol, Sequence

from evaluateur.models import GeneratedQuery, GeneratedTuple


class QueryGenerator(Protocol):
    """Protocol for async query generators."""

    async def generate(
        self, tuples: list[GeneratedTuple], context: str
    ) -> list[GeneratedQuery]:
        """Generate queries asynchronously from a list of tuples."""


class DSpyOptimizer(Protocol):
    """Minimal protocol for DSPy optimizers/teleprompters.

    Any object with a ``compile`` method matching this interface can be used.
    """

    def compile(
        self,
        student: Any,
        trainset: Sequence[Any] | None = None,
        valset: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> Any: ...

