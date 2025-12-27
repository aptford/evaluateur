from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol

from evaluateur.models import GeneratedQuery, GeneratedTuple


class QueryGenerator(Protocol):
    """Protocol for async query generators."""

    async def generate(
        self, tuples: AsyncIterator[GeneratedTuple], context: str
    ) -> AsyncIterator[GeneratedQuery]:
        """Generate queries asynchronously from tuples (streaming)."""


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

