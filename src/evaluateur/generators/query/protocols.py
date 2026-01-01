from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from evaluateur.models import GeneratedQuery, GeneratedTuple


class QueryGenerator(Protocol):
    """Protocol for async query generators."""

    async def generate(
        self, tuples: AsyncIterator[GeneratedTuple], context: str
    ) -> AsyncIterator[GeneratedQuery]:
        """Generate queries asynchronously from tuples (streaming)."""

