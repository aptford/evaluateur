from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Protocol, TypeAlias

from evaluateur.models import GeneratedQuery, GeneratedTuple, QueryMetadata


ContextBuilderMetadata: TypeAlias = Mapping[str, object] | QueryMetadata
ContextBuilderResult: TypeAlias = tuple[str, ContextBuilderMetadata]
ContextBuilder: TypeAlias = Callable[
    [GeneratedTuple], ContextBuilderResult | Awaitable[ContextBuilderResult]
]


class QueryGenerator(Protocol):
    """Protocol for async query generators."""

    async def generate(
        self,
        tuples: AsyncIterator[GeneratedTuple],
        context: str,
        *,
        context_builder: ContextBuilder | None = None,
    ) -> AsyncIterator[GeneratedQuery]:
        """Generate queries asynchronously from tuples (streaming)."""

