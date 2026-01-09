from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TypeVar

T = TypeVar("T")


async def to_async_iterator(values: Sequence[T] | AsyncIterator[T]) -> AsyncIterator[T]:
    """Normalize a sequence or async iterator into an async iterator.

    This keeps `Evaluator` APIs flexible (accepting both eager and streaming
    sources) while centralizing the conversion logic.
    """

    if hasattr(values, "__aiter__"):
        async for v in values:  # type: ignore[misc]
            yield v
        return

    for v in values:
        yield v


