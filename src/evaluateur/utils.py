from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import TypeVar

T = TypeVar("T")


async def to_async_iterator(items: Iterable[T] | AsyncIterator[T]) -> AsyncIterator[T]:
    """Normalize an iterable or async iterator to an async iterator."""

    if hasattr(items, "__aiter__"):
        async for item in items:  # type: ignore[misc]
            yield item
        return

    for item in items:  # type: ignore[not-an-iterable]
        yield item
