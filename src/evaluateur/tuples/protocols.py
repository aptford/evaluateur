from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from pydantic import BaseModel

from evaluateur.queries.models import GeneratedTuple


class TupleGenerator(Protocol):
    """Protocol for async tuple generators that yield tuples one at a time."""

    def generate(
        self,
        options: BaseModel,
        count: int,
        *,
        seed: int = 0,
        instructions: str | None = None,
    ) -> AsyncIterator[GeneratedTuple]:
        """Generate tuples asynchronously, yielding one at a time."""
