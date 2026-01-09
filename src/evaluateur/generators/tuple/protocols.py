from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, TypeVar

from pydantic import BaseModel

from evaluateur.models import GeneratedTuple

ModelT = TypeVar("ModelT", bound=BaseModel)


class TupleGenerator(Protocol[ModelT]):
    """Protocol for async tuple generators that yield tuples one at a time."""

    def generate(
        self,
        options: BaseModel,
        count: int,
        *,
        seed: int = 0,
    ) -> AsyncIterator[GeneratedTuple[ModelT]]:
        """Generate tuples asynchronously, yielding one at a time."""

