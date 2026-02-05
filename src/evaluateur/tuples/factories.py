from __future__ import annotations

from evaluateur.client import LLMClient
from evaluateur.tuples.protocols import TupleGenerator
from evaluateur.tuples.strategies import (
    CrossProductTupleGenerator,
    AITupleGenerator,
    TupleStrategy,
)


def build_tuple_generator(
    *,
    client: LLMClient | None,
    strategy: TupleStrategy,
) -> TupleGenerator:
    """Create a tuple generator instance for the given strategy."""

    if strategy == TupleStrategy.CROSS_PRODUCT:
        return CrossProductTupleGenerator(client=client)

    if strategy == TupleStrategy.AI:
        if client is None:
            raise ValueError("AI tuple strategy requires an LLMClient")
        return AITupleGenerator(client=client)

    raise ValueError(f"Unsupported tuple strategy: {strategy}")
