from __future__ import annotations

from evaluateur.client import LLMClient
from evaluateur.tuples.protocols import TupleGenerator
from evaluateur.tuples.strategies import (
    CrossProductTupleGenerator,
    DirectLLMTupleGenerator,
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

    if strategy == TupleStrategy.DIRECT_LLM:
        if client is None:
            raise ValueError("DIRECT_LLM tuple strategy requires an LLMClient")
        return DirectLLMTupleGenerator(client=client)

    raise ValueError(f"Unsupported tuple strategy: {strategy}")
