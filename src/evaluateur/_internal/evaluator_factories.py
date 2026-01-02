from __future__ import annotations

from evaluateur.client import LLMClient
from evaluateur.generators import QueryMode, TupleStrategy
from evaluateur.generators.queries import InstructorQueryGenerator, QueryGenerator
from evaluateur.generators.tuples import (
    CrossProductTupleGenerator,
    DirectLLMTupleGenerator,
    TupleGenerator,
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


def build_query_generator(*, client: LLMClient, mode: QueryMode) -> QueryGenerator:
    """Create a query generator instance for the given mode."""

    if mode == QueryMode.INSTRUCTOR:
        return InstructorQueryGenerator(client)
    raise ValueError(f"Unsupported query mode: {mode}")


