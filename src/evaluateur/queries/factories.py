from __future__ import annotations

from evaluateur.client import LLMClient
from evaluateur.queries.instructor import InstructorQueryGenerator
from evaluateur.queries.mode import QueryMode
from evaluateur.queries.protocols import QueryGenerator


def build_query_generator(
    *,
    client: LLMClient,
    mode: QueryMode = QueryMode.INSTRUCTOR,
) -> QueryGenerator:
    """Create a query generator instance for the given mode."""

    if mode == QueryMode.INSTRUCTOR:
        return InstructorQueryGenerator(client)
    raise ValueError(f"Unsupported query mode: {mode}")
