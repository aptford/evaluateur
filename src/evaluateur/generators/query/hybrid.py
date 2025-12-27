from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from evaluateur.client import LLMClient
from evaluateur.integrations.dspy import build_refiner_module, configure_lm, import_dspy
from evaluateur.generators.query.instructor import InstructorQueryGenerator
from evaluateur.models import GeneratedQuery, GeneratedTuple

log = logging.getLogger(__name__)


class HybridQueryGenerator:
    """Hybrid strategy: Instructor drafts, DSPy refines.

    First uses Instructor to generate baseline queries asynchronously, then (if
    DSPy is available) lets DSPy rewrite or enrich them. If DSPy is not
    installed this falls back to Instructor-only behaviour.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self._instructor_gen = InstructorQueryGenerator(client)

    async def generate(
        self, tuples: AsyncIterator[GeneratedTuple], context: str
    ) -> AsyncIterator[GeneratedQuery]:
        log.info("HybridQueryGenerator: generating base queries with Instructor (streaming)")

        dspy_available = import_dspy() is not None
        if dspy_available:
            configure_lm(self._client)
            refiner = build_refiner_module()

        async for t in tuples:
            async def _one() -> AsyncIterator[GeneratedTuple]:
                yield t

            base_query: GeneratedQuery | None = None
            async for q in self._instructor_gen.generate(_one(), context):
                base_query = q

            if base_query is None:  # pragma: no cover - defensive
                continue

            if not dspy_available:
                yield base_query
                continue

            pred = refiner(context=context, original_query=base_query.query)  # type: ignore[name-defined]
            yield GeneratedQuery(
                query=getattr(pred, "refined_query", base_query.query),
                source_tuple=base_query.source_tuple,
                metadata={"refined": True},
            )

