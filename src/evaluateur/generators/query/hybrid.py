from __future__ import annotations

import logging

from evaluateur.client import LLMClient
from evaluateur.generators.query.context import compose_context
from evaluateur.generators.query.dspy_backend import build_refiner_module, configure_lm, import_dspy
from evaluateur.generators.query.instructor import InstructorQueryGenerator
from evaluateur.goals import GoalSpec
from evaluateur.models import GeneratedQuery, GeneratedTuple

log = logging.getLogger(__name__)


class HybridQueryGenerator:
    """Hybrid strategy: Instructor drafts, DSPy refines.

    First uses Instructor to generate baseline queries asynchronously, then (if
    DSPy is available) lets DSPy rewrite or enrich them. If DSPy is not
    installed this falls back to Instructor-only behaviour.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        goal_spec: GoalSpec | None = None,
        goal_prompt: str | None = None,
    ) -> None:
        self._client = client
        self._instructor_gen = InstructorQueryGenerator(client)
        self._goal_prompt = goal_prompt or (
            goal_spec.render_prompt() if goal_spec is not None else None
        )

    async def generate(self, tuples: list[GeneratedTuple], context: str) -> list[GeneratedQuery]:
        log.info("HybridQueryGenerator: generating base queries with Instructor")
        base_context = compose_context(context, self._goal_prompt)
        base_queries = await self._instructor_gen.generate(tuples, base_context)

        if import_dspy() is None:
            log.debug("HybridQueryGenerator: DSPy unavailable, returning base queries")
            return base_queries

        log.info("HybridQueryGenerator: refining %d queries with DSPy", len(base_queries))
        configure_lm(self._client)
        refiner = build_refiner_module()

        effective_context = compose_context(context, self._goal_prompt)
        refined: list[GeneratedQuery] = []
        for i, g in enumerate(base_queries):
            log.debug("HybridQueryGenerator: refining query %d/%d", i + 1, len(base_queries))
            pred = refiner(context=effective_context, original_query=g.query)
            refined.append(
                GeneratedQuery(
                    query=getattr(pred, "refined_query", g.query),
                    source_tuple=g.source_tuple,
                    metadata={"refined": True, "goal_guided": bool(self._goal_prompt)},
                )
            )

        return refined

