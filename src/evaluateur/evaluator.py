from __future__ import annotations

import logging
import random
from collections.abc import AsyncIterator, Sequence
from typing import Type, TypeVar

from pydantic import BaseModel

from evaluateur.client import LLMClient
from evaluateur.generators import OptionsGenerator, QueryMode, TupleStrategy
from evaluateur.generators.query.context import compose_query_context
from evaluateur.goals import GoalFocusArea, GoalSpec
from evaluateur.goals import GoalMode
from evaluateur.models import GeneratedQuery, GeneratedTuple, QueryMetadata
from evaluateur._internal.async_iter import to_async_iterator
from evaluateur._internal.evaluator_factories import (
    build_query_generator,
    build_tuple_generator,
)
from evaluateur._internal.evaluator_goal_guidance import (
    GoalSamplingContextBuilder,
    normalize_goal_spec,
)

log = logging.getLogger(__name__)

QueryModelT = TypeVar("QueryModelT", bound=BaseModel)


class Evaluator:
    """Async synthetic evaluation helper following the dimensions → tuples → queries flow.

    The evaluator is parameterized by a Pydantic model that describes the
    dimensions of a query (e.g. payer, age, complexity, geography).
    """

    def __init__(
        self,
        model: Type[QueryModelT],
        *,
        client: LLMClient | None = None,
    ) -> None:
        self.model = model
        self.client = client or LLMClient.from_env()
        log.debug(
            "Evaluator initialized: model=%s, provider=%s, model_name=%s",
            model.__name__,
            self.client.provider,
            self.client.model_name,
        )

    async def options(
        self, *, instructions: str | None = None, count_per_field: int = 5
    ) -> BaseModel:
        """Generate an options ``BaseModel`` from the configured query model.

        Every simple field on the input model is turned into a sequence of
        options. Iterator fields (lists, tuples, etc.) are preserved.
        """
        log.info(
            "Generating options for %s (n=%d)",
            self.model.__name__,
            count_per_field,
        )
        options_generator = OptionsGenerator(self.client)
        result = await options_generator.generate_options(
            self.model,
            instructions=instructions,
            count_per_field=count_per_field,
        )
        log.debug("Generated options: %s", result)
        return result

    async def _ensure_options(
        self,
        maybe_options: BaseModel | None,
        *,
        instructions: str | None,
        count_per_field: int,
    ) -> BaseModel:
        """Ensure options are available, generating them if not provided."""
        if maybe_options is not None:
            return maybe_options
        return await self.options(
            instructions=instructions, count_per_field=count_per_field
        )

    async def tuples(
        self,
        options: BaseModel,
        *,
        strategy: TupleStrategy = TupleStrategy.CROSS_PRODUCT,
        count: int = 20,
        seed: int = 0,
        instructions: str | None = None,
    ) -> AsyncIterator[GeneratedTuple]:
        """Generate tuples as an async iterator, yielding one at a time.

        Instructions are forwarded to tuple generators that support them.
        """
        log.info(
            "Generating tuples: strategy=%s, count=%d",
            strategy.value,
            count,
        )

        log.debug("Using options: %s", options)

        tuple_gen = build_tuple_generator(client=self.client, strategy=strategy)
        generated_count = 0

        async for t in tuple_gen.generate(
            options, count, seed=seed, instructions=instructions
        ):
            generated_count += 1
            yield t

        log.info("Generated %d tuples", generated_count)

    async def queries(
        self,
        *,
        tuples: Sequence[GeneratedTuple] | AsyncIterator[GeneratedTuple],
        mode: QueryMode = QueryMode.INSTRUCTOR,
        instructions: str | None = None,
        goal_mode: GoalMode = "sample",
        seed: int = 0,
        goals: GoalSpec | str | None = None,
    ) -> AsyncIterator[GeneratedQuery]:
        log.info(
            "Generating queries: mode=%s, tuple_count=%s",
            mode.value,
            "streaming" if hasattr(tuples, "__aiter__") else len(tuples),  # type: ignore[arg-type]
        )

        query_gen = build_query_generator(client=self.client, mode=mode)

        goal_spec = await normalize_goal_spec(self.client, goals)
        focus_areas: list[GoalFocusArea] = (
            goal_spec.available_focus_areas() if goal_spec is not None else []
        )
        goal_guided = bool(focus_areas)

        run_metadata = QueryMetadata(
            mode=mode.value,
            goal_guided=goal_guided,
            goal_mode=goal_mode,
            query_goals=(
                goal_spec
                if goal_spec is not None and not goal_spec.is_empty()
                else None
            ),
        )

        q: GeneratedQuery
        if goal_mode == "sample" and goal_spec is not None and focus_areas:
            rng = random.Random(seed)
            context_builder = GoalSamplingContextBuilder(
                base_context=instructions,
                goal_spec=goal_spec,
                focus_areas=focus_areas,
                rng=rng,
            )

            async for q in query_gen.generate(
                to_async_iterator(tuples),
                instructions,
                context_builder=context_builder,
            ):
                yield GeneratedQuery(
                    query=q.query,
                    source_tuple=q.source_tuple,
                    metadata=QueryMetadata.merge(
                        run_metadata=run_metadata,
                        per_query_metadata=q.metadata,
                    ),
                )
            return

        goal_prompt: str | None = None
        if goal_mode == "full" and goal_spec is not None:
            goal_prompt = goal_spec.render_prompt()

        effective_context = compose_query_context(instructions, goal_prompt=goal_prompt)

        async for q in query_gen.generate(to_async_iterator(tuples), effective_context):
            yield GeneratedQuery(
                query=q.query,
                source_tuple=q.source_tuple,
                metadata=QueryMetadata.merge(
                    run_metadata=run_metadata,
                    per_query_metadata=q.metadata,
                ),
            )

    async def run(
        self,
        *,
        options: BaseModel | None = None,
        instructions: str | None = None,
        count_per_field: int = 5,
        tuple_strategy: TupleStrategy = TupleStrategy.CROSS_PRODUCT,
        tuple_count: int = 20,
        seed: int = 0,
        query_mode: QueryMode = QueryMode.INSTRUCTOR,
        goal_mode: GoalMode = "sample",
        goals: GoalSpec | str | None = None,
    ) -> AsyncIterator[GeneratedQuery]:
        """Convenience wrapper: options → tuples → queries (streaming).

        Parameters
        ----------
        options
            Pre-generated options model instance. If not provided, options
            will be generated using ``instructions`` and ``count_per_field``.
        instructions
            Optional instructions shared across options, tuples, and queries.
        count_per_field
            Number of options to generate per field.
        tuple_strategy
            Tuple sampling strategy.
        tuple_count
            Number of tuples to generate.
        seed
            Random seed for tuple sampling and goal sampling.
        query_mode
            Query generation mode.
        goal_mode
            Goal guidance mode ("sample" or "full").
        goals
            Goal specification for guided query generation.
        """

        options_instance = await self._ensure_options(
            options,
            instructions=instructions,
            count_per_field=count_per_field,
        )
        tuple_iter = self.tuples(
            options_instance,
            strategy=tuple_strategy,
            count=tuple_count,
            seed=seed,
            instructions=instructions,
        )
        async for q in self.queries(
            tuples=tuple_iter,
            mode=query_mode,
            instructions=instructions,
            goal_mode=goal_mode,
            seed=seed,
            goals=goals,
        ):
            yield q
