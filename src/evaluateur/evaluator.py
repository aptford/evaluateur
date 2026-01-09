from __future__ import annotations

import logging
import random
from collections.abc import AsyncIterator, Sequence
from typing import Type, TypeVar

from pydantic import BaseModel

from evaluateur.client import LLMClient
from evaluateur.configs import QueryConfig, TupleConfig
from evaluateur.generators import OptionsGenerator
from evaluateur.generators.query.context import compose_query_context
from evaluateur.goals import GoalFocusArea, GoalSpec
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
        context: str = "",
    ) -> None:
        self.model = model
        self.client = client or LLMClient.from_env()
        self.context = context
        self._options_generator = OptionsGenerator(self.client)
        log.debug(
            "Evaluator initialized: model=%s, provider=%s, model_name=%s",
            model.__name__,
            self.client.provider,
            self.client.model_name,
        )

    async def options(self, *, config: TupleConfig = TupleConfig()) -> BaseModel:
        """Generate an options ``BaseModel`` from the configured query model.

        Every simple field on the input model is turned into a sequence of
        options. Iterator fields (lists, tuples, etc.) are preserved.
        """
        log.info(
            "Generating options for %s (n=%d)",
            self.model.__name__,
            config.options_per_field,
        )
        result = await self._options_generator.generate_options(
            self.model,
            instructions=config.instructions,
            count_per_field=config.options_per_field,
        )
        log.debug("Generated options: %s", result)
        return result

    async def _ensure_options(
        self,
        maybe_options: BaseModel | None,
        *,
        config: TupleConfig,
    ) -> BaseModel:
        """Ensure options are available, generating them if not provided."""
        if maybe_options is not None:
            return maybe_options
        return await self.options(config=config)

    async def tuples(
        self,
        options: BaseModel | None = None,
        *,
        config: TupleConfig = TupleConfig(),
    ) -> AsyncIterator[GeneratedTuple]:
        """Generate tuples as an async iterator, yielding one at a time.

        If `options` is omitted, the evaluator will first generate options.
        """
        log.info(
            "Generating tuples: strategy=%s, count=%d",
            config.strategy.value,
            config.count,
        )

        options_instance = await self._ensure_options(options, config=config)
        log.debug("Using options: %s", options_instance)

        tuple_gen = build_tuple_generator(client=self.client, strategy=config.strategy)
        generated_count = 0

        async for t in tuple_gen.generate(
            options_instance, config.count, seed=config.seed
        ):
            generated_count += 1
            yield t

        log.info("Generated %d tuples", generated_count)

    async def queries(
        self,
        *,
        tuples: Sequence[GeneratedTuple] | AsyncIterator[GeneratedTuple],
        config: QueryConfig = QueryConfig(),
        goals: GoalSpec | str | None = None,
    ) -> AsyncIterator[GeneratedQuery]:
        log.info(
            "Generating queries: mode=%s, tuple_count=%s",
            config.mode.value,
            "streaming" if hasattr(tuples, "__aiter__") else len(tuples),  # type: ignore[arg-type]
        )

        query_gen = build_query_generator(client=self.client, mode=config.mode)

        goal_spec = await normalize_goal_spec(self.client, goals)
        focus_areas: list[GoalFocusArea] = (
            goal_spec.available_focus_areas() if goal_spec is not None else []
        )
        goal_guided = bool(focus_areas)

        run_metadata = QueryMetadata(
            mode=config.mode.value,
            goal_guided=goal_guided,
            goal_mode=config.goal_mode,
            query_goals=(
                goal_spec
                if goal_spec is not None and not goal_spec.is_empty()
                else None
            ),
        )

        # Base context is shared across the run; goal sampling can append a
        # per-tuple focused goal prompt via the context_builder.
        base_context = compose_query_context(
            self.context,
            instructions=config.instructions,
        )

        q: GeneratedQuery
        if config.goal_mode == "sample" and goal_spec is not None and focus_areas:
            rng = random.Random(config.goal_seed)
            context_builder = GoalSamplingContextBuilder(
                base_context=base_context,
                goal_spec=goal_spec,
                focus_areas=focus_areas,
                rng=rng,
                max_chars_for_goals=config.max_chars_for_goals,
            )

            async for q in query_gen.generate(
                to_async_iterator(tuples),
                base_context,
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
        if config.goal_mode == "full" and goal_spec is not None:
            # Full mode conditions the entire run on all goals at once.
            goal_prompt = goal_spec.render_prompt(max_chars=config.max_chars_for_goals)

        effective_context = compose_query_context(base_context, goal_prompt=goal_prompt)

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
        tuple_config: TupleConfig = TupleConfig(),
        query_config: QueryConfig = QueryConfig(),
        goals: GoalSpec | str | None = None,
    ) -> AsyncIterator[GeneratedQuery]:
        """Convenience wrapper: options → tuples → queries (streaming).

        Notes
        -----
        Instructions are configured via the config objects:
        - Use ``QueryConfig.instructions`` to guide query writing.
        - Use ``TupleConfig.options_instructions`` to guide option generation.
        """

        tuple_iter = self.tuples(
            options,
            config=tuple_config,
        )
        async for q in self.queries(
            tuples=tuple_iter,
            config=query_config,
            goals=goals,
        ):
            yield q
