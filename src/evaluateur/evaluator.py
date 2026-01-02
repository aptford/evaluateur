from __future__ import annotations

import logging
import random
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Type, TypeVar

from pydantic import BaseModel

from evaluateur.client import LLMClient
from evaluateur.generators import OptionsGenerator, QueryMode, TupleStrategy
from evaluateur.generators.query.context import compose_query_context
from evaluateur.goals import GoalFocusArea, GoalMode, GoalSpec
from evaluateur.generators.queries import (
    InstructorQueryGenerator,
    QueryGenerator,
)
from evaluateur.generators.tuples import (
    CrossProductTupleGenerator,
    DirectLLMTupleGenerator,
    TupleGenerator,
)
from evaluateur.models import GeneratedQuery, GeneratedTuple, QueryMetadata

log = logging.getLogger(__name__)

QueryModelT = TypeVar("QueryModelT", bound=BaseModel)


@dataclass(frozen=True)
class TupleConfig:
    """Configuration for tuple generation."""

    strategy: TupleStrategy = TupleStrategy.CROSS_PRODUCT
    count: int = 20
    seed: int = 0


@dataclass(frozen=True)
class QueryConfig:
    """Configuration for query generation."""

    mode: QueryMode = QueryMode.INSTRUCTOR
    goal_mode: GoalMode = "sample"
    goal_seed: int = 0


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

    async def options(
        self,
        instructions: str | None = None,
        n: int = 5,
    ) -> BaseModel:
        """Generate an options ``BaseModel`` from the configured query model.

        Every simple field on the input model is turned into a sequence of
        options. Iterator fields (lists, tuples, etc.) are preserved.
        """
        log.info(
            "Generating options for %s (n=%d)",
            self.model.__name__,
            n,
        )
        result = await self._options_generator.generate_options(
            self.model,
            instructions=instructions,
            count_per_field=n,
        )
        log.debug("Generated options: %s", result)
        return result

    def _build_tuple_generator(self, strategy: TupleStrategy) -> TupleGenerator:
        """Build a tuple generator based on the specified strategy."""
        if strategy == TupleStrategy.CROSS_PRODUCT:
            return CrossProductTupleGenerator(client=self.client)
        if strategy == TupleStrategy.DIRECT_LLM:
            return DirectLLMTupleGenerator(client=self.client)
        raise ValueError(f"Unsupported tuple strategy: {strategy}")

    def _build_query_generator(
        self,
        mode: QueryMode,
    ) -> QueryGenerator:
        """Build a query generator based on the specified mode."""
        if mode == QueryMode.INSTRUCTOR:
            return InstructorQueryGenerator(self.client)
        raise ValueError(f"Unsupported query mode: {mode}")

    async def _ensure_options(
        self, maybe_options: BaseModel | None, *, instructions: str | None = None
    ) -> BaseModel:
        """Ensure options are available, generating them if not provided."""
        if maybe_options is not None:
            return maybe_options
        return await self.options(instructions=instructions)

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

        options_instance = await self._ensure_options(options)
        log.debug("Using options: %s", options_instance)

        tuple_gen = self._build_tuple_generator(config.strategy)
        generated_count = 0

        async for t in tuple_gen.generate(
            options_instance, config.count, seed=config.seed
        ):
            generated_count += 1
            yield t

        log.info("Generated %d tuples", generated_count)

    async def _normalize_goals(
        self, goals: GoalSpec | str | None
    ) -> GoalSpec | None:
        if goals is None:
            return None
        if isinstance(goals, str):
            spec = await GoalSpec.from_text(self.client, goals)
        elif isinstance(goals, GoalSpec):
            spec = goals
        else:  # pragma: no cover - defensive
            raise TypeError("goals must be a GoalSpec, a string, or None")

        if spec.is_empty():
            return None
        return spec

    async def _aiter_tuples(
        self, tuples: Sequence[GeneratedTuple] | AsyncIterator[GeneratedTuple]
    ) -> AsyncIterator[GeneratedTuple]:
        if hasattr(tuples, "__aiter__"):
            async for t in tuples:  # type: ignore[misc]
                yield t
            return
        for t in tuples:
            yield t

    async def queries(
        self,
        *,
        tuples: Sequence[GeneratedTuple] | AsyncIterator[GeneratedTuple],
        config: QueryConfig = QueryConfig(),
        goals: GoalSpec | str | None = None,
        instructions: str | None = None,
    ) -> AsyncIterator[GeneratedQuery]:
        log.info(
            "Generating queries: mode=%s, tuple_count=%s",
            config.mode.value,
            "streaming" if hasattr(tuples, "__aiter__") else len(tuples),  # type: ignore[arg-type]
        )

        query_gen = self._build_query_generator(config.mode)

        goal_spec = await self._normalize_goals(goals)
        focus_areas: list[GoalFocusArea] = (
            goal_spec.available_focus_areas() if goal_spec is not None else []
        )
        goal_guided = bool(focus_areas)

        goal_prompt: str | None = None
        if config.goal_mode == "full" and goal_spec is not None:
            # Full mode conditions the entire run on all goals at once.
            goal_prompt = goal_spec.render_prompt()

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

        q: GeneratedQuery
        if config.goal_mode == "sample" and goal_spec is not None and focus_areas:
            rng = random.Random(config.goal_seed)

            def _context_builder(
                t: GeneratedTuple,
            ) -> tuple[str, dict[str, object]]:
                focus = rng.choice(focus_areas)
                focus_prompt = goal_spec.render_focus_prompt(focus_area=focus)
                ctx = compose_query_context(
                    self.context,
                    instructions=instructions,
                    goal_prompt=focus_prompt,
                )
                return ctx, {"goal_focus_area": focus}

            gen_with_builder = getattr(query_gen, "generate_with_context_builder", None)
            if gen_with_builder is not None:
                async for q in gen_with_builder(  # type: ignore[no-untyped-call]
                    self._aiter_tuples(tuples),
                    _context_builder,
                ):
                    merged = {
                        **run_metadata.model_dump(exclude_none=True),
                        **q.metadata.model_dump(exclude_none=True, exclude_unset=True),
                    }
                    yield GeneratedQuery(
                        query=q.query,
                        source_tuple=q.source_tuple,
                        metadata=QueryMetadata.model_validate(merged),
                    )
                return

            # Compatibility fallback: if the generator cannot vary context per
            # tuple, sample a single focus area once (seeded) and apply it to
            # the whole run.
            run_focus = rng.choice(focus_areas)
            run_metadata.goal_focus_area = run_focus
            goal_prompt = goal_spec.render_focus_prompt(focus_area=run_focus)

        effective_context = compose_query_context(
            self.context,
            instructions=instructions,
            goal_prompt=goal_prompt,
        )

        async for q in query_gen.generate(self._aiter_tuples(tuples), effective_context):
            merged = {
                # Run metadata provides defaults; per-query metadata should win
                # (e.g. generator may attach tracing keys).
                **run_metadata.model_dump(exclude_none=True),
                **q.metadata.model_dump(exclude_none=True, exclude_unset=True),
            }
            yield GeneratedQuery(
                query=q.query,
                source_tuple=q.source_tuple,
                metadata=QueryMetadata.model_validate(merged),
            )

    async def run(
        self,
        *,
        options: BaseModel | None = None,
        tuple_config: TupleConfig = TupleConfig(),
        query_config: QueryConfig = QueryConfig(),
        goals: GoalSpec | str | None = None,
        instructions: str | None = None,
    ) -> AsyncIterator[GeneratedQuery]:
        """Convenience wrapper: options → tuples → queries (streaming)."""

        options_instance = await self._ensure_options(options, instructions=instructions)
        tuple_iter = self.tuples(options_instance, config=tuple_config)
        async for q in self.queries(
            tuples=tuple_iter,
            config=query_config,
            goals=goals,
            instructions=instructions,
        ):
            yield q
