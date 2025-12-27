from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Sequence, Type, TypeVar

from pydantic import BaseModel

from evaluateur.client import LLMClient
from evaluateur.generators import OptionsGenerator, QueryMode, TupleStrategy
from evaluateur.generators.query.context import compose_context
from evaluateur.goals import GoalSpec
from evaluateur.generators.queries import (
    DSpyOptimizer,
    InstructorQueryGenerator,
    QueryGenerator,
)
from evaluateur.generators.tuples import (
    CrossProductTupleGenerator,
    DirectLLMTupleGenerator,
    TupleGenerator,
)
from evaluateur.models import GeneratedQuery, GeneratedTuple, QueryMetadata
from evaluateur.optimizers import GoalGuidedQueryOptimizer, JudgeBackend

log = logging.getLogger(__name__)

QueryModelT = TypeVar("QueryModelT", bound=BaseModel)


@dataclass(frozen=True)
class TupleConfig:
    """Configuration for tuple generation."""

    strategy: TupleStrategy = TupleStrategy.CROSS_PRODUCT
    count: int = 20
    seed: int = 0


@dataclass(frozen=True)
class DSpyConfig:
    """Configuration for DSPy-based query generation."""

    optimize: bool = False
    # Which implicit DSPy optimizer to use when `optimize=True` (or when goal-guided
    # optimization auto-creates an optimizer). Defaults to GEPA.
    optimizer_name: Literal["gepa", "miprov2"] = "gepa"
    optimizer: DSpyOptimizer | None = None
    trainset: Sequence[Any] | None = None
    valset: Sequence[Any] | None = None

    # Goal-guided compile-time optimization (when no explicit optimizer is provided).
    goal_guided: bool = False
    judge_backend: JudgeBackend = JudgeBackend.LLM


@dataclass(frozen=True)
class QueryConfig:
    """Configuration for query generation."""

    mode: QueryMode = QueryMode.INSTRUCTOR
    dspy: DSpyConfig | None = None


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
        *,
        optimize_dspy: bool,
        dspy_optimizer_name: Literal["gepa", "miprov2"],
        dspy_optimizer: DSpyOptimizer | None,
        dspy_trainset: Sequence[Any] | None,
        dspy_valset: Sequence[Any] | None,
    ) -> QueryGenerator:
        """Build a query generator based on the specified mode."""
        if mode == QueryMode.INSTRUCTOR:
            return InstructorQueryGenerator(self.client)
        if mode == QueryMode.DSPY:
            from evaluateur.generators.queries import DSpyQueryGenerator

            return DSpyQueryGenerator(
                self.client,
                optimize=optimize_dspy,
                optimizer_name=dspy_optimizer_name,
                optimizer=dspy_optimizer,
                trainset=dspy_trainset,
                valset=dspy_valset,
            )
        if mode == QueryMode.HYBRID:
            from evaluateur.generators.queries import HybridQueryGenerator

            return HybridQueryGenerator(self.client)
        raise ValueError(f"Unsupported query mode: {mode}")

    async def _ensure_options(self, maybe_options: BaseModel | None) -> BaseModel:
        """Ensure options are available, generating them if not provided."""
        if maybe_options is not None:
            return maybe_options
        return await self.options()

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
    ) -> tuple[GoalSpec | None, str | None]:
        if goals is None:
            return None, None
        if isinstance(goals, str):
            spec = await GoalSpec.from_text(self.client, goals)
        elif isinstance(goals, GoalSpec):
            spec = goals
        else:  # pragma: no cover - defensive
            raise TypeError("goals must be a GoalSpec, a string, or None")

        if spec.is_empty():
            return None, None
        return spec, spec.render_prompt()

    def _resolve_dspy_optimizer(
        self,
        dspy: DSpyConfig,
        *,
        mode: QueryMode,
        goal_spec: GoalSpec | None,
        goal_prompt: str | None,
    ) -> tuple[DSpyOptimizer | None, GoalSpec | None, str | None]:
        resolved_goal_spec = goal_spec
        resolved_goal_prompt = goal_prompt
        resolved_optimizer = dspy.optimizer

        if (
            resolved_optimizer is None
            and dspy.goal_guided
            and mode in (QueryMode.DSPY, QueryMode.HYBRID)
        ):
            if resolved_goal_spec is None:
                resolved_goal_spec = GoalSpec(
                    title="Goal-guided (no explicit goals provided)"
                )
                resolved_goal_prompt = resolved_goal_spec.render_prompt()
            resolved_optimizer = GoalGuidedQueryOptimizer(
                goal_spec=resolved_goal_spec,
                judge_backend=dspy.judge_backend,
                optimizer_name=dspy.optimizer_name,
            )

        return resolved_optimizer, resolved_goal_spec, resolved_goal_prompt

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
    ) -> AsyncIterator[GeneratedQuery]:
        log.info(
            "Generating queries: mode=%s, tuple_count=%s",
            config.mode.value,
            "streaming" if hasattr(tuples, "__aiter__") else len(tuples),  # type: ignore[arg-type]
        )

        goal_spec, goal_prompt = await self._normalize_goals(goals)

        dspy_cfg = config.dspy or DSpyConfig()
        resolved_optimizer, goal_spec, goal_prompt = self._resolve_dspy_optimizer(
            dspy_cfg,
            mode=config.mode,
            goal_spec=goal_spec,
            goal_prompt=goal_prompt,
        )
        effective_context = compose_context(self.context, goal_prompt)

        query_gen = self._build_query_generator(
            config.mode,
            optimize_dspy=dspy_cfg.optimize,
            dspy_optimizer_name=dspy_cfg.optimizer_name,
            dspy_optimizer=resolved_optimizer,
            dspy_trainset=dspy_cfg.trainset,
            dspy_valset=dspy_cfg.valset,
        )

        run_metadata = QueryMetadata(
            mode=config.mode.value,
            goal_guided=bool(goal_prompt),
            query_goals=(
                goal_spec.to_metadata()
                if goal_spec is not None and not goal_spec.is_empty()
                else None
            ),
        )

        async for q in query_gen.generate(
            self._aiter_tuples(tuples), effective_context
        ):
            merged = {
                **q.metadata.model_dump(),
                **run_metadata.model_dump(exclude_none=True),
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
    ) -> AsyncIterator[GeneratedQuery]:
        """Convenience wrapper: options → tuples → queries (streaming)."""

        options_instance = await self._ensure_options(options)
        tuple_iter = self.tuples(options_instance, config=tuple_config)
        async for q in self.queries(
            tuples=tuple_iter, config=query_config, goals=goals
        ):
            yield q
