from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from evaluateur.client import resolve_client
from evaluateur.config import DEFAULT_CONFIG, EvaluatorConfig
from evaluateur.goals.models import GoalMode, GoalSpec
from evaluateur.goals.planning import plan_goal_guidance
from evaluateur.options import OptionsGenerator
from evaluateur.queries import (
    GeneratedQuery,
    GeneratedTuple,
    QueryMode,
    build_query_generator,
)
from evaluateur.queries.merge import merge_query_metadata
from evaluateur.tuples import TupleStrategy, build_tuple_generator
from evaluateur.utils import to_async_iterator

log = logging.getLogger(__name__)

QueryModelT = TypeVar("QueryModelT", bound=BaseModel)


class Evaluator:
    """Async synthetic evaluation helper following the dimensions → tuples → queries flow.

    The evaluator is parameterized by a Pydantic model that describes the
    dimensions of a query (e.g. payer, age, complexity, geography).

    Parameters
    ----------
    model
        A Pydantic model class describing query dimensions.
    llm
        A ``"provider/model-name"`` string, e.g. ``"openai/gpt-4.1-mini"``
        or ``"anthropic/claude-3-5-sonnet-latest"``.  Mutually exclusive
        with *client*.  When omitted, reads the ``EVALUATEUR_MODEL`` env
        var (default: ``"openai/gpt-4.1-mini"``).
    client
        A pre-configured async Instructor client for advanced use cases
        (observability wrappers, custom providers).  Must be paired with
        *model_name*.  Mutually exclusive with *llm*.
    model_name
        The model identifier passed to ``chat.completions.create(model=...)``.
        Required when *client* is provided, ignored otherwise.
    config
        Optional :class:`EvaluatorConfig` for default parameter values.

    Examples
    --------
    Simple usage::

        evaluator = Evaluator(MyQuery, llm="openai/gpt-4.1-mini")

    Switch providers::

        evaluator = Evaluator(MyQuery, llm="anthropic/claude-3-5-sonnet-latest")

    Default from environment (reads ``EVALUATEUR_MODEL``)::

        evaluator = Evaluator(MyQuery)

    Advanced — bring your own Instructor client::

        import instructor
        from openai import AsyncOpenAI

        inst = instructor.from_openai(AsyncOpenAI())
        evaluator = Evaluator(MyQuery, client=inst, model_name="gpt-4o")
    """

    def __init__(
        self,
        model: Type[QueryModelT],
        *,
        llm: str | None = None,
        client: Any | None = None,
        model_name: str | None = None,
        config: EvaluatorConfig | None = None,
    ) -> None:
        self.model = model
        self._client = resolve_client(llm=llm, client=client, model_name=model_name)
        self.config = config or DEFAULT_CONFIG
        log.debug(
            "Evaluator initialized: model=%s, llm=%s",
            model.__name__,
            self._client.model_name,
        )

    async def options(
        self, *, instructions: str | None = None, count_per_field: int | None = None
    ) -> BaseModel:
        """Generate an options ``BaseModel`` from the configured query model.

        Every simple field on the input model is turned into a sequence of
        options. Iterator fields (lists, tuples, etc.) are preserved.

        Parameters default to config values if not provided.
        """
        effective_count = (
            count_per_field
            if count_per_field is not None
            else self.config.options_count_per_field
        )
        log.info(
            "Generating options for %s (n=%d)",
            self.model.__name__,
            effective_count,
        )
        options_generator = OptionsGenerator(self._client)
        result = await options_generator.generate_options(
            self.model,
            instructions=instructions,
            count_per_field=effective_count,
        )
        log.debug("Generated options: %s", result)
        return result

    async def _ensure_options(
        self,
        maybe_options: BaseModel | None,
        *,
        instructions: str | None,
        count_per_field: int | None,
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
        strategy: TupleStrategy | None = None,
        count: int | None = None,
        seed: int | None = None,
        instructions: str | None = None,
    ) -> AsyncIterator[GeneratedTuple]:
        """Generate tuples as an async iterator, yielding one at a time.

        Instructions are forwarded to tuple generators that support them.
        Parameters default to config values if not provided.
        """
        effective_strategy = (
            strategy if strategy is not None else self.config.get_tuple_strategy()
        )
        effective_count = count if count is not None else self.config.tuples_count
        effective_seed = seed if seed is not None else self.config.tuples_seed

        log.info(
            "Generating tuples: strategy=%s, count=%d",
            effective_strategy.value,
            effective_count,
        )

        log.debug("Using options: %s", options)

        tuple_gen = build_tuple_generator(client=self._client, strategy=effective_strategy)
        generated_count = 0

        async for t in tuple_gen.generate(
            options, effective_count, seed=effective_seed, instructions=instructions
        ):
            generated_count += 1
            yield t

        log.info("Generated %d tuples", generated_count)

    async def queries(
        self,
        *,
        tuples: Sequence[GeneratedTuple] | AsyncIterator[GeneratedTuple],
        instructions: str | None = None,
        goal_mode: GoalMode | None = None,
        query_mode: QueryMode | None = None,
        seed: int | None = None,
        goals: GoalSpec | str | None = None,
    ) -> AsyncIterator[GeneratedQuery]:
        """Generate natural language queries from tuples.

        Parameters
        ----------
        tuples
            Sequence or async stream of tuples to turn into queries.
        instructions
            Optional instructions for query generation.
        goal_mode
            Goal guidance mode ("sample", "cycle", or "full"). Defaults to config.
        query_mode
            Query generator mode. Defaults to config.
        seed
            Random seed for goal sampling. Defaults to config.
        goals
            Goal specification for guided query generation.
        """
        effective_goal_mode = (
            goal_mode if goal_mode is not None else self.config.get_goal_mode()
        )
        effective_query_mode = (
            query_mode if query_mode is not None else self.config.get_query_mode()
        )
        effective_seed = seed if seed is not None else self.config.tuples_seed

        log.info(
            "Generating queries: tuple_count=%s",
            "streaming" if hasattr(tuples, "__aiter__") else len(tuples),  # type: ignore[arg-type]
        )

        guidance = await plan_goal_guidance(
            client=self._client,
            goals=goals,
            goal_mode=effective_goal_mode,
            instructions=instructions,
            seed=effective_seed,
        )

        query_gen = build_query_generator(client=self._client, mode=effective_query_mode)

        async for q in query_gen.generate(
            to_async_iterator(tuples),
            guidance.context,
            context_builder=guidance.context_builder,
        ):
            yield GeneratedQuery(
                query=q.query,
                source_tuple=q.source_tuple,
                metadata=merge_query_metadata(
                    run_metadata=guidance.run_metadata,
                    per_query_metadata=q.metadata,
                ),
            )

    async def run(
        self,
        *,
        options: BaseModel | None = None,
        instructions: str | None = None,
        count_per_field: int | None = None,
        tuple_strategy: TupleStrategy | None = None,
        tuple_count: int | None = None,
        seed: int | None = None,
        goal_mode: GoalMode | None = None,
        query_mode: QueryMode | None = None,
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
            Number of options to generate per field. Defaults to config.
        tuple_strategy
            Tuple sampling strategy. Defaults to config.
        tuple_count
            Number of tuples to generate. Defaults to config.
        seed
            Random seed for tuple sampling and goal sampling. Defaults to config.
        goal_mode
            Goal guidance mode ("sample", "cycle", or "full"). Defaults to config.
        query_mode
            Query generator mode. Defaults to config.
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
            instructions=instructions,
            goal_mode=goal_mode,
            query_mode=query_mode,
            seed=seed,
            goals=goals,
        ):
            yield q
