from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence, Type, TypeVar

from pydantic import BaseModel

from evaluator.client import LLMClient
from evaluator.generators import OptionsGenerator, QueryMode, TupleStrategy
from evaluator.generators.queries import (
    DSpyOptimizer,
    DSpyQueryGenerator,
    HybridQueryGenerator,
    InstructorQueryGenerator,
    QueryGenerator,
)
from evaluator.generators.tuples import (
    CrossProductTupleGenerator,
    DirectLLMTupleGenerator,
    TupleGenerator,
)
from evaluator.models import EvaluatorOutput, GeneratedQuery, GeneratedTuple

log = logging.getLogger(__name__)

QueryModelT = TypeVar("QueryModelT", bound=BaseModel)


@dataclass
class Evaluator:
    """Synthetic query evaluator following the dimensions → tuples → queries flow.

    The evaluator is parameterised by a Pydantic ``BaseModel`` that describes the
    dimensions of a query (e.g. payer, age, complexity, geography). It can:

    - Generate *options* for each dimension using Instructor
    - Turn those options into tuples via cross-product or direct LLM generation
    - Convert tuples into natural language queries using Instructor, DSPy, or a hybrid
    """

    model: Type[QueryModelT]
    client: LLMClient
    context: str = ""

    def __init__(
        self,
        model: Type[QueryModelT],
        client: LLMClient | None = None,
        context: str = "",
        provider: str = "openai",
        model_name: str | None = "gpt-4o-mini",
    ) -> None:
        self.model = model
        self.client = client or LLMClient.from_env(
            provider=provider,
            model_name=model_name,
        )
        self.context = context
        self._options_generator = OptionsGenerator(self.client)
        log.debug(
            "Evaluator initialized: model=%s, provider=%s, model_name=%s",
            model.__name__,
            self.client.provider,
            self.client.model_name,
        )

    def generate_options(
        self,
        instructions: str | None = None,
        count_per_field: int = 5,
    ) -> BaseModel:
        """Generate an options ``BaseModel`` from the configured query model.

        Every simple field on the input model is turned into a sequence of
        options. Iterator fields (lists, tuples, etc.) are preserved.
        """
        log.info(
            "Generating options for %s (count_per_field=%d)",
            self.model.__name__,
            count_per_field,
        )
        result = self._options_generator.generate_options(
            self.model,
            instructions=instructions,
            count_per_field=count_per_field,
        )
        log.debug("Generated options: %s", result)
        return result

    def _build_tuple_generator(self, strategy: TupleStrategy) -> TupleGenerator:
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
        dspy_optimizer: DSpyOptimizer | None,
        dspy_trainset: Sequence[Any] | None,
        dspy_valset: Sequence[Any] | None,
    ) -> QueryGenerator:
        if mode == QueryMode.INSTRUCTOR:
            return InstructorQueryGenerator(self.client)
        if mode == QueryMode.DSPY:
            return DSpyQueryGenerator(
                self.client,
                optimize=optimize_dspy,
                optimizer=dspy_optimizer,
                trainset=dspy_trainset,
                valset=dspy_valset,
            )
        if mode == QueryMode.HYBRID:
            return HybridQueryGenerator(self.client)
        raise ValueError(f"Unsupported query mode: {mode}")

    def _ensure_options(self, maybe_options: BaseModel | None) -> BaseModel:
        if maybe_options is not None:
            return maybe_options
        # If no options are provided, we generate them automatically.
        return self.generate_options()

    def generate_queries(
        self,
        options: BaseModel | None = None,
        *,
        mode: QueryMode = QueryMode.INSTRUCTOR,
        tuple_strategy: TupleStrategy = TupleStrategy.CROSS_PRODUCT,
        tuple_count: int = 20,
        optimize_dspy: bool = False,
        dspy_optimizer: DSpyOptimizer | None = None,
        dspy_trainset: Sequence[Any] | None = None,
        dspy_valset: Sequence[Any] | None = None,
    ) -> EvaluatorOutput:
        """Generate natural language queries based on the model and options.

        Parameters
        ----------
        options:
            A model instance whose fields are iterables of concrete values. If
            omitted, the evaluator will first generate options automatically
            using Instructor.
        mode:
            How to turn tuples into queries (Instructor, DSPy, or hybrid).
        tuple_strategy:
            How to generate tuples from the options (cross product or direct LLM).
        tuple_count:
            Target number of tuples to generate.
        optimize_dspy:
            Whether to enable default DSPy optimisation in DSPy-based modes.
        dspy_optimizer:
            Optional custom DSPy optimizer / teleprompter. Any object with a
            ``compile(student, trainset, valset, **kwargs)`` method can be
            used, for example ``dspy.MIPROv2`` or ``dspy.BootstrapFewShot``.
        dspy_trainset:
            Optional training set passed to the optimiser's ``compile``.
        dspy_valset:
            Optional validation set passed to the optimiser's ``compile``.
        """

        log.info(
            "Generating queries: mode=%s, tuple_strategy=%s, tuple_count=%d",
            mode.value,
            tuple_strategy.value,
            tuple_count,
        )

        options_instance = self._ensure_options(options)
        log.debug("Using options: %s", options_instance)

        tuple_gen = self._build_tuple_generator(tuple_strategy)
        tuples: list[GeneratedTuple] = tuple_gen.generate(options_instance, tuple_count)
        log.info("Generated %d tuples", len(tuples))

        query_gen = self._build_query_generator(
            mode,
            optimize_dspy=optimize_dspy,
            dspy_optimizer=dspy_optimizer,
            dspy_trainset=dspy_trainset,
            dspy_valset=dspy_valset,
        )
        queries: list[GeneratedQuery] = query_gen.generate(tuples, self.context)
        log.info("Generated %d queries", len(queries))

        return EvaluatorOutput(tuples=tuples, queries=queries, metadata={})
