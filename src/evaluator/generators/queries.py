from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Protocol, Sequence

from pydantic import BaseModel

from evaluator.client import LLMClient
from evaluator.models import GeneratedQuery, GeneratedTuple

log = logging.getLogger(__name__)

try:
    import dspy  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    dspy = None  # type: ignore[assignment]


class QueryMode(str, Enum):
    """Available strategies for turning tuples into natural language queries."""

    INSTRUCTOR = "instructor"
    DSPY = "dspy"
    HYBRID = "hybrid"  # Instructor draft, DSPy refinement


class QueryGenerator(Protocol):
    """Protocol for query generators."""

    def generate(self, tuples: list[GeneratedTuple], context: str) -> list[GeneratedQuery]: ...


class DSpyOptimizer(Protocol):
    """Minimal protocol for DSPy optimizers/teleprompters.

    Any object with a ``compile`` method matching this interface can be used.
    """

    def compile(
        self,
        student: Any,
        trainset: Sequence[Any] | None = None,
        valset: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> Any: ...


class InstructorQueryGenerator:
    """Use Instructor directly to synthesize queries from tuples."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def generate(self, tuples: list[GeneratedTuple], context: str) -> list[GeneratedQuery]:
        client = self._client.instructor_client
        log.info("InstructorQueryGenerator: generating queries for %d tuples", len(tuples))

        class QueryListModel(BaseModel):
            queries: list[GeneratedQuery]

        tuple_descriptions = [
            f"- {t.values}" for t in tuples
        ]

        system_message = (
            "You create realistic user queries for evaluating an AI system. "
            "Each query should reflect the intent encoded in the tuple of dimension values."
        )
        user_message = (
            f"Domain/context:\n{context or 'General'}\n\n"
            "For each of the following tuples (dimension combinations), write a natural language "
            "query that a user might ask. Be specific and include enough detail to fully express "
            "the combination.\n\n"
            + "\n".join(tuple_descriptions)
        )
        log.debug("InstructorQueryGenerator prompt:\n%s", user_message)

        result: QueryListModel = client.chat.completions.create(  # type: ignore[assignment]
            model=self._client.model_name,
            response_model=QueryListModel,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )
        log.debug("InstructorQueryGenerator: received %d queries", len(result.queries))

        return result.queries


class _TupleToQuerySignature:
    """DSPy signature for converting a tuple into a query."""

    if dspy is not None:  # pragma: no branch
        class TupleToQuery(dspy.Signature):  # type: ignore[valid-type]
            """Convert a structured tuple into a natural language query."""

            context = dspy.InputField(desc="Domain or product context for the query.")
            tuple_json = dspy.InputField(desc="JSON representation of the tuple values.")
            query = dspy.OutputField(desc="A realistic natural language query.")


class DSpyQueryGenerator:
    """Use DSPy to turn tuples into queries, optionally with optimization."""

    def __init__(
        self,
        client: LLMClient,
        *,
        optimize: bool = False,
        optimizer: DSpyOptimizer | None = None,
        trainset: Sequence[Any] | None = None,
        valset: Sequence[Any] | None = None,
    ) -> None:
        if dspy is None:  # pragma: no cover - defensive
            raise RuntimeError("DSPy is not installed but DSPy query mode was requested.")

        self._client = client
        dspy.settings.configure(lm=client.dspy_lm or dspy.LM(f"{client.provider}/{client.model_name}"))
        log.debug("DSpyQueryGenerator: configured DSPy LM")

        # Simple chain-of-thought module for tuple -> query.
        base_module = dspy.ChainOfThought(_TupleToQuerySignature.TupleToQuery)  # type: ignore[attr-defined]
        self._module = base_module
        self._compiled_module: Any | None = None

        # If the caller supplied a custom optimizer, compile immediately.
        if optimizer is not None:
            log.info("DSpyQueryGenerator: compiling with custom optimizer")
            self._compiled_module = optimizer.compile(
                student=base_module,
                trainset=trainset,
                valset=valset,
            )
            log.debug("DSpyQueryGenerator: custom optimizer compilation complete")
        elif optimize:
            # Provide a sane default optimizer while still allowing callers to
            # plug in their own teleprompters. Metric and data are delegated
            # to the caller via ``trainset``/``valset`` when available.
            log.info("DSpyQueryGenerator: compiling with default MIPROv2 optimizer")
            try:
                default_optimizer = dspy.MIPROv2(  # type: ignore[attr-defined]
                    metric=lambda _ex, _pred, _truth: 0.0,
                    auto="light",
                )
                self._compiled_module = default_optimizer.compile(
                    student=base_module,
                    trainset=trainset or [],
                    valset=valset,
                )
                log.debug("DSpyQueryGenerator: default optimizer compilation complete")
            except Exception:
                # If default optimisation fails, fall back to the base module.
                log.warning("DSpyQueryGenerator: default optimizer failed, using base module")
                self._compiled_module = None

    def generate(self, tuples: list[GeneratedTuple], context: str) -> list[GeneratedQuery]:
        results: list[GeneratedQuery] = []
        module = self._compiled_module or self._module
        using_compiled = self._compiled_module is not None
        log.info(
            "DSpyQueryGenerator: generating queries for %d tuples (compiled=%s)",
            len(tuples),
            using_compiled,
        )

        for i, t in enumerate(tuples):
            tuple_json = t.model_dump_json()
            log.debug("DSpyQueryGenerator: processing tuple %d/%d", i + 1, len(tuples))
            pred = module(context=context, tuple_json=tuple_json)
            results.append(GeneratedQuery(query=pred.query, source_tuple=t, metadata={}))

        log.debug("DSpyQueryGenerator: generated %d queries", len(results))
        return results


class HybridQueryGenerator:
    """Hybrid strategy: Instructor drafts, DSPy refines.

    First uses Instructor to generate baseline queries, then (if DSPy is
    available) lets DSPy rewrite or enrich them. If DSPy is not installed this
    gracefully falls back to Instructor-only behaviour.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self._instructor_gen = InstructorQueryGenerator(client)

    def generate(self, tuples: list[GeneratedTuple], context: str) -> list[GeneratedQuery]:
        log.info("HybridQueryGenerator: generating base queries with Instructor")
        base_queries = self._instructor_gen.generate(tuples, context)

        if dspy is None or self._client.dspy_lm is None:
            log.debug("HybridQueryGenerator: DSPy unavailable, returning base queries")
            return base_queries

        # Lightweight refinement: ask DSPy to improve clarity/coverage.
        log.info("HybridQueryGenerator: refining %d queries with DSPy", len(base_queries))
        dspy.settings.configure(lm=self._client.dspy_lm)

        class RefineSignature(dspy.Signature):  # type: ignore[valid-type]
            """Refine an existing query for better evaluation coverage."""

            context = dspy.InputField()
            original_query = dspy.InputField()
            refined_query = dspy.OutputField()

        refiner = dspy.ChainOfThought(RefineSignature)

        refined: list[GeneratedQuery] = []
        for i, g in enumerate(base_queries):
            log.debug("HybridQueryGenerator: refining query %d/%d", i + 1, len(base_queries))
            pred = refiner(context=context, original_query=g.query)
            refined.append(
                GeneratedQuery(
                    query=getattr(pred, "refined_query", g.query),
                    source_tuple=g.source_tuple,
                    metadata={"refined": True},
                )
            )

        log.debug("HybridQueryGenerator: refinement complete, %d queries", len(refined))
        return refined

