from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from evaluator.types import ScalarValue


ModelT = TypeVar("ModelT", bound=BaseModel)


class GeneratedTuple(BaseModel, Generic[ModelT]):
    """A concrete combination of dimension values.

    The keys in ``values`` correspond to field names on the original query
    model, and the values are the selected option for that field.

    The generic parameter ``ModelT`` represents the source model type,
    preserving type information for downstream consumers.
    """

    values: dict[str, ScalarValue]


class GeneratedQuery(BaseModel):
    """Natural language query with full traceability back to its tuple."""

    query: str
    source_tuple: GeneratedTuple
    metadata: dict[str, Any] = {}


class EvaluatorOutput(BaseModel):
    """Full structured output of the evaluator.

    This keeps both the intermediate tuples and the final queries so that
    downstream evaluation code can reason about coverage and failure modes.
    """

    tuples: list[GeneratedTuple]
    queries: list[GeneratedQuery]
    metadata: dict[str, Any] = {}
