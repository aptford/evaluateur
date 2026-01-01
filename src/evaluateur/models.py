from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evaluateur.goals import GoalSpec
from evaluateur.types import ScalarValue


ModelT = TypeVar("ModelT", bound=BaseModel)


class GeneratedTuple(BaseModel, Generic[ModelT]):
    """A concrete combination of dimension values.

    The keys in ``values`` correspond to field names on the original query
    model, and the values are the selected option for that field.

    The generic parameter ``ModelT`` represents the source model type,
    preserving type information for downstream consumers.
    """

    values: dict[str, ScalarValue]


class QueryMetadata(BaseModel):
    """Metadata associated with a generated query.

    This includes both:
    - run-level metadata injected by the evaluator (mode, goal_guided, query_goals)
    - per-query metadata set by generators (free-form keys)

    Extra keys are allowed for experimentation and backend-specific tracing.
    """

    model_config = ConfigDict(extra="allow")

    # Run-level fields (injected by Evaluator.queries)
    mode: Literal["instructor"] | None = None
    goal_guided: bool = False
    query_goals: GoalSpec | None = None


class GeneratedQuery(BaseModel):
    """Natural language query with full traceability back to its tuple."""

    query: str
    source_tuple: GeneratedTuple
    metadata: QueryMetadata = Field(default_factory=QueryMetadata)
