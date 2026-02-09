from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from evaluateur.goals.models import GoalMode, GoalSpec
from evaluateur.options.types import ScalarValue


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

    goal_guided: bool = False
    query_goals: GoalSpec | None = None
    goal_mode: GoalMode | None = None
    goal_focus: str | None = None
    goal_category: str | None = None


class GeneratedQuery(BaseModel):
    """Natural language query with full traceability back to its tuple."""

    query: str
    source_tuple: GeneratedTuple
    metadata: QueryMetadata = Field(default_factory=QueryMetadata)
