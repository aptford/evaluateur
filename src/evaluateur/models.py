from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evaluateur.goals import GoalFocusArea, GoalMode, GoalSpec
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
    goal_mode: GoalMode | None = None
    goal_focus_area: GoalFocusArea | None = None

    @classmethod
    def merge(
        cls,
        *,
        run_metadata: QueryMetadata,
        per_query_metadata: QueryMetadata,
    ) -> QueryMetadata:
        """Merge evaluator run metadata with per-query metadata.

        Run metadata provides defaults; per-query metadata wins on conflicts.
        """

        merged = {
            **run_metadata.model_dump(exclude_none=True),
            **per_query_metadata.model_dump(exclude_none=True, exclude_unset=True),
        }
        return cls.model_validate(merged)


class GeneratedQuery(BaseModel):
    """Natural language query with full traceability back to its tuple."""

    query: str
    source_tuple: GeneratedTuple
    metadata: QueryMetadata = Field(default_factory=QueryMetadata)
