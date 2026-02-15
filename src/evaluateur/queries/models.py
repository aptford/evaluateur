from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from evaluateur.goals.models import GoalMode, GoalSpec
from evaluateur.tuples.models import GeneratedTuple


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
