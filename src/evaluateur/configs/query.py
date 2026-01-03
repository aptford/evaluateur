from __future__ import annotations

from dataclasses import dataclass

from evaluateur.generators import QueryMode
from evaluateur.goals import GoalMode


@dataclass(frozen=True)
class QueryConfig:
    """Configuration for query generation."""

    mode: QueryMode = QueryMode.INSTRUCTOR
    #: Optional instructions that guide *query writing* (natural language).
    #:
    #: These instructions are appended to the evaluator context for query
    #: generation (e.g. "Keep questions short and specific.").
    instructions: str | None = None
    goal_mode: GoalMode = "sample"
    goal_seed: int = 0
    #: Optional cap for goal prompt length.
    #:
    #: This value is passed to `GoalSpec.render_prompt(..., max_chars=...)` and
    #: `GoalSpec.render_focused_prompt(..., max_chars=...)` so large goal specs
    #: can be truncated deterministically. It does **not** truncate other prompt
    #: text (e.g. domain context or user instructions).
    max_chars_for_goals: int | None = None


