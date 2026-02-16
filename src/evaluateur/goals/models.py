"""Goal specification models.

This module contains the data models for goal specifications.
Parsing logic is in the parsing module; rendering logic is in the rendering module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, field_validator

from evaluateur.goals.rendering import render_focused_goal_prompt, render_goal_prompt

if TYPE_CHECKING:
    from evaluateur.queries.models import QueryMetadata
    from evaluateur.queries.protocols import ContextBuilder

GoalMode = Literal["full", "sample", "cycle"]


class Goal(BaseModel):
    """A single evaluation goal used to guide query generation.

    Goals are flat, flexible, and optionally categorized. The ``category``
    field supports the CTO framework (components / trajectories / outcomes)
    or any user-defined string.
    """

    name: str = Field(default="", description="Short goal label")
    text: str = Field(..., description="Full goal description")
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Relative importance. 0 disables the goal without deleting it.",
    )
    category: str = Field(
        default="",
        description=("CTO category (one of 'components', 'trajectories', 'outcomes')"),
    )

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, v: float) -> float:
        """Ensure weight is a finite, non-negative float.

        Note: 0.0 is allowed and used as a "disabled goal" sentinel.
        """
        v = float(v)
        if not math.isfinite(v):
            raise ValueError("weight must be a finite number")
        if v < 0:
            raise ValueError("weight must be >= 0")
        return v


class GoalSpec(BaseModel):
    """User-provided guidance for shaping evaluation queries.

    A flat list of goals, optionally categorized. The CTO framework
    (components / trajectories / outcomes) is supported via the
    ``Goal.category`` field but is not structurally enforced.
    """

    goals: list[Goal] = Field(default_factory=list)

    def is_empty(self) -> bool:
        """Return True if no active goals are specified."""
        return not self.goals or all(g.weight <= 0 for g in self.goals)

    def available_goals(self) -> list[Goal]:
        """Return goals with positive weight."""
        return [g for g in self.goals if g.weight > 0]

    def render_prompt(self) -> str:
        """Render this spec into a compact instruction block."""
        return render_goal_prompt(self)

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable metadata representation."""
        return self.model_dump(exclude_none=True)


@dataclass(frozen=True)
class GoalGuidancePlan:
    """Resolved goal guidance for a query-generation run."""

    goal_spec: GoalSpec | None
    run_metadata: QueryMetadata
    context: str
    context_builder: ContextBuilder | None


@dataclass(frozen=True)
class GoalFocusPlan:
    """Prepared goals and weights for sampling."""

    choices: list[Goal]
    weights: list[float]
