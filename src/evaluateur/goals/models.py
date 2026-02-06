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

GoalFocusArea = Literal["components", "trajectories", "outcomes"]
GoalMode = Literal["full", "sample", "cycle"]


class GoalItem(BaseModel):
    """A single user goal used to guide query optimization.

    Goals are intentionally flexible: they can represent checklists (binary),
    weighted preferences, or concrete inclusion/avoidance constraints.
    """

    name: str = Field(..., description="Short goal name")
    description: str | None = Field(
        default=None, description="Plain-language description of the goal"
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Relative importance. 0 disables the goal without deleting it.",
    )

    must_include: list[str] = Field(
        default_factory=list,
        description=(
            "Tokens/phrases/requirements the query should explicitly include. "
            "Use for checklist-style constraints."
        ),
    )
    avoid: list[str] = Field(
        default_factory=list,
        description="Tokens/phrases/requirements the query should avoid.",
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Optional examples of queries that satisfy this goal.",
    )

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, v: float) -> float:
        """Ensure weight is a finite, non-negative float.

        Note: 0.0 is allowed and used as a "disabled goal" sentinel.
        """

        # Pydantic will typically coerce int/str inputs to float before validators.
        # Keep this explicit to ensure consistent output type.
        v = float(v)
        if not math.isfinite(v):
            raise ValueError("weight must be a finite number")
        if v < 0:
            raise ValueError("weight must be >= 0")
        return v


class GoalLayer(BaseModel):
    """A set of goals for one framework layer (components/trajectories/outcomes)."""

    summary: str | None = Field(
        default=None,
        description="Optional one-paragraph summary of what matters in this layer.",
    )
    items: list[GoalItem] = Field(default_factory=list)

    def weight(self) -> float:
        """Return the effective weight for this layer."""
        if self.items:
            total = math.fsum(
                float(it.weight) for it in self.items if float(it.weight) > 0.0
            )
            if total > 0.0:
                return total
        if isinstance(self.summary, str) and self.summary.strip():
            return 1.0
        return 0.0

    def has_content(self) -> bool:
        """Return True if the layer has any active goals or summary text."""
        if isinstance(self.summary, str) and self.summary.strip():
            return True
        return any(it.weight > 0 for it in self.items)


class GoalSpec(BaseModel):
    """User-provided guidance for shaping evaluation queries."""

    components: GoalLayer = Field(default_factory=GoalLayer)
    trajectories: GoalLayer = Field(default_factory=GoalLayer)
    outcomes: GoalLayer = Field(default_factory=GoalLayer)

    def is_empty(self) -> bool:
        """Return True if no goals are specified."""

        return (
            not self.components.items
            and not self.trajectories.items
            and not self.outcomes.items
            and not (
                self.components.summary
                or self.trajectories.summary
                or self.outcomes.summary
            )
        )

    def render_prompt(self) -> str:
        """Render this spec into a compact instruction block."""
        return render_goal_prompt(self)

    def available_focus_areas(self) -> list[GoalFocusArea]:
        """Return the goal layers that are non-empty (considering weights)."""

        areas: list[GoalFocusArea] = []
        if self.components.has_content():
            areas.append("components")
        if self.trajectories.has_content():
            areas.append("trajectories")
        if self.outcomes.has_content():
            areas.append("outcomes")
        return areas

    def focus_weight(self, focus_area: GoalFocusArea) -> float:
        """Return the effective weight for a focus area."""
        if focus_area == "components":
            return self.components.weight()
        if focus_area == "trajectories":
            return self.trajectories.weight()
        return self.outcomes.weight()

    def render_focused_prompt(self, *, focus_area: GoalFocusArea) -> str:
        """Render only a single goal layer (components/trajectories/outcomes)."""
        return render_focused_goal_prompt(self, focus_area=focus_area)

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
    """Prepared focus areas and weights for sampling."""

    choices: list[GoalFocusArea]
    weights: list[float]
