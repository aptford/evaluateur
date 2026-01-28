from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from evaluateur.client import LLMClient
from evaluateur.goal_parsing import build_goal_spec_messages
from evaluateur.goal_rendering import render_focused_goal_prompt, render_goal_prompt


GoalFocusArea = Literal["components", "trajectories", "outcomes"]
GoalMode = Literal["full", "sample"]


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


class GoalSpec(BaseModel):
    """User-provided guidance for shaping evaluation queries."""

    components: GoalLayer = Field(default_factory=GoalLayer)
    trajectories: GoalLayer = Field(default_factory=GoalLayer)
    outcomes: GoalLayer = Field(default_factory=GoalLayer)

    @classmethod
    async def from_text(cls, client: LLMClient, text: str) -> GoalSpec:
        """Parse free-form user text into a structured `GoalSpec`.

        This uses Instructor + Pydantic parsing, so callers get a stable schema.
        """

        cleaned = text.strip()
        if not cleaned:
            return cls()
        messages = build_goal_spec_messages(cleaned)
        if not messages:
            return cls()

        inst = client.instructor_client
        parsed: GoalSpec = await inst.chat.completions.create(
            model=client.model_name,
            response_model=cls,
            messages=messages,
        )
        return parsed

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

        def _has_any(layer: GoalLayer) -> bool:
            if layer.summary and layer.summary.strip():
                return True
            return any(it.weight > 0 for it in layer.items)

        areas: list[GoalFocusArea] = []
        if _has_any(self.components):
            areas.append("components")
        if _has_any(self.trajectories):
            areas.append("trajectories")
        if _has_any(self.outcomes):
            areas.append("outcomes")
        return areas

    def render_focused_prompt(self, *, focus_area: GoalFocusArea) -> str:
        """Render only a single goal layer (components/trajectories/outcomes)."""
        return render_focused_goal_prompt(self, focus_area=focus_area)

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable metadata representation."""

        return self.model_dump(exclude_none=True)
