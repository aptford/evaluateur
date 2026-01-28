"""Goal planning policy.

This module contains the policy decisions for goal guidance planning,
separating them from the mechanism (execution) in planning.py.
"""

from __future__ import annotations

from typing import Callable, Literal

from evaluateur.goals.models import GoalFocusArea, GoalMode, GoalSpec


# Type alias for plan builder functions
PlanBuilderType = Literal["sample", "full"]


def select_plan_type(
    goal_mode: GoalMode,
    goal_spec: GoalSpec | None,
    focus_areas: list[GoalFocusArea],
) -> PlanBuilderType:
    """Select which plan builder to use based on goal mode and available data.

    This function encapsulates the policy decision of which planning approach
    to use. The mechanism (actual plan building) is handled by the planning module.

    Args:
        goal_mode: The requested goal guidance mode ("sample" or "full").
        goal_spec: The parsed goal specification, or None.
        focus_areas: Available focus areas from the goal spec.

    Returns:
        The plan builder type to use: "sample" or "full".
    """
    if goal_mode == "sample" and goal_spec is not None and focus_areas:
        return "sample"
    return "full"


def should_include_goal_spec_in_full_plan(goal_spec: GoalSpec | None) -> bool:
    """Determine if goal spec should be included in a full plan.

    Args:
        goal_spec: The parsed goal specification, or None.

    Returns:
        True if the goal spec should be included.
    """
    return goal_spec is not None


def should_warn_no_focus_areas(
    goal_mode: GoalMode,
    goal_spec: GoalSpec | None,
    focus_areas: list[GoalFocusArea],
) -> bool:
    """Determine if we should warn about missing focus areas.

    Args:
        goal_mode: The requested goal guidance mode.
        goal_spec: The parsed goal specification, or None.
        focus_areas: Available focus areas from the goal spec.

    Returns:
        True if a warning should be logged.
    """
    return goal_mode == "sample" and goal_spec is not None and not focus_areas
