"""Goal planning policy.

This module contains the policy decisions for goal guidance planning,
separating them from the mechanism (execution) in planning.py.
"""

from __future__ import annotations

from typing import Literal

from evaluateur.goals.models import GoalMode, GoalSpec


# Type alias for plan builder functions
PlanBuilderType = Literal["sample", "cycle", "full"]


def select_plan_type(
    goal_mode: GoalMode,
    goal_spec: GoalSpec | None,
) -> PlanBuilderType:
    """Select which plan builder to use based on goal mode and available goals.

    Args:
        goal_mode: The requested goal guidance mode ("sample", "cycle", or "full").
        goal_spec: The parsed goal specification, or None.

    Returns:
        The plan builder type to use: "sample", "cycle", or "full".
    """
    if (
        goal_mode in ("sample", "cycle")
        and goal_spec is not None
        and goal_spec.available_goals()
    ):
        return goal_mode  # type: ignore[return-value]
    return "full"


def should_include_goal_spec_in_full_plan(goal_spec: GoalSpec | None) -> bool:
    """Determine if goal spec should be included in a full plan.

    Args:
        goal_spec: The parsed goal specification, or None.

    Returns:
        True if the goal spec should be included.
    """
    return goal_spec is not None


def should_warn_no_goals(
    goal_mode: GoalMode,
    goal_spec: GoalSpec | None,
) -> bool:
    """Determine if we should warn about missing goals.

    Args:
        goal_mode: The requested goal guidance mode.
        goal_spec: The parsed goal specification, or None.

    Returns:
        True if a warning should be logged.
    """
    return (
        goal_mode in ("sample", "cycle")
        and goal_spec is not None
        and not goal_spec.available_goals()
    )
