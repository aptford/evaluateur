"""Goal prompt rendering.

Renders GoalSpec and individual Goals into compact prompt strings
for query generation context.
"""

from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluateur.goals.models import Goal, GoalSpec


def _render_goal_line(goal: "Goal") -> str:
    """Render a single goal as a bullet line."""
    parts: list[str] = []
    if goal.name:
        parts.append(goal.name)
    parts.append(goal.text)
    return "- " + " - ".join(parts)


def render_goal_prompt(spec: "GoalSpec") -> str:
    """Render a GoalSpec into a compact instruction block.

    Goals are grouped by category when categories are present.
    Uncategorized goals appear under a general heading.
    """
    active = [g for g in spec.goals if g.weight > 0]
    if not active:
        return ""

    chunks: list[str] = ["Query optimization goals:"]

    # Separate categorized from uncategorized
    categorized = [g for g in active if g.category]
    uncategorized = [g for g in active if not g.category]

    if categorized:
        # Group by category, preserving input order within each group
        for category, goals in groupby(
            sorted(categorized, key=lambda g: g.category), key=lambda g: g.category
        ):
            label = category.capitalize()
            chunks.append(f"\n{label}:")
            for goal in goals:
                chunks.append(_render_goal_line(goal))

    if uncategorized:
        if categorized:
            chunks.append("\nGeneral:")
        for goal in uncategorized:
            chunks.append(_render_goal_line(goal))

    return "\n".join(chunks).strip() + "\n"


def render_focused_goal_prompt(goal: "Goal") -> str:
    """Render a single Goal as the focus for a query."""
    parts: list[str] = []

    label = goal.category.capitalize() if goal.category else "Goal"
    focus_name = goal.name or goal.text[:40]
    parts.append(f"Query optimization goals (focus: {focus_name}):")

    if goal.category:
        parts.append(f"\n{label}:")

    parts.append(_render_goal_line(goal))

    return "\n".join(parts).strip() + "\n"
