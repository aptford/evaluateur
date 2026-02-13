"""Goal prompt rendering.

Renders GoalSpec and individual Goals into structured prompt blocks
for query generation context, using XML tags for clear delineation.
"""

from __future__ import annotations

import math
from itertools import groupby
from typing import TYPE_CHECKING

from evaluateur.constants import (
    TAG_GOAL_CLOSE,
    TAG_GOAL_OPEN,
    TAG_GOALS_CLOSE,
    TAG_GOALS_OPEN,
)

if TYPE_CHECKING:
    from evaluateur.goals.models import Goal, GoalSpec

# Preamble for full-mode (multiple goals)
_FULL_PREAMBLE = (
    "Each goal describes a behavior or failure mode observed in the system under test.\n"
    "Shape the query so a real user would naturally trigger the described scenario."
)

# Preamble for focused-mode (single goal)
_FOCUSED_PREAMBLE = (
    "This goal describes a specific behavior to stress-test.\n"
    "Shape the query so a real user would naturally trigger this scenario."
)


def _render_goal_lines(goal: "Goal", *, normalized_weight: float | None = None) -> str:
    """Render a single goal as Goal:/Description: lines.

    When *normalized_weight* is provided (0-1 range), it is shown as a
    percentage on the Goal line so the LLM can prioritise accordingly.
    """
    weight_suffix = ""
    if normalized_weight is not None:
        pct = round(normalized_weight * 100)
        weight_suffix = f" ({pct}%)"

    lines: list[str] = []
    if goal.name:
        lines.append(f"- Goal: {goal.name.capitalize()}{weight_suffix}")
        lines.append(f"  Description: {goal.text}")
    else:
        lines.append(f"- Description:{weight_suffix} {goal.text}")
    return "\n".join(lines)


def render_goal_prompt(spec: "GoalSpec") -> str:
    """Render a GoalSpec into a structured XML-tagged instruction block.

    Goals are grouped by category when categories are present.
    Uncategorized goals appear under a general heading.
    """
    active = [g for g in spec.goals if g.weight > 0]
    if not active:
        return ""

    # Compute normalized weights when there are multiple goals so the LLM
    # can see relative priority.  Single-goal specs skip the annotation.
    show_weights = len(active) > 1
    weight_map: dict[int, float] = {}
    if show_weights:
        total = math.fsum(g.weight for g in active)
        if total > 0 and math.isfinite(total):
            weight_map = {id(g): g.weight / total for g in active}

    def _lines(goal: "Goal") -> str:
        nw = weight_map.get(id(goal))
        return _render_goal_lines(goal, normalized_weight=nw)

    chunks: list[str] = [TAG_GOALS_OPEN, _FULL_PREAMBLE]

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
                chunks.append(_lines(goal))

    if uncategorized:
        if categorized:
            chunks.append("\nGeneral:")
        for goal in uncategorized:
            chunks.append(_lines(goal))

    chunks.append(TAG_GOALS_CLOSE)
    return "\n".join(chunks).strip()


def render_focused_goal_prompt(goal: "Goal") -> str:
    """Render a single Goal as a focused XML-tagged block."""
    lines: list[str] = [TAG_GOAL_OPEN, _FOCUSED_PREAMBLE]

    if goal.category:
        lines.append(f"\nCategory: {goal.category.capitalize()}")

    lines.append(_render_goal_lines(goal))
    lines.append(TAG_GOAL_CLOSE)

    return "\n".join(lines).strip()
