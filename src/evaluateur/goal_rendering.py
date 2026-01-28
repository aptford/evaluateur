from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from evaluateur.goals import GoalFocusArea, GoalItem, GoalLayer, GoalSpec


_FOCUS_LABELS: dict[str, str] = {
    "components": "Components",
    "trajectories": "Trajectories",
    "outcomes": "Outcomes",
}


def _render_items(items: Iterable["GoalItem"]) -> list[str]:
    lines: list[str] = []
    for it in items:
        if it.weight <= 0:
            continue
        parts: list[str] = [it.name]
        if it.description:
            parts.append(it.description)

        line = " - ".join(parts)
        extra: list[str] = []
        if it.must_include:
            extra.append(
                "must include: " + ", ".join(f"`{token}`" for token in it.must_include)
            )
        if it.avoid:
            extra.append("avoid: " + ", ".join(f"`{token}`" for token in it.avoid))
        if extra:
            line += " (" + "; ".join(extra) + ")"
        lines.append(f"- {line}")

        if it.examples:
            examples = [ex.strip() for ex in it.examples if ex.strip()]
            if examples:
                lines.append("  - examples:")
                lines.extend(f'    - "{ex}"' for ex in examples)
    return lines


def _render_layer(label: str, layer: "GoalLayer") -> list[str]:
    chunks: list[str] = [f"\n{label}:"]
    if layer.summary:
        chunks.append(textwrap.fill(layer.summary, width=96))
    chunks.extend(_render_items(layer.items))
    return chunks


def render_goal_prompt(spec: "GoalSpec") -> str:
    """Render a GoalSpec into a compact instruction block."""
    chunks: list[str] = ["Query optimization goals:"]

    if spec.components.summary or spec.components.items:
        chunks.extend(_render_layer(_FOCUS_LABELS["components"], spec.components))
    if spec.trajectories.summary or spec.trajectories.items:
        chunks.extend(_render_layer(_FOCUS_LABELS["trajectories"], spec.trajectories))
    if spec.outcomes.summary or spec.outcomes.items:
        chunks.extend(_render_layer(_FOCUS_LABELS["outcomes"], spec.outcomes))

    return "\n".join(chunks).strip() + "\n"


def render_focused_goal_prompt(
    spec: "GoalSpec", *, focus_area: "GoalFocusArea"
) -> str:
    """Render a single goal layer (components/trajectories/outcomes)."""
    label = _FOCUS_LABELS.get(str(focus_area), "Goals")
    layer = getattr(spec, str(focus_area))

    chunks: list[str] = [
        f"Query optimization goals (focus area: {label}):",
        f"\n{label}:",
    ]
    if layer.summary:
        chunks.append(textwrap.fill(layer.summary, width=96))
    chunks.extend(_render_items(layer.items))
    return "\n".join(chunks).strip() + "\n"
