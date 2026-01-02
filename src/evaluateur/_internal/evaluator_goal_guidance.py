from __future__ import annotations

import math
import random
from collections.abc import Mapping

from evaluateur.client import LLMClient
from evaluateur.generators.query.context import compose_query_context
from evaluateur.goals import GoalFocusArea, GoalLayer, GoalSpec
from evaluateur.models import GeneratedTuple


async def normalize_goal_spec(
    client: LLMClient, goals: GoalSpec | str | None
) -> GoalSpec | None:
    """Normalize `goals` into a non-empty GoalSpec, or return None."""

    if goals is None:
        return None
    if isinstance(goals, str):
        spec = await GoalSpec.from_text(client, goals)
    elif isinstance(goals, GoalSpec):
        spec = goals
    else:  # pragma: no cover - defensive
        raise TypeError("goals must be a GoalSpec, a string, or None")

    if spec.is_empty():
        return None
    return spec


def _layer_weight(layer: GoalLayer) -> float:
    """Compute the effective sampling weight for a goal layer.

    Rules:
    - Sum all GoalItem.weight values where weight > 0.
    - If there are no positive-weight items but the layer has a non-empty
      summary, default to weight 1.0 (so summaries can still be sampled).
    - Otherwise treat the layer as weight 0 (ineligible).
    """

    if layer.items:
        total = math.fsum(float(it.weight) for it in layer.items if float(it.weight) > 0.0)
        if total > 0.0:
            return total

    if isinstance(layer.summary, str) and layer.summary.strip():
        return 1.0
    return 0.0


def _focus_weight(spec: GoalSpec, focus: GoalFocusArea) -> float:
    if focus == "components":
        return _layer_weight(spec.components)
    if focus == "trajectories":
        return _layer_weight(spec.trajectories)
    return _layer_weight(spec.outcomes)


class GoalSamplingContextBuilder:
    """Per-tuple context builder that samples a goal focus area.

    This is used by `Evaluator.queries()` when `goal_mode == "sample"`.
    """

    def __init__(
        self,
        *,
        base_context: str,
        goal_spec: GoalSpec,
        focus_areas: list[GoalFocusArea],
        rng: random.Random,
        max_chars_for_goals: int | None,
    ) -> None:
        self._base_context = base_context
        self._goal_spec = goal_spec
        self._focus_areas = focus_areas
        self._rng = rng
        self._max_chars_for_goals = max_chars_for_goals

        weighted: list[tuple[GoalFocusArea, float]] = []
        for focus in focus_areas:
            w = _focus_weight(goal_spec, focus)
            if w > 0.0:
                weighted.append((focus, w))
        self._weighted_focus_areas = weighted

    def _choose_focus_area(self) -> GoalFocusArea:
        """Choose a focus area using numerically-stable weighted sampling."""

        # Defensive fallback: if all weights are effectively 0, fall back to
        # uniform sampling across eligible focus areas.
        if not self._weighted_focus_areas:
            return self._rng.choice(self._focus_areas)

        max_w = max(w for _, w in self._weighted_focus_areas)
        if not (max_w > 0.0) or not math.isfinite(max_w):
            return self._rng.choice(self._focus_areas)

        scaled = [w / max_w for _, w in self._weighted_focus_areas]
        total = math.fsum(scaled)
        if not (total > 0.0) or not math.isfinite(total):
            return self._rng.choice(self._focus_areas)

        r = self._rng.random() * total
        acc = 0.0
        for (focus, _), s in zip(self._weighted_focus_areas, scaled, strict=True):
            acc += s
            if r < acc:
                return focus
        # Floating-point edge case: return the last focus area.
        return self._weighted_focus_areas[-1][0]

    def __call__(self, t: GeneratedTuple) -> tuple[str, Mapping[str, object]]:
        _ = t
        focus = self._choose_focus_area()
        focus_prompt = self._goal_spec.render_focused_prompt(
            focus_area=focus,
            max_chars=self._max_chars_for_goals,
        )
        ctx = compose_query_context(self._base_context, goal_prompt=focus_prompt)
        return ctx, {"goal_focus_area": focus}


