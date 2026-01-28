from __future__ import annotations

import logging
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass

from evaluateur.client import LLMClient
from evaluateur.generators.query.context import compose_query_context
from evaluateur.generators.query.protocols import ContextBuilder
from evaluateur.goals import GoalFocusArea, GoalLayer, GoalMode, GoalSpec
from evaluateur.models import GeneratedTuple, QueryMetadata

log = logging.getLogger(__name__)


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
    """Compute the effective sampling weight for a goal layer."""

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
    """Per-tuple context builder that samples a goal focus area."""

    def __init__(
        self,
        *,
        base_context: str | None,
        goal_spec: GoalSpec,
        focus_areas: list[GoalFocusArea],
        rng: random.Random,
    ) -> None:
        self._base_context = base_context or ""
        self._goal_spec = goal_spec
        self._focus_areas = focus_areas
        self._rng = rng

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
        focus_prompt = self._goal_spec.render_focused_prompt(focus_area=focus)
        ctx = compose_query_context(self._base_context, goal_prompt=focus_prompt)
        return ctx, {"goal_focus_area": focus}


@dataclass(frozen=True)
class GoalGuidancePlan:
    """Resolved goal guidance for a query-generation run."""

    goal_spec: GoalSpec | None
    run_metadata: QueryMetadata
    context: str
    context_builder: ContextBuilder | None


def _validate_goal_mode(goal_mode: GoalMode) -> None:
    if goal_mode not in ("sample", "full"):
        raise ValueError(f"Unsupported goal mode: {goal_mode}")


async def plan_goal_guidance(
    *,
    client: LLMClient,
    goals: GoalSpec | str | None,
    goal_mode: GoalMode,
    instructions: str | None,
    seed: int,
) -> GoalGuidancePlan:
    """Build guidance details for a query run."""

    _validate_goal_mode(goal_mode)

    goal_spec = await normalize_goal_spec(client, goals)
    focus_areas: list[GoalFocusArea] = (
        goal_spec.available_focus_areas() if goal_spec is not None else []
    )
    goal_guided = bool(focus_areas)

    run_metadata = QueryMetadata(
        goal_guided=goal_guided,
        goal_mode=goal_mode,
        query_goals=(
            goal_spec
            if goal_spec is not None and not goal_spec.is_empty()
            else None
        ),
    )

    if goal_mode == "sample" and goal_spec is not None and focus_areas:
        rng = random.Random(seed)
        context_builder = GoalSamplingContextBuilder(
            base_context=instructions,
            goal_spec=goal_spec,
            focus_areas=focus_areas,
            rng=rng,
        )
        return GoalGuidancePlan(
            goal_spec=goal_spec,
            run_metadata=run_metadata,
            context=instructions or "",
            context_builder=context_builder,
        )

    if goal_mode == "sample" and goal_spec is not None and not focus_areas:
        log.debug("Goal sampling requested but no focus areas are available.")

    goal_prompt: str | None = None
    if goal_mode == "full" and goal_spec is not None:
        goal_prompt = goal_spec.render_prompt()

    context = compose_query_context(instructions or "", goal_prompt=goal_prompt)
    return GoalGuidancePlan(
        goal_spec=goal_spec,
        run_metadata=run_metadata,
        context=context,
        context_builder=None,
    )
