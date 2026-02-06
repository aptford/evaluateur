from __future__ import annotations

import math
import random
from collections.abc import Mapping

from evaluateur.queries.context import compose_query_context
from evaluateur.goals.models import GoalFocusArea, GoalFocusPlan, GoalSpec
from evaluateur.queries.models import GeneratedTuple


class WeightedSampler:
    """Numerically-stable sampler for weighted categories."""

    def __init__(
        self,
        *,
        choices: list[GoalFocusArea],
        weights: list[float],
        rng: random.Random,
    ) -> None:
        if len(choices) != len(weights):
            raise ValueError("WeightedSampler choices/weights length mismatch")
        self._choices = choices
        self._weights = weights
        self._rng = rng

    def sample(self) -> GoalFocusArea:
        if not self._choices:
            raise ValueError("WeightedSampler requires at least one choice")

        max_w = max(self._weights, default=0.0)
        if not (max_w > 0.0) or not math.isfinite(max_w):
            return self._rng.choice(self._choices)

        scaled = [w / max_w for w in self._weights]
        total = math.fsum(scaled)
        if not (total > 0.0) or not math.isfinite(total):
            return self._rng.choice(self._choices)

        r = self._rng.random() * total
        acc = 0.0
        for choice, weight in zip(self._choices, scaled, strict=True):
            acc += weight
            if r < acc:
                return choice
        return self._choices[-1]


class GoalSamplingContextBuilder:
    """Per-tuple context builder that samples a goal focus area."""

    def __init__(
        self,
        *,
        base_context: str | None,
        goal_spec: GoalSpec,
        focus_plan: GoalFocusPlan,
        rng: random.Random,
    ) -> None:
        self._base_context = base_context or ""
        self._goal_spec = goal_spec

        self._sampler = WeightedSampler(
            choices=focus_plan.choices,
            weights=focus_plan.weights,
            rng=rng,
        )

    def _choose_focus_area(self) -> GoalFocusArea:
        """Choose a focus area using numerically-stable weighted sampling."""

        return self._sampler.sample()

    def __call__(self, t: GeneratedTuple) -> tuple[str, Mapping[str, object]]:
        _ = t
        focus = self._choose_focus_area()
        focus_prompt = self._goal_spec.render_focused_prompt(focus_area=focus)
        ctx = compose_query_context(self._base_context, goal_prompt=focus_prompt)
        return ctx, {"goal_focus_area": focus}


class RoundRobinContextBuilder:
    """Per-tuple context builder that cycles through focus areas consecutively.

    Unlike GoalSamplingContextBuilder which randomly samples a focus area,
    this builder walks through focus areas in order, wrapping around when
    all areas have been used. This guarantees even coverage across all
    focus areas while exhausting every tuple.
    """

    def __init__(
        self,
        *,
        base_context: str | None,
        goal_spec: GoalSpec,
        focus_plan: GoalFocusPlan,
    ) -> None:
        self._base_context = base_context or ""
        self._goal_spec = goal_spec
        self._choices = focus_plan.choices
        if not self._choices:
            raise ValueError("RoundRobinContextBuilder requires at least one choice")
        self._index = 0

    def __call__(self, t: GeneratedTuple) -> tuple[str, Mapping[str, object]]:
        _ = t
        focus = self._choices[self._index % len(self._choices)]
        self._index += 1
        focus_prompt = self._goal_spec.render_focused_prompt(focus_area=focus)
        ctx = compose_query_context(self._base_context, goal_prompt=focus_prompt)
        return ctx, {"goal_focus_area": focus}


def resolve_focus_plan(
    goal_spec: GoalSpec, focus_areas: list[GoalFocusArea]
) -> GoalFocusPlan:
    weights = [goal_spec.focus_weight(focus) for focus in focus_areas]
    active_pairs = [(focus, weight) for focus, weight in zip(focus_areas, weights)]
    active_pairs = [(focus, weight) for focus, weight in active_pairs if weight > 0.0]

    if active_pairs:
        choices = [focus for focus, _ in active_pairs]
        choice_weights = [weight for _, weight in active_pairs]
    else:
        choices = focus_areas
        choice_weights = [1.0 for _ in focus_areas]

    return GoalFocusPlan(choices=choices, weights=choice_weights)
