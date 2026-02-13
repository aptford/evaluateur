"""Goal sampling mechanisms.

Provides weighted random sampling and category-interleaved round-robin
cycling over individual Goals (not CTO layers).
"""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Mapping

from evaluateur.goals.constants import CTO_CATEGORIES
from evaluateur.goals.models import Goal, GoalFocusPlan, GoalSpec
from evaluateur.goals.rendering import render_focused_goal_prompt
from evaluateur.queries.context import compose_query_context
from evaluateur.queries.models import GeneratedTuple

# Lookup for canonical CTO ordering; lower rank = earlier in the cycle.
_CTO_RANK: dict[str, int] = {cat: i for i, cat in enumerate(CTO_CATEGORIES)}


def interleave_by_category(goals: list[Goal]) -> list[Goal]:
    """Reorder goals so consecutive entries cycle through categories.

    Groups goals by ``Goal.category`` (preserving insertion order within
    each group), then round-robins across groups.  CTO categories are
    visited in **C-T-O order**; any other categories follow in
    first-seen order.  When a category is exhausted its slot is skipped
    for the remaining rounds.

    Returns a new list; the original is not mutated.

    Example – categories ``COTTT`` become ``C,T,O,T,T``.
    """
    buckets: dict[str, deque[Goal]] = {}
    insertion_order: dict[str, int] = {}
    for goal in goals:
        key = goal.category or ""
        if key not in buckets:
            insertion_order[key] = len(insertion_order)
            buckets[key] = deque()
        buckets[key].append(goal)

    if len(buckets) <= 1:
        return list(goals)  # copy, no rearranging needed

    # Sort categories: CTO categories in canonical order, then custom
    # categories in first-seen order.
    sorted_keys = sorted(
        buckets,
        key=lambda k: (_CTO_RANK.get(k, len(_CTO_RANK)), insertion_order[k]),
    )

    result: list[Goal] = []
    queues = [buckets[k] for k in sorted_keys]
    while queues:
        remaining: list[deque[Goal]] = []
        for q in queues:
            result.append(q.popleft())
            if q:
                remaining.append(q)
        queues = remaining
    return result


class WeightedSampler:
    """Numerically-stable sampler for weighted goals."""

    def __init__(
        self,
        *,
        choices: list[Goal],
        weights: list[float],
        rng: random.Random,
    ) -> None:
        if len(choices) != len(weights):
            raise ValueError("WeightedSampler choices/weights length mismatch")
        self._choices = choices
        self._weights = weights
        self._rng = rng

    def sample(self) -> Goal:
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
    """Per-tuple context builder that samples an individual goal."""

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

    def __call__(self, t: GeneratedTuple) -> tuple[str, Mapping[str, object]]:
        _ = t
        goal = self._sampler.sample()
        focus_prompt = render_focused_goal_prompt(goal)
        ctx = compose_query_context(instructions=self._base_context, goal_prompt=focus_prompt)
        meta: dict[str, object] = {}
        if goal.name:
            meta["goal_focus"] = goal.name
        if goal.category:
            meta["goal_category"] = goal.category
        return ctx, meta


class RoundRobinContextBuilder:
    """Per-tuple context builder that cycles through individual goals.

    Unlike GoalSamplingContextBuilder which randomly samples a goal,
    this builder interleaves goals by category so that consecutive
    calls cycle through different categories before repeating one.
    When all goals have been visited it wraps around.
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
        self._choices = interleave_by_category(focus_plan.choices)
        if not self._choices:
            raise ValueError("RoundRobinContextBuilder requires at least one choice")
        self._index = 0

    def __call__(self, t: GeneratedTuple) -> tuple[str, Mapping[str, object]]:
        _ = t
        goal = self._choices[self._index % len(self._choices)]
        self._index += 1
        focus_prompt = render_focused_goal_prompt(goal)
        ctx = compose_query_context(instructions=self._base_context, goal_prompt=focus_prompt)
        meta: dict[str, object] = {}
        if goal.name:
            meta["goal_focus"] = goal.name
        if goal.category:
            meta["goal_category"] = goal.category
        return ctx, meta


def resolve_focus_plan(goal_spec: GoalSpec) -> GoalFocusPlan:
    """Resolve active goals and their weights into a GoalFocusPlan."""
    active = goal_spec.available_goals()
    if not active:
        return GoalFocusPlan(choices=[], weights=[])

    weights = [g.weight for g in active]
    return GoalFocusPlan(choices=active, weights=weights)
