"""Goal guidance planning.

This module provides the mechanism for building goal guidance plans.
Policy decisions are delegated to the policy module.
"""

from __future__ import annotations

import logging
import random

from evaluateur.client import LLMClient
from evaluateur.goals.models import GoalGuidancePlan, GoalMode, GoalSpec
from evaluateur.goals.parsing import parse_goal_spec
from evaluateur.goals.policy import (
    select_plan_type,
    should_include_goal_spec_in_full_plan,
    should_warn_no_goals,
)
from evaluateur.goals.sampling import (
    GoalSamplingContextBuilder,
    RoundRobinContextBuilder,
    resolve_focus_plan,
)
from evaluateur.queries.context import compose_query_context
from evaluateur.queries.models import QueryMetadata

log = logging.getLogger(__name__)


async def normalize_goal_spec(
    client: LLMClient, goals: GoalSpec | str | None
) -> GoalSpec | None:
    """Return a non-empty GoalSpec for valid inputs, else None.

    This function normalizes different goal input formats into a consistent
    GoalSpec instance, using parse_goal_spec for string inputs.
    """
    if goals is None:
        return None
    if isinstance(goals, str):
        spec = await parse_goal_spec(client, goals)
    elif isinstance(goals, GoalSpec):
        spec = goals
    else:  # pragma: no cover - defensive
        raise TypeError("goals must be a GoalSpec, a string, or None")

    if spec.is_empty():
        return None
    return spec


def _build_run_metadata(
    *,
    goal_spec: GoalSpec | None,
    goal_mode: GoalMode,
) -> QueryMetadata:
    goal_guided = bool(goal_spec and goal_spec.available_goals())
    return QueryMetadata(
        goal_guided=goal_guided,
        goal_mode=goal_mode,
        query_goals=(
            goal_spec if goal_spec is not None and not goal_spec.is_empty() else None
        ),
    )


def _build_sample_plan(
    *,
    goal_spec: GoalSpec,
    run_metadata: QueryMetadata,
    instructions: str | None,
    seed: int,
) -> GoalGuidancePlan:
    rng = random.Random(seed)
    focus_plan = resolve_focus_plan(goal_spec)
    context_builder = GoalSamplingContextBuilder(
        base_context=instructions,
        goal_spec=goal_spec,
        focus_plan=focus_plan,
        rng=rng,
    )
    return GoalGuidancePlan(
        goal_spec=goal_spec,
        run_metadata=run_metadata,
        context=instructions or "",
        context_builder=context_builder,
    )


def _build_cycle_plan(
    *,
    goal_spec: GoalSpec,
    run_metadata: QueryMetadata,
    instructions: str | None,
) -> GoalGuidancePlan:
    focus_plan = resolve_focus_plan(goal_spec)
    context_builder = RoundRobinContextBuilder(
        base_context=instructions,
        goal_spec=goal_spec,
        focus_plan=focus_plan,
    )
    return GoalGuidancePlan(
        goal_spec=goal_spec,
        run_metadata=run_metadata,
        context=instructions or "",
        context_builder=context_builder,
    )


def _build_full_plan(
    *,
    goal_spec: GoalSpec | None,
    run_metadata: QueryMetadata,
    instructions: str | None,
) -> GoalGuidancePlan:
    goal_prompt: str | None = None
    if goal_spec is not None:
        goal_prompt = goal_spec.render_prompt()

    context = compose_query_context(instructions=instructions, goal_prompt=goal_prompt)
    return GoalGuidancePlan(
        goal_spec=goal_spec,
        run_metadata=run_metadata,
        context=context,
        context_builder=None,
    )


async def plan_goal_guidance(
    *,
    client: LLMClient,
    goals: GoalSpec | str | None,
    goal_mode: GoalMode,
    instructions: str | None,
    seed: int,
) -> GoalGuidancePlan:
    """Build guidance details for a query run.

    This function orchestrates goal planning by:
    1. Normalizing the goal specification
    2. Using policy functions to determine the planning approach
    3. Delegating to the appropriate plan builder
    """
    goal_spec = await normalize_goal_spec(client, goals)
    run_metadata = _build_run_metadata(
        goal_spec=goal_spec,
        goal_mode=goal_mode,
    )

    # Use policy to select plan type
    plan_type = select_plan_type(goal_mode, goal_spec)

    if plan_type == "sample":
        assert goal_spec is not None
        return _build_sample_plan(
            goal_spec=goal_spec,
            run_metadata=run_metadata,
            instructions=instructions,
            seed=seed,
        )

    if plan_type == "cycle":
        assert goal_spec is not None
        return _build_cycle_plan(
            goal_spec=goal_spec,
            run_metadata=run_metadata,
            instructions=instructions,
        )

    # Check if we should warn about missing goals
    if should_warn_no_goals(goal_mode, goal_spec):
        log.debug("Goal sampling requested but no active goals are available.")

    # Use policy to determine if goal_spec should be included
    effective_goal_spec = (
        goal_spec if should_include_goal_spec_in_full_plan(goal_spec) else None
    )
    return _build_full_plan(
        goal_spec=effective_goal_spec,
        run_metadata=run_metadata,
        instructions=instructions,
    )
