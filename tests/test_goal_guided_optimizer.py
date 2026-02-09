from __future__ import annotations

import math

import pytest

from evaluateur.goals import Goal, GoalSpec


def test_goal_spec_render_prompt_is_compact_and_stable() -> None:
    spec = GoalSpec(
        goals=[
            Goal(
                name="freshness",
                text="Checks effective dates and prefers newest policy version.",
                category="components",
            ),
        ],
    )

    prompt = spec.render_prompt()
    assert "Query optimization goals" in prompt
    assert "Components:" in prompt
    assert "freshness" in prompt
    assert len(prompt) <= 2000


def test_goal_spec_render_prompt_with_mixed_categories() -> None:
    spec = GoalSpec(
        goals=[
            Goal(name="c1", text="Component goal", category="components"),
            Goal(name="t1", text="Trajectory goal", category="trajectories"),
            Goal(name="uncategorized", text="General goal"),
        ],
    )

    prompt = spec.render_prompt()
    assert "Components:" in prompt
    assert "Trajectories:" in prompt
    assert "General:" in prompt
    assert "uncategorized" in prompt


def test_goal_spec_render_prompt_uncategorized_only() -> None:
    spec = GoalSpec(
        goals=[
            Goal(name="g1", text="First goal"),
            Goal(name="g2", text="Second goal"),
        ],
    )

    prompt = spec.render_prompt()
    assert "Query optimization goals" in prompt
    assert "g1" in prompt
    assert "g2" in prompt
    # No category headers when all uncategorized
    assert "Components:" not in prompt
    assert "General:" not in prompt


def test_goal_spec_is_empty_when_no_goals() -> None:
    assert GoalSpec().is_empty() is True


def test_goal_spec_is_empty_when_all_disabled() -> None:
    spec = GoalSpec(goals=[Goal(text="disabled", weight=0.0)])
    assert spec.is_empty() is True


def test_goal_spec_available_goals() -> None:
    spec = GoalSpec(
        goals=[
            Goal(text="active", weight=1.0),
            Goal(text="disabled", weight=0.0),
            Goal(text="also active", weight=2.0),
        ],
    )
    active = spec.available_goals()
    assert len(active) == 2
    assert active[0].text == "active"
    assert active[1].text == "also active"


def test_goal_weight_must_be_finite_and_non_negative() -> None:
    with pytest.raises(ValueError):
        Goal(text="bad", weight=-0.1)
    with pytest.raises(ValueError):
        Goal(text="bad", weight=math.inf)
    with pytest.raises(ValueError):
        Goal(text="bad", weight=-math.inf)
    with pytest.raises(ValueError):
        Goal(text="bad", weight=math.nan)

    # 0.0 is allowed and used to disable goals without deleting them.
    it = Goal(text="disabled", weight=0.0)
    assert it.weight == 0.0
