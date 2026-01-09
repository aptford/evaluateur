from __future__ import annotations

import math

import pytest

from evaluateur.goals import GoalItem, GoalLayer, GoalSpec


def test_goal_spec_render_prompt_is_compact_and_stable() -> None:
    spec = GoalSpec(
        components=GoalLayer(
            summary="Focus on freshness and grounding.",
            items=[
                GoalItem(
                    name="freshness",
                    must_include=["effective date", "as of"],
                    avoid=["undated"],
                    examples=[
                        "As of today, what is the latest effective date for this policy?",
                        "Please answer using the most recent version and cite the relevant section.",
                    ],
                )
            ],
        ),
    )

    prompt = spec.render_prompt(max_chars=2000)
    assert "Query optimization goals" in prompt
    assert "Components:" in prompt
    assert "freshness" in prompt
    assert "must include" in prompt
    assert "examples:" in prompt
    assert "latest effective date" in prompt
    assert len(prompt) <= 2000

    focus_prompt = spec.render_focused_prompt(focus_area="components", max_chars=2000)
    assert "focus area" in focus_prompt
    assert "Components:" in focus_prompt
    assert "freshness" in focus_prompt
    assert "examples:" in focus_prompt
    assert "Trajectories:" not in focus_prompt
    assert "Outcomes:" not in focus_prompt


def test_goal_spec_is_empty_when_no_goals() -> None:
    assert GoalSpec().is_empty() is True


def test_goal_spec_render_prompt_truncation_respects_max_chars() -> None:
    # Make a prompt that will definitely exceed the max.
    spec = GoalSpec(
        components=GoalLayer(
            summary=("x" * 500),
            items=[GoalItem(name="n", must_include=["a" * 200])],
        ),
    )

    max_chars = 80
    prompt = spec.render_prompt(max_chars=max_chars)

    assert len(prompt) <= max_chars
    assert prompt.endswith("…\n")


def test_goal_item_weight_must_be_finite_and_non_negative() -> None:
    with pytest.raises(ValueError):
        GoalItem(name="bad", weight=-0.1)
    with pytest.raises(ValueError):
        GoalItem(name="bad", weight=math.inf)
    with pytest.raises(ValueError):
        GoalItem(name="bad", weight=-math.inf)
    with pytest.raises(ValueError):
        GoalItem(name="bad", weight=math.nan)

    # 0.0 is allowed and used to disable goals without deleting them.
    it = GoalItem(name="disabled", weight=0.0)
    assert it.weight == 0.0
