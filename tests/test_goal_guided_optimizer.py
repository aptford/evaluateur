from __future__ import annotations

from evaluateur.goals import GoalItem, GoalLayer, GoalSpec


def test_goal_spec_render_prompt_is_compact_and_stable() -> None:
    spec = GoalSpec(
        title="Test",
        components=GoalLayer(
            summary="Focus on freshness and grounding.",
            items=[
                GoalItem(
                    name="freshness",
                    must_include=["effective date", "as of"],
                    avoid=["undated"],
                )
            ],
        ),
    )

    prompt = spec.render_prompt(max_chars=2000)
    assert "Query optimization goals" in prompt
    assert "Components:" in prompt
    assert "freshness" in prompt
    assert "must include" in prompt
    assert len(prompt) <= 2000


def test_goal_spec_is_empty_when_no_goals() -> None:
    assert GoalSpec().is_empty() is True
