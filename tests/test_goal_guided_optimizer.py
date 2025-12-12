from __future__ import annotations

import pytest

from evaluateur.goals import GoalItem, GoalLayer, GoalSpec
from evaluateur.optimizers.goal_guided import GoalGuidedQueryOptimizer, HeuristicGoalJudge, JudgeBackend


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


def test_heuristic_goal_judge_rewards_must_include_and_penalizes_avoid() -> None:
    spec = GoalSpec(
        components=GoalLayer(
            items=[
                GoalItem(
                    name="freshness",
                    must_include=["effective date", "latest policy"],
                    avoid=["undated"],
                )
            ]
        )
    )

    judge = HeuristicGoalJudge(spec)

    good = judge.score(
        query="What is the effective date of the latest policy for this payer?",
        tuple_json=None,
        context=None,
    )
    bad = judge.score(
        query="Summarize the undated policy.",
        tuple_json=None,
        context=None,
    )

    assert good > bad
    assert 0.0 <= good <= 1.0
    assert 0.0 <= bad <= 1.0


def test_goal_guided_optimizer_compile_uses_teleprompter_factory_without_dspy() -> None:
    """Ensure compile wiring works without invoking real DSPy internals.

    We pass a teleprompter factory stub and use the HEURISTIC judge backend.
    """

    gg = pytest.importorskip("evaluateur.optimizers.goal_guided")
    if getattr(gg, "dspy", None) is None:
        pytest.skip("DSPy not installed")

    spec = GoalSpec(
        components=GoalLayer(items=[GoalItem(name="include payer token", must_include=["payer"])])
    )

    class _Ex:
        context = "Healthcare prior authorization"
        tuple_json = '{"values":{"payer":"Cigna","age":"pediatric"}}'

    class _Pred:
        query = "For Cigna pediatric PA letters, what is the effective date of the latest payer policy?"

    class _DummyTeleprompter:
        def __init__(self, *, metric, auto: str) -> None:  # type: ignore[no-untyped-def]
            self.metric = metric
            self.auto = auto

        def compile(self, *, student, trainset, valset, **kwargs):  # type: ignore[no-untyped-def]
            score = float(self.metric(_Ex(), _Pred()))
            assert 0.0 <= score <= 1.0
            assert self.auto == "light"
            assert isinstance(trainset, list)
            assert isinstance(valset, list)
            return student

    opt = GoalGuidedQueryOptimizer(
        goal_spec=spec,
        judge_backend=JudgeBackend.HEURISTIC,
        auto="light",
        teleprompter_factory=lambda **kw: _DummyTeleprompter(**kw),
    )

    compiled = opt.compile(student=object(), trainset=[], valset=[])
    assert compiled is not None
