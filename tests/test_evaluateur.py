from __future__ import annotations

import os
from typing import List
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, Field

from unittest.mock import MagicMock

from evaluateur import Evaluator, GeneratedTuple, TupleStrategy
from evaluateur.client import LLMClient
from evaluateur.config import EvaluatorConfig
from evaluateur.goals import Goal, GoalSpec
from evaluateur.options import OptionsGenerator
from evaluateur.queries.models import GeneratedQuery
from evaluateur.tuples import CrossProductTupleGenerator
from evaluateur.options.types import create_options_model, is_iterator_field


class Query(BaseModel):
    payer: str = Field(..., description="insurance payer, like Cigna")
    age: str = Field(
        ..., description="patient age category, like 'adult' or 'pediatric'"
    )
    complexity: str = Field(
        ...,
        description="complexity of the query to account for the edge cases, like 'off-label', 'comorbidities', etc",
    )
    geography: str = Field(
        ...,
        description="geography indicator, like a zip code, specific state or county",
    )


class SimpleOptions(BaseModel):
    """Options model for testing tuple generation."""

    payer: list[str] = ["Cigna", "Aetna"]
    age: list[str] = ["adult", "pediatric"]


def test_is_iterator_field_simple_and_iterables() -> None:
    assert not is_iterator_field(str)
    assert is_iterator_field(list[str])
    assert is_iterator_field(tuple[int, ...])


def test_create_options_model_turns_scalars_into_lists() -> None:
    OptionsModel = create_options_model(Query)

    fields = OptionsModel.model_fields
    assert "payer" in fields
    assert "age" in fields

    for name in ("payer", "age", "complexity", "geography"):
        annotation = fields[name].annotation
        assert getattr(annotation, "__origin__", None) in (list, List)


def test_evaluator_instantiation() -> None:
    evaluator = Evaluator(Query)
    assert evaluator.model is Query


class TestCrossProductTupleGenerator:
    """Tests for CrossProductTupleGenerator async iterator."""

    @pytest.fixture
    def options(self) -> SimpleOptions:
        return SimpleOptions()

    @pytest.fixture
    def generator(self) -> CrossProductTupleGenerator:
        return CrossProductTupleGenerator(client=None)

    async def test_generates_cross_product(
        self, generator: CrossProductTupleGenerator, options: SimpleOptions
    ) -> None:
        """Test that cross product generates expected combinations."""
        tuples = [t async for t in generator.generate(options, count=0)]

        assert len(tuples) == 4

        values_set = {(t.values["payer"], t.values["age"]) for t in tuples}
        expected = {
            ("Cigna", "adult"),
            ("Cigna", "pediatric"),
            ("Aetna", "adult"),
            ("Aetna", "pediatric"),
        }
        assert values_set == expected

    async def test_respects_count_limit(
        self, generator: CrossProductTupleGenerator, options: SimpleOptions
    ) -> None:
        """Test that count parameter limits output."""
        tuples = [t async for t in generator.generate(options, count=2)]
        assert len(tuples) == 2

    async def test_yields_generated_tuple_instances(
        self, generator: CrossProductTupleGenerator, options: SimpleOptions
    ) -> None:
        """Test that yielded items are GeneratedTuple instances."""
        async for t in generator.generate(options, count=1):
            assert isinstance(t, GeneratedTuple)
            assert isinstance(t.values, dict)


class TestEvaluatorGenerateTuples:
    """Tests for Evaluator.generate_tuples async iterator."""

    @pytest.fixture
    def evaluator(self) -> Evaluator:
        return Evaluator(Query)

    async def test_generate_tuples_with_options(self, evaluator: Evaluator) -> None:
        """Test tuples() with pre-provided options."""
        options = SimpleOptions()

        tuples = [
            t
            async for t in evaluator.tuples(
                options,
                strategy=TupleStrategy.CROSS_PRODUCT,
                count=2,
            )
        ]

        assert len(tuples) == 2
        for t in tuples:
            assert isinstance(t, GeneratedTuple)
            assert "payer" in t.values
            assert "age" in t.values

    async def test_generate_tuples_is_async_iterator(
        self, evaluator: Evaluator
    ) -> None:
        """Test that tuples() returns an async iterator."""
        options = SimpleOptions()
        gen = evaluator.tuples(options, count=2)

        assert hasattr(gen, "__anext__")

        count = 0
        async for _ in gen:
            count += 1
        assert count == 2


class LargeOptions(BaseModel):
    a: list[int] = list(range(10_000))
    b: list[int] = list(range(10_000))


async def test_cross_product_does_not_materialize_full_space() -> None:
    """Guard against materializing the full cartesian product in memory."""

    gen = CrossProductTupleGenerator(client=None)
    tuples = [t async for t in gen.generate(LargeOptions(), count=1)]
    assert len(tuples) == 1
    assert 0 <= int(tuples[0].values["a"]) < 10_000
    assert 0 <= int(tuples[0].values["b"]) < 10_000


async def test_evaluator_queries_is_streaming_and_injects_run_metadata() -> None:
    evaluator = Evaluator(Query)

    class DummyQueryGenerator:
        async def generate(  # type: ignore[no-untyped-def]
            self, tuples, context, *, context_builder=None
        ):
            assert context_builder is not None
            async for t in tuples:
                _, meta = context_builder(t)
                yield GeneratedQuery(
                    query=f"Q:{t.values}",
                    source_tuple=t,
                    metadata={"generator": "dummy", **meta},
                )

    goals = GoalSpec(
        goals=[
            Goal(name="force payer", text="Test payer handling", category="components")
        ],
    )

    t1 = GeneratedTuple(values={"payer": "Cigna", "age": "adult"})
    t2 = GeneratedTuple(values={"payer": "Aetna", "age": "pediatric"})

    with patch(
        "evaluateur.evaluator.build_query_generator", return_value=DummyQueryGenerator()
    ):
        results = [
            q
            async for q in evaluator.queries(
                tuples=[t1, t2],
                goals=goals,
            )
        ]

    assert len(results) == 2
    assert results[0].source_tuple.values["payer"] == "Cigna"
    assert results[1].source_tuple.values["payer"] == "Aetna"

    for q in results:
        assert q.metadata.generator == "dummy"
        assert q.metadata.goal_guided is True
        assert q.metadata.goal_mode == "cycle"
        assert isinstance(q.metadata.query_goals, GoalSpec)
        assert q.metadata.query_goals.goals[0].name == "force payer"
        assert q.metadata.goal_focus == "force payer"
        assert q.metadata.goal_category == "components"


async def test_evaluator_queries_includes_instructions_in_context() -> None:
    evaluator = Evaluator(Query)

    captured: list[str] = []

    class DummyQueryGenerator:
        async def generate(  # type: ignore[no-untyped-def]
            self, tuples, context, *, context_builder=None
        ):
            captured.append(context)
            async for t in tuples:
                yield GeneratedQuery(query="x", source_tuple=t)

    t1 = GeneratedTuple(values={"payer": "Cigna", "age": "adult"})

    with patch(
        "evaluateur.evaluator.build_query_generator", return_value=DummyQueryGenerator()
    ):
        _ = [
            q
            async for q in evaluator.queries(
                tuples=[t1],
                instructions="Keep it short.",
            )
        ]

    assert len(captured) == 1
    assert "<instructions>" in captured[0]
    assert "Keep it short." in captured[0]
    assert "</instructions>" in captured[0]


async def test_evaluator_queries_instructions_included_with_goals() -> None:
    """When both instructions and goals are provided, instructions appear in
    the per-tuple context wrapped in <instructions> tags."""
    evaluator = Evaluator(Query)

    captured_contexts: list[str] = []

    class DummyQueryGenerator:
        async def generate(  # type: ignore[no-untyped-def]
            self, tuples, context, *, context_builder=None
        ):
            assert context_builder is not None
            async for t in tuples:
                ctx, meta = context_builder(t)
                captured_contexts.append(ctx)
                yield GeneratedQuery(query="x", source_tuple=t, metadata=meta)

    goals = GoalSpec(
        goals=[
            Goal(name="freshness", text="Test date handling", category="components")
        ],
    )

    t1 = GeneratedTuple(values={"payer": "Cigna", "age": "adult"})

    with patch(
        "evaluateur.evaluator.build_query_generator", return_value=DummyQueryGenerator()
    ):
        _ = [
            q
            async for q in evaluator.queries(
                tuples=[t1],
                instructions="Focus on US payers only.",
                goals=goals,
            )
        ]

    assert len(captured_contexts) == 1
    ctx = captured_contexts[0]
    # Instructions must be wrapped in <instructions> tags
    assert "<instructions>" in ctx
    assert "Focus on US payers only." in ctx
    assert "</instructions>" in ctx
    # Goal must also be present
    assert "<evaluation_goal>" in ctx
    assert "Test date handling" in ctx


async def test_evaluator_queries_goal_sampling_sets_focus_per_query() -> None:
    evaluator = Evaluator(Query)

    class DummyQueryGenerator:
        async def generate(  # type: ignore[no-untyped-def]
            self, tuples, context, *, context_builder=None
        ):
            assert context_builder is not None
            async for t in tuples:
                _, meta = context_builder(t)
                yield GeneratedQuery(query="x", source_tuple=t, metadata=meta)

    goals = GoalSpec(
        goals=[
            Goal(
                name="c", text="Component check", category="components", weight=1000.0
            ),
            Goal(
                name="t", text="Trajectory check", category="trajectories", weight=1.0
            ),
            Goal(name="o", text="Outcome check", category="outcomes", weight=1.0),
        ],
    )

    tuples = [
        GeneratedTuple(values={"payer": "Cigna", "age": "adult"}) for _ in range(200)
    ]

    with patch(
        "evaluateur.evaluator.build_query_generator", return_value=DummyQueryGenerator()
    ):
        results = [
            q
            async for q in evaluator.queries(
                tuples=tuples,
                goals=goals,
                goal_mode="sample",
                seed=123,
            )
        ]

    assert len(results) == len(tuples)
    assert all(q.metadata.goal_mode == "sample" for q in results)

    # Count by goal_focus (individual goal names)
    counts: dict[str, int] = {"c": 0, "t": 0, "o": 0}
    for q in results:
        assert q.metadata.goal_focus in counts
        counts[q.metadata.goal_focus] += 1

    # Higher weight goal should be sampled more often
    assert counts["c"] > counts["t"]
    assert counts["c"] > counts["o"]


async def test_evaluator_queries_cycle_mode_round_robins_goals() -> None:
    evaluator = Evaluator(Query)

    class DummyQueryGenerator:
        async def generate(  # type: ignore[no-untyped-def]
            self, tuples, context, *, context_builder=None
        ):
            assert context_builder is not None
            async for t in tuples:
                _, meta = context_builder(t)
                yield GeneratedQuery(query="x", source_tuple=t, metadata=meta)

    goals = GoalSpec(
        goals=[
            Goal(name="c", text="Component check", category="components", weight=10.0),
            Goal(
                name="t", text="Trajectory check", category="trajectories", weight=1.0
            ),
            Goal(name="o", text="Outcome check", category="outcomes", weight=1.0),
        ],
    )

    tuples = [
        GeneratedTuple(values={"payer": "Cigna", "age": "adult"}) for _ in range(9)
    ]

    with patch(
        "evaluateur.evaluator.build_query_generator", return_value=DummyQueryGenerator()
    ):
        results = [
            q
            async for q in evaluator.queries(
                tuples=tuples,
                goals=goals,
                goal_mode="cycle",
            )
        ]

    assert len(results) == 9
    assert all(q.metadata.goal_mode == "cycle" for q in results)

    # Cycle should produce: c, t, o, c, t, o, c, t, o
    expected_goals = ["c", "t", "o"] * 3
    actual_goals = [q.metadata.goal_focus for q in results]
    assert actual_goals == expected_goals


async def test_evaluator_queries_cycle_mode_interleaves_categories() -> None:
    """Cycle mode interleaves goals from different categories."""
    evaluator = Evaluator(Query)

    class DummyQueryGenerator:
        async def generate(  # type: ignore[no-untyped-def]
            self, tuples, context, *, context_builder=None
        ):
            assert context_builder is not None
            async for t in tuples:
                _, meta = context_builder(t)
                yield GeneratedQuery(query="x", source_tuple=t, metadata=meta)

    goals = GoalSpec(
        goals=[
            Goal(name="c1", text="Component 1", category="components"),
            Goal(name="c2", text="Component 2", category="components"),
            Goal(name="c3", text="Component 3", category="components"),
            Goal(name="c4", text="Component 4", category="components"),
            Goal(name="c5", text="Component 5", category="components"),
            Goal(name="t1", text="Trajectory 1", category="trajectories"),
            Goal(name="t2", text="Trajectory 2", category="trajectories"),
            Goal(name="t3", text="Trajectory 3", category="trajectories"),
            Goal(name="o1", text="Outcome 1", category="outcomes"),
            Goal(name="o2", text="Outcome 2", category="outcomes"),
        ],
    )

    tuples = [
        GeneratedTuple(values={"payer": "Cigna", "age": "adult"}) for _ in range(10)
    ]

    with patch(
        "evaluateur.evaluator.build_query_generator", return_value=DummyQueryGenerator()
    ):
        results = [
            q
            async for q in evaluator.queries(
                tuples=tuples,
                goals=goals,
                goal_mode="cycle",
            )
        ]

    assert len(results) == 10

    # CCCCCTTTOO interleaved → c1,t1,o1, c2,t2,o2, c3,t3, c4, c5
    expected_goals = [
        "c1",
        "t1",
        "o1",
        "c2",
        "t2",
        "o2",
        "c3",
        "t3",
        "c4",
        "c5",
    ]
    actual_goals = [q.metadata.goal_focus for q in results]
    assert actual_goals == expected_goals


async def test_evaluator_queries_cycle_mode_sorts_cto_categories() -> None:
    """Cycle mode uses canonical C-T-O order regardless of input order."""
    evaluator = Evaluator(Query)

    class DummyQueryGenerator:
        async def generate(  # type: ignore[no-untyped-def]
            self, tuples, context, *, context_builder=None
        ):
            assert context_builder is not None
            async for t in tuples:
                _, meta = context_builder(t)
                yield GeneratedQuery(query="x", source_tuple=t, metadata=meta)

    # Input order is C, O, T, T, T — not canonical CTO order
    goals = GoalSpec(
        goals=[
            Goal(name="c1", text="Component 1", category="components"),
            Goal(name="o1", text="Outcome 1", category="outcomes"),
            Goal(name="t1", text="Trajectory 1", category="trajectories"),
            Goal(name="t2", text="Trajectory 2", category="trajectories"),
            Goal(name="t3", text="Trajectory 3", category="trajectories"),
        ],
    )

    tuples = [
        GeneratedTuple(values={"payer": "Cigna", "age": "adult"}) for _ in range(5)
    ]

    with patch(
        "evaluateur.evaluator.build_query_generator", return_value=DummyQueryGenerator()
    ):
        results = [
            q
            async for q in evaluator.queries(
                tuples=tuples,
                goals=goals,
                goal_mode="cycle",
            )
        ]

    assert len(results) == 5

    # Despite input order COTTT, cycle should follow C-T-O order
    expected_goals = ["c1", "t1", "o1", "t2", "t3"]
    actual_goals = [q.metadata.goal_focus for q in results]
    assert actual_goals == expected_goals


async def test_evaluator_queries_cycle_mode_single_category_preserves_order() -> None:
    """When all goals share a category, cycle preserves original order."""
    evaluator = Evaluator(Query)

    class DummyQueryGenerator:
        async def generate(  # type: ignore[no-untyped-def]
            self, tuples, context, *, context_builder=None
        ):
            assert context_builder is not None
            async for t in tuples:
                _, meta = context_builder(t)
                yield GeneratedQuery(query="x", source_tuple=t, metadata=meta)

    goals = GoalSpec(
        goals=[
            Goal(name="a", text="Alpha", category="components"),
            Goal(name="b", text="Beta", category="components"),
            Goal(name="c", text="Gamma", category="components"),
        ],
    )

    tuples = [
        GeneratedTuple(values={"payer": "Cigna", "age": "adult"}) for _ in range(6)
    ]

    with patch(
        "evaluateur.evaluator.build_query_generator", return_value=DummyQueryGenerator()
    ):
        results = [
            q
            async for q in evaluator.queries(
                tuples=tuples,
                goals=goals,
                goal_mode="cycle",
            )
        ]

    assert len(results) == 6

    # Single category — original order preserved, wraps around
    expected_goals = ["a", "b", "c", "a", "b", "c"]
    actual_goals = [q.metadata.goal_focus for q in results]
    assert actual_goals == expected_goals


async def test_evaluator_queries_goal_sampling_excludes_disabled_goals() -> None:
    evaluator = Evaluator(Query)

    class DummyQueryGenerator:
        async def generate(  # type: ignore[no-untyped-def]
            self, tuples, context, *, context_builder=None
        ):
            assert context_builder is not None
            async for t in tuples:
                _, meta = context_builder(t)
                yield GeneratedQuery(query="x", source_tuple=t, metadata=meta)

    goals = GoalSpec(
        goals=[
            Goal(name="c", text="Component check", category="components", weight=10.0),
            Goal(name="t", text="Disabled", category="trajectories", weight=0.0),
            Goal(name="o", text="Disabled too", category="outcomes", weight=0.0),
        ],
    )

    tuples = [
        GeneratedTuple(values={"payer": "Cigna", "age": "adult"}) for _ in range(200)
    ]

    with patch(
        "evaluateur.evaluator.build_query_generator", return_value=DummyQueryGenerator()
    ):
        results = [
            q
            async for q in evaluator.queries(
                tuples=tuples,
                goals=goals,
                goal_mode="sample",
                seed=123,
            )
        ]

    counts: dict[str | None, int] = {}
    for q in results:
        counts[q.metadata.goal_focus] = counts.get(q.metadata.goal_focus, 0) + 1

    # Only the active goal should appear
    assert counts.get("c", 0) == 200
    assert counts.get("t", 0) == 0
    assert counts.get("o", 0) == 0


async def test_run_with_options_instructions() -> None:
    """Test that run() uses option instructions for option generation."""

    evaluator = Evaluator(Query)

    OptionsModel = create_options_model(Query)
    fake_options = OptionsModel(
        payer=["Cigna"],
        age=["adult"],
        complexity=["simple"],
        geography=["CA"],
    )

    captured: dict[str, str | None] = {"instructions": None}

    async def _fake_options(  # type: ignore[no-untyped-def]
        *, instructions: str | None = None, count_per_field: int = 5
    ):
        _ = count_per_field
        captured["instructions"] = instructions
        return fake_options

    class DummyQueryGenerator:
        async def generate(  # type: ignore[no-untyped-def]
            self, tuples, context, *, context_builder=None
        ):
            _ = context, context_builder
            async for t in tuples:
                yield GeneratedQuery(query="x", source_tuple=t)

    with patch.object(evaluator, "options", new=AsyncMock(side_effect=_fake_options)):
        with patch(
            "evaluateur.evaluator.build_query_generator",
            return_value=DummyQueryGenerator(),
        ):
            _ = [
                q
                async for q in evaluator.run(
                    options=None,
                    instructions="Focus on US payers.",
                    tuple_count=1,
                )
            ]

    assert captured["instructions"] == "Focus on US payers."


async def test_config_instructions_used_by_options() -> None:
    """Config-level instructions are used by options() when none are passed."""
    config = EvaluatorConfig(instructions="Config-level hint.")
    evaluator = Evaluator(Query, config=config)

    captured: dict[str, str | None] = {"instructions": None}

    fake_result = MagicMock()

    async def _fake_generate(self, model, *, instructions=None, **kw):  # type: ignore[no-untyped-def]
        captured["instructions"] = instructions
        return fake_result

    with patch.object(OptionsGenerator, "generate_options", _fake_generate):
        await evaluator.options()

    assert captured["instructions"] == "Config-level hint."


async def test_config_instructions_overridden_by_method_options() -> None:
    """Method-level instructions override config-level in options()."""
    config = EvaluatorConfig(instructions="Config default.")
    evaluator = Evaluator(Query, config=config)

    captured: dict[str, str | None] = {"instructions": None}

    fake_result = MagicMock()

    async def _fake_generate(self, model, *, instructions=None, **kw):  # type: ignore[no-untyped-def]
        captured["instructions"] = instructions
        return fake_result

    with patch.object(OptionsGenerator, "generate_options", _fake_generate):
        await evaluator.options(instructions="Override!")

    assert captured["instructions"] == "Override!"


async def test_config_instructions_used_by_tuples() -> None:
    """Config-level instructions are used by tuples() when none are passed."""
    config = EvaluatorConfig(instructions="Tuple config hint.")
    evaluator = Evaluator(Query, config=config)

    captured: dict[str, str | None] = {"instructions": None}

    from evaluateur.tuples.strategies.ai import AITupleGenerator as _AI

    async def _fake_gen(self, options, count, *, instructions=None, **kw):  # type: ignore[no-untyped-def]
        captured["instructions"] = instructions
        yield GeneratedTuple(values={"payer": "Cigna", "age": "adult"})

    opts = SimpleOptions()
    with patch.object(_AI, "generate", _fake_gen):
        _ = [
            t async for t in evaluator.tuples(opts, strategy=TupleStrategy.AI, count=2)
        ]

    assert captured["instructions"] == "Tuple config hint."


async def test_config_instructions_overridden_by_method_tuples() -> None:
    """Method-level instructions override config-level in tuples()."""
    config = EvaluatorConfig(instructions="Tuple config default.")
    evaluator = Evaluator(Query, config=config)

    captured: dict[str, str | None] = {"instructions": None}

    from evaluateur.tuples.strategies.ai import AITupleGenerator as _AI

    async def _fake_gen(self, options, count, *, instructions=None, **kw):  # type: ignore[no-untyped-def]
        captured["instructions"] = instructions
        yield GeneratedTuple(values={"payer": "Cigna", "age": "adult"})

    opts = SimpleOptions()
    with patch.object(_AI, "generate", _fake_gen):
        _ = [
            t
            async for t in evaluator.tuples(
                opts,
                strategy=TupleStrategy.AI,
                count=2,
                instructions="Tuple override!",
            )
        ]

    assert captured["instructions"] == "Tuple override!"


async def test_config_instructions_used_by_queries() -> None:
    """Config-level instructions flow into queries() context."""
    config = EvaluatorConfig(instructions="Query config hint.")
    evaluator = Evaluator(Query, config=config)

    captured: list[str] = []

    class DummyQueryGenerator:
        async def generate(  # type: ignore[no-untyped-def]
            self, tuples, context, *, context_builder=None
        ):
            captured.append(context)
            async for t in tuples:
                yield GeneratedQuery(query="x", source_tuple=t)

    t1 = GeneratedTuple(values={"payer": "Cigna", "age": "adult"})

    with patch(
        "evaluateur.evaluator.build_query_generator", return_value=DummyQueryGenerator()
    ):
        _ = [q async for q in evaluator.queries(tuples=[t1])]

    assert len(captured) == 1
    assert "<instructions>" in captured[0]
    assert "Query config hint." in captured[0]


async def test_config_instructions_overridden_by_method_queries() -> None:
    """Method-level instructions override config-level in queries()."""
    config = EvaluatorConfig(instructions="Query config default.")
    evaluator = Evaluator(Query, config=config)

    captured: list[str] = []

    class DummyQueryGenerator:
        async def generate(  # type: ignore[no-untyped-def]
            self, tuples, context, *, context_builder=None
        ):
            captured.append(context)
            async for t in tuples:
                yield GeneratedQuery(query="x", source_tuple=t)

    t1 = GeneratedTuple(values={"payer": "Cigna", "age": "adult"})

    with patch(
        "evaluateur.evaluator.build_query_generator", return_value=DummyQueryGenerator()
    ):
        _ = [
            q
            async for q in evaluator.queries(
                tuples=[t1], instructions="Query override!"
            )
        ]

    assert len(captured) == 1
    assert "Query override!" in captured[0]
    assert "Query config default." not in captured[0]


def _make_mock_llm_client(provider: str = "openai") -> LLMClient:
    """Create a mock LLM client that returns a minimal options response."""
    mock_instructor = MagicMock()
    mock_result = MagicMock()
    mock_result.model_dump.return_value = {
        "payer": ["Cigna"],
        "age": ["adult"],
        "complexity": ["simple"],
        "geography": ["CA"],
    }
    mock_instructor.chat.completions.create = AsyncMock(return_value=mock_result)
    return LLMClient(
        instructor_client=mock_instructor,
        model_name="test-model",
        provider=provider,
    )


async def test_options_generator_instructions_reach_system_message() -> None:
    """Instructions passed to OptionsGenerator appear in the LLM system message."""
    client = _make_mock_llm_client()
    gen = OptionsGenerator(client)

    await gen.generate_options(
        Query, instructions="Focus on US payers.", count_per_field=1
    )

    call_kwargs = client.instructor_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    system_msg = messages[0]["content"]
    assert "<instructions>" in system_msg
    assert "Focus on US payers." in system_msg
    assert "</instructions>" in system_msg


async def test_options_generator_no_instructions_tag_when_none() -> None:
    """No <instructions> tag appears in the system message when instructions are None."""
    client = _make_mock_llm_client()
    gen = OptionsGenerator(client)

    await gen.generate_options(Query, count_per_field=1)

    call_kwargs = client.instructor_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    system_msg = messages[0]["content"]
    assert "<instructions>" not in system_msg


@pytest.mark.env
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires LLM API credentials",
)
@pytest.mark.asyncio
async def test_evaluator_tuples_respects_seed_and_temperature() -> None:
    """Integration test: Evaluator.tuples() with AI strategy respects seed and temperature."""
    evaluator = Evaluator(Query)

    from evaluateur.options.types import create_options_model

    OptionsModel = create_options_model(Query)
    opts = OptionsModel(
        payer=["Cigna", "Aetna", "UnitedHealth"],
        age=["adult", "pediatric", "geriatric"],
        complexity=["simple", "moderate"],
        geography=["CA", "NY"],
    )

    # Use high temperature so the LLM actually produces varied outputs.
    # With low temperature the model gravitates to the same "best" combos
    # regardless of seed.
    tuples1 = [
        t
        async for t in evaluator.tuples(
            opts, strategy=TupleStrategy.AI, count=5, seed=100, temperature=1.5
        )
    ]
    tuples2 = [
        t
        async for t in evaluator.tuples(
            opts, strategy=TupleStrategy.AI, count=5, seed=200, temperature=1.5
        )
    ]

    values1 = {tuple(sorted(t.values.items())) for t in tuples1}
    values2 = {tuple(sorted(t.values.items())) for t in tuples2}

    # With high temperature and different seeds, the sets should not be
    # fully identical.  We check that at least one tuple differs.
    assert values1 != values2, (
        "Different seeds with high temperature should produce at least "
        "one different tuple"
    )


@pytest.mark.env
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires LLM API credentials",
)
@pytest.mark.asyncio
async def test_evaluator_tuples_temperature_parameter() -> None:
    """Test that temperature parameter is accepted by Evaluator.tuples()."""
    evaluator = Evaluator(Query)

    from evaluateur.options.types import create_options_model

    OptionsModel = create_options_model(Query)
    opts = OptionsModel(
        payer=["Cigna", "Aetna"],
        age=["adult", "pediatric"],
        complexity=["simple"],
        geography=["CA"],
    )

    # Should not raise an error
    tuples = [
        t
        async for t in evaluator.tuples(
            opts, strategy=TupleStrategy.AI, count=2, seed=1, temperature=0.8
        )
    ]

    assert len(tuples) == 2
