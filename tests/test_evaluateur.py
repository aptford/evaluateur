from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, Field

from evaluateur import Evaluator, GeneratedTuple, TupleStrategy
from evaluateur.goals import GoalItem, GoalLayer, GoalSpec
from evaluateur.generators.tuples import CrossProductTupleGenerator
from evaluateur.models import GeneratedQuery
from evaluateur.types import create_options_model, is_iterator_field


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
        components=GoalLayer(items=[GoalItem(name="force payer")]),
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
        assert q.metadata.mode == "instructor"
        assert q.metadata.goal_guided is True
        assert q.metadata.goal_mode == "sample"
        assert isinstance(q.metadata.query_goals, GoalSpec)
        assert q.metadata.query_goals.components.items[0].name == "force payer"
        assert q.metadata.goal_focus_area == "components"


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


async def test_evaluator_queries_goal_sampling_sets_focus_area_per_query() -> None:
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
        components=GoalLayer(items=[GoalItem(name="c", weight=1000.0)]),
        trajectories=GoalLayer(items=[GoalItem(name="t", weight=1.0)]),
        outcomes=GoalLayer(items=[GoalItem(name="o", weight=1.0)]),
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

    counts: dict[str, int] = {"components": 0, "trajectories": 0, "outcomes": 0}
    for q in results:
        assert q.metadata.goal_focus_area in counts
        counts[q.metadata.goal_focus_area] += 1

    assert counts["components"] > counts["trajectories"]
    assert counts["components"] > counts["outcomes"]


async def test_evaluator_queries_goal_sampling_includes_summary_only_layer() -> None:
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
        components=GoalLayer(items=[GoalItem(name="c", weight=10.0)]),
        trajectories=GoalLayer(
            summary="Prefer careful tool choice.",
            items=[GoalItem(name="t", weight=0.0)],
        ),
        outcomes=GoalLayer(items=[GoalItem(name="o", weight=0.0)]),
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

    counts: dict[str, int] = {}
    for q in results:
        counts[q.metadata.goal_focus_area] = (
            counts.get(q.metadata.goal_focus_area, 0) + 1
        )

    assert counts.get("trajectories", 0) > 0
    assert counts.get("outcomes", 0) == 0
    assert counts.get("components", 0) > counts.get("trajectories", 0)


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
