from __future__ import annotations

import pytest
from pydantic import BaseModel

from evaluateur.tuples import CrossProductTupleGenerator


class SmallOptions(BaseModel):
    a: list[int] = [0, 1, 2]
    b: list[str] = ["x", "y"]


@pytest.mark.asyncio
async def test_seeded_sampling_is_deterministic() -> None:
    gen = CrossProductTupleGenerator(client=None)
    opts = SmallOptions()

    out1 = [t.values async for t in gen.generate(opts, count=4, seed=123)]
    out2 = [t.values async for t in gen.generate(opts, count=4, seed=123)]

    assert out1 == out2
    assert len(out1) == 4


@pytest.mark.asyncio
async def test_seeded_sampling_differs_for_different_seeds() -> None:
    gen = CrossProductTupleGenerator(client=None)
    opts = SmallOptions()

    out1 = [t.values async for t in gen.generate(opts, count=4, seed=1)]
    out2 = [t.values async for t in gen.generate(opts, count=4, seed=2)]

    assert out1 != out2


@pytest.mark.asyncio
async def test_seeded_sampling_has_no_duplicates() -> None:
    gen = CrossProductTupleGenerator(client=None)
    opts = SmallOptions()

    out = [t.values async for t in gen.generate(opts, count=5, seed=999)]
    pairs = [(d["a"], d["b"]) for d in out]

    assert len(pairs) == len(set(pairs))


@pytest.mark.asyncio
async def test_count_ge_total_yields_full_product() -> None:
    gen = CrossProductTupleGenerator(client=None)
    opts = SmallOptions()

    # total = 3 * 2 = 6
    out = [t.values async for t in gen.generate(opts, count=999, seed=0)]
    assert len(out) == 6

    pairs = {(d["a"], d["b"]) for d in out}
    expected = {(a, b) for a in opts.a for b in opts.b}
    assert pairs == expected


class LargeOptions(BaseModel):
    """Options model with 5 dimensions for diversity testing."""

    payer: list[str] = ["Cigna", "Aetna", "UnitedHealthcare", "Blue Cross", "Humana"]
    age: list[int] = [18, 35, 50, 65, 80]
    indication: list[str] = [
        "diabetes",
        "hypertension",
        "asthma",
        "arthritis",
        "depression",
    ]
    complexity: list[str] = ["low", "medium", "high", "critical", "terminal"]
    state: list[str] = ["NY", "CA", "TX", "FL", "WA"]


def _count_differences(a: dict[str, object], b: dict[str, object]) -> int:
    """Count the number of dimensions where values differ."""
    return sum(1 for k in a if a[k] != b[k])


@pytest.mark.asyncio
async def test_diverse_sampling_maximizes_spread() -> None:
    """Verify that samples differ on multiple dimensions, not just one."""
    gen = CrossProductTupleGenerator(client=None)
    opts = LargeOptions()

    # Sample 10 tuples from 5^5 = 3125 combinations
    out = [t.values async for t in gen.generate(opts, count=10, seed=0)]

    # Check that consecutive pairs differ on at least 2 dimensions
    # With 5 dimensions, FPS should spread samples widely
    for i in range(len(out) - 1):
        diff_count = _count_differences(out[i], out[i + 1])
        assert diff_count >= 2, (
            f"Samples {i} and {i+1} only differ on {diff_count} dimension(s): "
            f"{out[i]} vs {out[i+1]}"
        )


@pytest.mark.asyncio
async def test_diverse_sampling_average_pairwise_distance() -> None:
    """Verify that the average pairwise distance is high."""
    gen = CrossProductTupleGenerator(client=None)
    opts = LargeOptions()

    out = [t.values async for t in gen.generate(opts, count=10, seed=42)]

    # Compute average pairwise Hamming distance
    total_dist = 0
    pair_count = 0
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            total_dist += _count_differences(out[i], out[j])
            pair_count += 1

    avg_distance = total_dist / pair_count if pair_count > 0 else 0

    # With 5 dimensions and FPS, we expect average distance to be high
    # Random sampling would give ~2.5 on average (50% chance of difference per dim)
    # FPS should achieve significantly better (closer to 3-4)
    assert (
        avg_distance >= 2.5
    ), f"Average pairwise distance {avg_distance:.2f} is too low"


@pytest.mark.asyncio
async def test_diverse_sampling_minimum_pairwise_distance() -> None:
    """Verify that the minimum pairwise distance is at least 2."""
    gen = CrossProductTupleGenerator(client=None)
    opts = LargeOptions()

    out = [t.values async for t in gen.generate(opts, count=8, seed=0)]

    # Find minimum pairwise distance
    min_dist = float("inf")
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            dist = _count_differences(out[i], out[j])
            min_dist = min(min_dist, dist)

    # FPS guarantees maximizing minimum distance
    # With 8 samples from 3125 combinations across 5 dimensions,
    # we should be able to maintain at least 2 differences
    assert min_dist >= 2, f"Minimum pairwise distance {min_dist} is too low"
