from __future__ import annotations

import pytest
from pydantic import BaseModel

from evaluateur.generators.tuples import CrossProductTupleGenerator


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

