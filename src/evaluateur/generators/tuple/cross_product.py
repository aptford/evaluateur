from __future__ import annotations

import itertools
import logging
import math
import random
from collections.abc import AsyncIterator

from pydantic import BaseModel

from evaluateur.client import LLMClient
from evaluateur.generators.tuple.options_adapter import extract_dimension_values
from evaluateur.models import GeneratedTuple

log = logging.getLogger(__name__)


class CrossProductTupleGenerator:
    """Generate tuples via cross product as an async iterator.

    This mirrors the "cross product then filter" approach from Hamel Husain's
    FAQ: it guarantees coverage of the dimension space at the cost of volume.
    """

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client

    def _index_to_combo(self, index: int, value_lists: list[list[object]]) -> tuple[object, ...]:
        """Map a flat index into the cartesian product tuple (mixed-radix decomposition)."""

        if not value_lists:
            return ()

        sizes = [len(v) for v in value_lists]
        # Defensive: if any dimension is empty, there are no combinations.
        if any(s <= 0 for s in sizes):
            raise ValueError("Cannot map index for empty dimension list.")

        out: list[object] = [None] * len(value_lists)
        rem = index
        # Mixed radix: last dimension varies fastest.
        for i in range(len(value_lists) - 1, -1, -1):
            base = sizes[i]
            rem, digit = divmod(rem, base)
            out[i] = value_lists[i][digit]
        return tuple(out)

    def _sample_indices_floyd(self, *, total: int, k: int, rng: random.Random) -> list[int]:
        """Sample k unique integers from range(total) uniformly (Floyd's algorithm)."""

        if k <= 0:
            return []
        if k >= total:
            return list(range(total))

        selected: set[int] = set()
        # Iterate j over the last k values in [0, total).
        for j in range(total - k, total):
            t = rng.randrange(0, j + 1)
            if t in selected:
                selected.add(j)
            else:
                selected.add(t)
        indices = list(selected)
        rng.shuffle(indices)
        return indices

    async def generate(self, options: BaseModel, count: int, *, seed: int = 0) -> AsyncIterator[GeneratedTuple]:
        field_names, value_lists = extract_dimension_values(options)

        # No dimensions: exactly one empty combination.
        if not field_names:
            # The cartesian product over an empty set of dimensions is a single empty tuple.
            # This mirrors the behavior below where `count <= 0` means "no limit".
            yield GeneratedTuple(values={})
            return

        # If any dimension has zero values, there are no combinations.
        if any(len(v) == 0 for v in value_lists):
            return

        total = math.prod((len(v) for v in value_lists), start=1) if field_names else 1
        log.debug(
            "CrossProductTupleGenerator: ~%d total combinations from %d fields",
            total,
            len(field_names),
        )

        # If we want a strict subset, sample uniformly without replacement with a deterministic RNG.
        if 0 < count < total:
            rng = random.Random(seed)
            indices = self._sample_indices_floyd(total=total, k=count, rng=rng)
            yielded = 0
            for idx in indices:
                combo = self._index_to_combo(idx, value_lists)  # type: ignore[arg-type]
                yielded += 1
                yield GeneratedTuple(values={name: value for name, value in zip(field_names, combo)})
            log.debug("CrossProductTupleGenerator: yielded %d tuples", yielded)
            return

        combos = itertools.product(*value_lists)
        if count > 0:
            combos = itertools.islice(combos, count)

        yielded = 0
        for combo in combos:
            yielded += 1
            yield GeneratedTuple(values={name: value for name, value in zip(field_names, combo)})

        log.debug("CrossProductTupleGenerator: yielded %d tuples", yielded)

