from __future__ import annotations

import itertools
import logging
import math
import random
from collections.abc import AsyncIterator

from pydantic import BaseModel

from evaluateur.client import LLMClient
from evaluateur.queries.models import GeneratedTuple

from ..options_adapter import extract_dimension_values, shuffle_value_lists

log = logging.getLogger(__name__)


class CrossProductTupleGenerator:
    """Generate tuples via cross product as an async iterator.

    This mirrors the "cross product then filter" approach from Hamel Husain's
    FAQ: it guarantees coverage of the dimension space at the cost of volume.

    When sampling a subset, uses Farthest Point Sampling (FPS) to maximize
    diversity across all dimensions. Each selected sample is chosen to be
    maximally different from all previously selected samples.
    """

    _POOL_CAP = 10_000

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client

    def _index_to_combo(
        self, index: int, value_lists: list[list[object]]
    ) -> tuple[object, ...]:
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

    def _hamming_distance(
        self, combo_a: tuple[object, ...], combo_b: tuple[object, ...]
    ) -> int:
        """Count dimensions where values differ (Hamming distance)."""
        return sum(1 for a, b in zip(combo_a, combo_b) if a != b)

    def _sample_indices_diverse(
        self,
        *,
        total: int,
        k: int,
        value_lists: list[list[object]],
        rng: random.Random,
    ) -> list[int]:
        """Sample k indices using Farthest Point Sampling for maximum diversity.

        This algorithm greedily selects points that maximize the minimum distance
        to all previously selected points. The distance metric is Hamming distance
        (count of dimensions where values differ).

        To avoid materializing the full cartesian product (which can be enormous),
        FPS operates on a bounded candidate pool. For small spaces the pool covers
        every combination; for large spaces a random subset is drawn first.

        Algorithm:
            1. Build a candidate pool (all indices if small, random sample if large)
            2. Select a random first point from the pool (seeded for determinism)
            3. For each subsequent selection, find the pool candidate that maximizes
               its minimum distance to the already-selected set
            4. Repeat until k points are selected

        Time complexity: O(k * P * d) where P = min(total, POOL_CAP), d = dimensions
        Space complexity: O(P) for tracking minimum distances

        Args:
            total: Total number of combinations in the cross product
            k: Number of samples to select
            value_lists: List of value options per dimension
            rng: Seeded random number generator for determinism

        Returns:
            List of k indices into the cross product space
        """
        if k <= 0:
            return []
        if k >= total:
            return list(range(total))

        # Bound the candidate pool to avoid materializing huge spaces.
        pool_size = min(total, max(k * 10, self._POOL_CAP))
        if pool_size >= total:
            pool_indices = list(range(total))
        else:
            pool_indices = rng.sample(range(total), pool_size)

        pool_combos = [self._index_to_combo(idx, value_lists) for idx in pool_indices]

        # Select first point randomly (seeded for reproducibility)
        first = rng.randrange(len(pool_indices))
        selected = [first]

        # min_distance[i] = minimum Hamming distance from pool combo i to any selected combo
        # -1 means the combo has been selected (sentinel value)
        min_distance = [
            self._hamming_distance(pool_combos[i], pool_combos[first])
            for i in range(len(pool_indices))
        ]
        min_distance[first] = -1  # Mark as selected

        # Greedily select k-1 more points
        for _ in range(k - 1):
            # Find candidate with maximum min_distance (farthest from selected set)
            best_pool_idx = -1
            best_dist = -1
            for i in range(len(pool_indices)):
                if min_distance[i] > best_dist:
                    best_dist = min_distance[i]
                    best_pool_idx = i

            selected.append(best_pool_idx)
            new_combo = pool_combos[best_pool_idx]

            # Update min_distances based on newly selected point
            for i in range(len(pool_indices)):
                if min_distance[i] >= 0:  # Not yet selected
                    d = self._hamming_distance(pool_combos[i], new_combo)
                    if d < min_distance[i]:
                        min_distance[i] = d
            min_distance[best_pool_idx] = -1  # Mark as selected

        return [pool_indices[i] for i in selected]

    async def generate(
        self,
        options: BaseModel,
        count: int,
        *,
        seed: int = 0,
        temperature: float = 0.5,
        instructions: str | None = None,
    ) -> AsyncIterator[GeneratedTuple]:
        _ = instructions
        _ = temperature  # Unused: cross-product doesn't call LLMs
        field_names, value_lists = extract_dimension_values(options)

        # No dimensions: exactly one empty combination.
        if not field_names:
            # The cartesian product over an empty set of dimensions is a single empty tuple.
            # This mirrors the behavior below where `count <= 0` means "no limit".
            yield GeneratedTuple({})
            return

        # If any dimension has zero values, there are no combinations.
        if any(len(v) == 0 for v in value_lists):
            return

        # Shuffle the value lists using the seed so that different seeds
        # produce a fundamentally different combinatorial space mapping.
        # This is done on copies — the original options model is not mutated.
        rng = random.Random(seed)
        value_lists = shuffle_value_lists(value_lists, rng)

        total = math.prod((len(v) for v in value_lists), start=1) if field_names else 1
        log.debug(
            "CrossProductTupleGenerator: ~%d total combinations from %d fields",
            total,
            len(field_names),
        )

        # If we want a strict subset, use diversity-maximizing sampling.
        if 0 < count < total:
            indices = self._sample_indices_diverse(
                total=total, k=count, value_lists=value_lists, rng=rng  # type: ignore[arg-type]
            )
            yielded = 0
            for idx in indices:
                combo = self._index_to_combo(idx, value_lists)  # type: ignore[arg-type]
                yielded += 1
                yield GeneratedTuple(
                    {name: value for name, value in zip(field_names, combo)}
                )
            log.debug("CrossProductTupleGenerator: yielded %d tuples", yielded)
            return

        combos = itertools.product(*value_lists)
        if count > 0:
            combos = itertools.islice(combos, count)

        yielded = 0
        for combo in combos:
            yielded += 1
            yield GeneratedTuple(
                {name: value for name, value in zip(field_names, combo)}
            )

        log.debug("CrossProductTupleGenerator: yielded %d tuples", yielded)
