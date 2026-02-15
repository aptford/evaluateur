from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import cast

from pydantic import BaseModel

from evaluateur.options.types import ScalarValue


def extract_dimension_values(options: BaseModel) -> tuple[list[str], list[list[ScalarValue]]]:
    """Extract (field_names, value_lists) from an options model instance.

    This normalizes scalar values and iterables into a list-of-values per field.
    """

    field_names = list(type(options).model_fields.keys())
    value_lists: list[list[ScalarValue]] = []

    for name in field_names:
        value = getattr(options, name)

        # Strings are iterables but should be treated as a single scalar.
        if isinstance(value, str) or not isinstance(value, Iterable) or isinstance(value, Mapping):
            value_lists.append([cast(ScalarValue, value)])
            continue

        value_lists.append([cast(ScalarValue, v) for v in list(value)])

    return field_names, value_lists


def shuffle_value_lists(
    value_lists: list[list[ScalarValue]],
    rng: random.Random,
) -> list[list[ScalarValue]]:
    """Return a new list of shuffled copies of each value list.

    Each inner list is copied and independently shuffled using the
    provided RNG.  The original lists are never mutated.
    """
    shuffled: list[list[ScalarValue]] = []
    for vl in value_lists:
        copy = list(vl)
        rng.shuffle(copy)
        shuffled.append(copy)
    return shuffled


def shuffle_dimensions(
    field_names: list[str],
    value_lists: list[list[ScalarValue]],
    rng: random.Random,
) -> tuple[list[str], list[list[ScalarValue]]]:
    """Shuffle dimension presentation order.

    Returns new lists with dimensions in a random order determined by
    *rng*.  The originals are not mutated.  This changes which dimensions
    the LLM sees first, significantly affecting its output anchoring.
    """
    perm = list(range(len(field_names)))
    rng.shuffle(perm)
    return [field_names[i] for i in perm], [value_lists[i] for i in perm]
