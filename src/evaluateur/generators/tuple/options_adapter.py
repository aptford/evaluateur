from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast

from pydantic import BaseModel

from evaluateur.types import ScalarValue


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

