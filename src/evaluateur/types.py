from __future__ import annotations

import types
from typing import Any, Iterable, Mapping, Sequence, Type, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, create_model


ModelT = TypeVar("ModelT", bound=BaseModel)

ScalarValue = str | int | float | bool | None
"""Type alias for scalar values that can appear in generated tuples."""


ITERABLE_ORIGINS: tuple[type, ...] = (list, tuple, set, frozenset, Sequence, Iterable)


def _is_mapping_type(tp: Any) -> bool:
    origin = get_origin(tp) or tp
    return issubclass(origin, Mapping) if isinstance(origin, type) else False


def is_iterator_field(field_type: Any) -> bool:
    """Return True if the field type should be treated as an iterator of values.

    Fields that are already iterables (e.g. ``list[str]``, ``tuple[int, ...]``)
    are considered iterator fields and are preserved as-is when constructing
    an options model.
    """

    origin = get_origin(field_type) or field_type

    if isinstance(origin, type) and issubclass(origin, BaseModel):
        # Nested models are not considered simple iterators.
        return False

    if _is_mapping_type(field_type):
        # Dict-like fields are treated as structured, not simple iterators.
        return False

    return origin in ITERABLE_ORIGINS


def _listify_type(tp: Any) -> Any:
    """Convert a scalar type into a list-of type suitable for options models.

    This works for common base types like ``str``, ``int``, ``float``, and also
    for optional and union types by wrapping the original annotation in
    ``list[...]``. The concrete value semantics are delegated to Pydantic and
    Instructor.
    """

    origin = get_origin(tp)
    args = get_args(tp)

    if origin is None:
        # Simple type like ``str`` -> ``list[str]``
        return list[tp]  # type: ignore[index]

    # Handle Optional[T] (Union[T, None]) and T | None (types.UnionType)
    if origin is Union or isinstance(tp, types.UnionType):
        # Extract the non-None type from the union
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return list[non_none_args[0]]  # type: ignore[index]

    # Fallback: wrap the original type
    return list[tp]  # type: ignore[index]


def create_options_model(model: Type[ModelT], name_suffix: str = "Options") -> Type[BaseModel]:
    """Create a dynamic options model for a given Pydantic model.

    For each *non-iterator* field on ``model``, the corresponding field on the
    returned model becomes a list of that field's type (e.g. ``str`` -> ``list[str]``).
    Iterator fields (lists, tuples, etc.) are preserved as-is.
    """

    fields: dict[str, tuple[Any, Any]] = {}

    for field_name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        default = field_info.default

        if is_iterator_field(annotation):
            # Preserve existing iterator fields unchanged. If no default is
            # provided we make the field optional with an empty list so that
            # users can still supply options manually without involving the LLM.
            if default is None:
                fields[field_name] = (annotation | None, None)  # type: ignore[operator]
            else:
                fields[field_name] = (annotation, default)
        else:
            # Simple scalar field: turn into a list-of type. If there is a
            # scalar default (e.g. "adult" for ``age: str``), treat it as an
            # initial option by wrapping it in a single-element list.
            list_type = _listify_type(annotation)
            if default is None:
                fields[field_name] = (list_type, ...)
            else:
                fields[field_name] = (list_type, [default])

    options_model_name = f"{model.__name__}{name_suffix}"
    return create_model(options_model_name, __base__=BaseModel, **fields)

