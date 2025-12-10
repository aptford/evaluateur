from __future__ import annotations

import itertools
import logging
from collections.abc import AsyncIterator
from enum import Enum
from typing import Protocol, TypeVar

from pydantic import BaseModel, create_model

from evaluator.client import LLMClient
from evaluator.models import GeneratedTuple
from evaluator.types import ScalarValue

log = logging.getLogger(__name__)


ModelT = TypeVar("ModelT", bound=BaseModel)


class TupleStrategy(str, Enum):
    """Strategies for generating dimension tuples."""

    CROSS_PRODUCT = "cross_product"
    DIRECT_LLM = "direct_llm"


class TupleGenerator(Protocol[ModelT]):
    """Protocol for async tuple generators that yield tuples one at a time."""

    def generate(self, options: BaseModel, count: int) -> AsyncIterator[GeneratedTuple[ModelT]]:
        """Generate tuples asynchronously, yielding one at a time."""
        ...


class CrossProductTupleGenerator:
    """Generate tuples via cross product as an async iterator.

    This mirrors the "cross product then filter" approach from Hamel Husain's
    FAQ: it guarantees coverage of the dimension space at the cost of volume.
    Yields tuples one at a time for streaming consumption.
    """

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client

    async def generate(self, options: BaseModel, count: int) -> AsyncIterator[GeneratedTuple[ModelT]]:
        """Yield tuples one at a time from the cross product."""
        field_names = list(type(options).model_fields.keys())
        value_lists: list[list[ScalarValue]] = []

        for name in field_names:
            value = getattr(options, name)
            if isinstance(value, str):
                value_lists.append([value])
            else:
                value_lists.append(list(value))

        all_combinations = list(itertools.product(*value_lists))
        log.debug(
            "CrossProductTupleGenerator: %d total combinations from %d fields",
            len(all_combinations),
            len(field_names),
        )

        selected = all_combinations[:count] if count > 0 else all_combinations
        log.debug("CrossProductTupleGenerator: selected %d tuples", len(selected))

        for combo in selected:
            yield GeneratedTuple(
                values={name: value for name, value in zip(field_names, combo)}
            )


class DirectLLMTupleGenerator:
    """Generate tuples directly with Instructor as an async iterator.

    This uses the LLM to propose realistic combinations instead of enumerating
    all possibilities. It tends to produce more natural data but may miss some
    edge cases. Yields tuples one at a time after the LLM response is received.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def generate(self, options: BaseModel, count: int) -> AsyncIterator[GeneratedTuple[ModelT]]:
        """Yield tuples one at a time after fetching from LLM."""
        client = self._client.instructor_client
        log.info("DirectLLMTupleGenerator: requesting ~%d tuples from LLM", count)

        # Build a compact description of the available options for each field.
        field_names = list[str](type(options).model_fields.keys())
        option_lines: list[str] = []
        for name in field_names:
            value = getattr(options, name)
            values_list = list(value) if not isinstance(value, str) else [value]
            display = ", ".join(map(str, values_list))
            option_lines.append(f"- {name}: {display}")

        system_message = (
            "You are generating structured synthetic test cases for an evaluation suite. "
            "Each test case is a tuple that selects exactly one value for every dimension."
        )
        user_message = (
            "Using the following options per dimension, generate diverse tuples. "
            f"Return around {count} combinations, preferring realistic and high-value cases.\n\n"
            + "\n".join(option_lines)
        )
        log.debug("DirectLLMTupleGenerator prompt:\n%s", user_message)

        # Dynamically create a model with the actual field names so the LLM
        # knows exactly what fields to generate.
        FlatTuple = create_model(
            "FlatTuple",
            **{name: (ScalarValue, ...) for name in field_names},
        )

        class TupleListModel(BaseModel):
            tuples: list[FlatTuple]  # type: ignore[valid-type]

        result: TupleListModel = await client.chat.completions.create(
            model=self._client.model_name,
            response_model=TupleListModel,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )
        log.debug("DirectLLMTupleGenerator: received %d tuples", len(result.tuples))

        for t in result.tuples:
            yield GeneratedTuple(values=t.model_dump())
