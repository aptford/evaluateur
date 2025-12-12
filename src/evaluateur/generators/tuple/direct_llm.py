from __future__ import annotations

import logging

from pydantic import BaseModel, create_model

from evaluateur.client import LLMClient
from evaluateur.generators.tuple.options_adapter import extract_dimension_values
from evaluateur.models import GeneratedTuple
from evaluateur.types import ScalarValue

log = logging.getLogger(__name__)


def _build_flat_tuple_model(field_names: list[str]) -> type[BaseModel]:
    """Build a Pydantic model matching the dimension names for Instructor."""

    return create_model(
        "FlatTuple",
        **{name: (ScalarValue, ...) for name in field_names},
    )


class DirectLLMTupleGenerator:
    """Generate tuples directly with Instructor as an async iterator.

    This uses the LLM to propose realistic combinations instead of enumerating
    all possibilities. It tends to produce more natural data but may miss some
    edge cases.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def generate(self, options: BaseModel, count: int, *, seed: int = 0):
        # Note: `seed` is accepted for API consistency with other tuple generators,
        # but is not currently used because the LLM backend is non-deterministic.
        _ = seed
        client = self._client.instructor_client
        log.info("DirectLLMTupleGenerator: requesting ~%d tuples from LLM", count)

        field_names, value_lists = extract_dimension_values(options)

        option_lines: list[str] = []
        for name, values in zip(field_names, value_lists):
            display = ", ".join(map(str, values))
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

        FlatTuple = _build_flat_tuple_model(field_names)

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

