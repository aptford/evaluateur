from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, Callable

from pydantic import BaseModel, create_model

from evaluateur.client import LLMClient
from evaluateur.options.types import ScalarValue
from evaluateur.prompts.tuples import format_tuples_prompts
from evaluateur.queries.models import GeneratedTuple

from ..options_adapter import extract_dimension_values

log = logging.getLogger(__name__)


def _build_flat_tuple_model(field_names: list[str]) -> type[BaseModel]:
    """Build a Pydantic model matching the dimension names for Instructor."""

    return create_model(
        "FlatTuple",
        **{name: (ScalarValue, ...) for name in field_names},
    )


class AITupleGenerator:
    """Generate tuples directly with Instructor as an async iterator.

    This uses the LLM to propose realistic combinations instead of enumerating
    all possibilities. It tends to produce more natural data but may miss some
    edge cases.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def generate(
        self,
        options: BaseModel,
        count: int,
        *,
        seed: int = 0,
        temperature: float = 0.5,
        instructions: str | None = None,
        prompt_formatter: Callable[
            [list[str], list[list[object]], int, str | None, int], tuple[str, str]
        ] = format_tuples_prompts,
    ) -> AsyncIterator[GeneratedTuple]:
        """Generate tuples using the LLM.

        Parameters
        ----------
        options
            The options model containing dimension values.
        count
            Number of tuples to generate.
        seed
            Random seed for variation control. Different seeds produce different outputs
            via prompt variation (all providers). Additionally forwarded as an API
            parameter for providers that support it (OpenAI). With OpenAI, same seed
            provides ~80-90% reproducibility.
        temperature
            LLM sampling temperature (0.0-2.0). Lower values produce more consistent and
            conservative outputs; higher values produce more diverse and creative outputs.
            Default: 0.5 (balanced). This parameter works across all LLM providers.
        instructions
            Optional additional instructions for the LLM.
        prompt_formatter
            Callable that formats the prompts. Defaults to the standard formatter.
            This allows customizing prompts without changing mechanism code.
        """
        client = self._client.instructor_client
        log.info("AITupleGenerator: requesting ~%d tuples from LLM", count)

        field_names, value_lists = extract_dimension_values(options)

        system_message, user_message = prompt_formatter(
            field_names,
            value_lists,
            count,
            instructions,
            seed,
        )
        log.debug("AITupleGenerator prompt:\n%s", user_message)

        FlatTuple = _build_flat_tuple_model(field_names)

        class TupleListModel(BaseModel):
            tuples: list[FlatTuple]  # type: ignore[valid-type]

        extra_kwargs: dict[str, Any] = {"temperature": temperature}
        # Only OpenAI supports the seed API parameter; other providers
        # (Anthropic, etc.) reject unknown kwargs.  Prompt-level variation
        # from seed is already handled by the prompt formatter above.
        if self._client.provider == "openai":
            extra_kwargs["seed"] = seed

        result: TupleListModel = await client.chat.completions.create(
            model=self._client.model_name,
            response_model=TupleListModel,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            **extra_kwargs,
        )
        log.debug("AITupleGenerator: received %d tuples", len(result.tuples))

        for t in result.tuples:
            yield GeneratedTuple(values=t.model_dump())
