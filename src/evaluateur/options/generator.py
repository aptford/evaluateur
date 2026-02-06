from __future__ import annotations

from typing import Any, Callable, Type

from pydantic import BaseModel, Field, create_model

from evaluateur.client import LLMClient
from evaluateur.options.types import ModelT, create_options_model, is_iterator_field
from evaluateur.prompts.options import format_options_prompts


class OptionsGenerator:
    """Generate discrete options for each field of a query model asynchronously."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def _build_response_model(
        self,
        original_model: Type[BaseModel],
        options_model: Type[BaseModel],
        count_per_field: int,
    ) -> Type[BaseModel]:
        """Create a Pydantic model for Instructor to populate options."""
        fields: dict[str, tuple[Any, Any]] = {}

        for name, field in options_model.model_fields.items():
            original_field = original_model.model_fields.get(name)
            original_annotation = (
                original_field.annotation
                if original_field is not None
                else field.annotation
            )

            # If the original model already exposes this field as an iterator
            # (e.g. ``list[str]``), we treat it as a user-provided list of
            # options and do not ask the LLM to regenerate it.
            if is_iterator_field(original_annotation):
                continue

            # We use the same type as on the options model, but we add
            # descriptions that help the LLM generate targeted values.
            description = field.description or f"Options for dimension '{name}'"
            fields[name] = (
                field.annotation,
                Field(default_factory=list, description=description, min_length=count_per_field),
            )

        return create_model(
            f"{options_model.__name__}Response", __base__=BaseModel, **fields
        )

    async def generate_options(
        self,
        model: Type[ModelT],
        *,
        instructions: str | None = None,
        count_per_field: int = 5,
        prompt_formatter: Callable[
            [type[BaseModel], int, str | None], tuple[str, str]
        ] = format_options_prompts,
    ) -> BaseModel:
        """Generate an options model instance for the given query model.

        Simple scalar fields (e.g. ``str``) are converted to lists of values.
        Fields that are already iterables (lists, tuples, etc.) are preserved
        with their existing types.

        Parameters
        ----------
        model
            The Pydantic model defining the query dimensions.
        instructions
            Optional additional instructions for the LLM.
        count_per_field
            Number of options to generate per field.
        prompt_formatter
            Callable that formats the prompts. Defaults to the standard formatter.
            This allows customizing prompts without changing mechanism code.
        """
        options_model = create_options_model(model)
        response_model = self._build_response_model(model, options_model, count_per_field)

        system_message, user_message = prompt_formatter(
            model, count_per_field, instructions
        )

        client = self._client.instructor_client

        result: BaseModel = await client.chat.completions.create(
            model=self._client.model_name,
            response_model=response_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )

        # Reconstruct an instance of the dynamically-created options model using
        # the values returned by Instructor. Any iterator fields present on the
        # original model (lists, tuples, etc.) are left unchanged, so user-
        # supplied lists of options are honoured and not overwritten.
        return options_model(**result.model_dump())
