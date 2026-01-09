from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel, Field, create_model

from evaluateur.client import LLMClient
from evaluateur.types import ModelT, create_options_model, is_iterator_field


class OptionsGenerator:
    """Generate discrete options for each field of a query model asynchronously."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def _build_response_model(
        self, original_model: Type[BaseModel], options_model: Type[BaseModel]
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
                Field(default_factory=list, description=description),
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
    ) -> BaseModel:
        """Generate an options model instance for the given query model.

        Simple scalar fields (e.g. ``str``) are converted to lists of values.
        Fields that are already iterables (lists, tuples, etc.) are preserved
        with their existing types.
        """
        options_model = create_options_model(model)
        response_model = self._build_response_model(model, options_model)

        system_instructions = (
            "You are generating diverse, realistic options for each dimension of a synthetic "
            "evaluation schema. For each field, produce a list of concise labels that cover both "
            "common and edge-case values."
        )
        if instructions:
            system_instructions += " Additional instructions:\n"
            system_instructions += f"<instructions>\n{instructions}\n</instructions>\n"

        # Build a text description of the dimensions for the user message.
        dimension_descriptions: list[str] = []
        for name, field in model.model_fields.items():
            field_desc = field.description or ""
            field_type = str(field.annotation)
            dimension_descriptions.append(f"- {name} ({field_type}): {field_desc}")

        user_message = (
            "Given the following dimensions, generate options for each field. "
            f"Provide around {count_per_field} distinct, high-quality options per field.\n\n"
            + "\n".join(dimension_descriptions)
        )

        client = self._client.instructor_client

        result: BaseModel = await client.chat.completions.create(
            model=self._client.model_name,
            response_model=response_model,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_message},
            ],
        )

        # Reconstruct an instance of the dynamically-created options model using
        # the values returned by Instructor. Any iterator fields present on the
        # original model (lists, tuples, etc.) are left unchanged, so user-
        # supplied lists of options are honoured and not overwritten.
        return options_model(**result.model_dump())
