"""Prompt templates for options generation.

These templates define the policy for how the LLM should generate dimension
options. The mechanism (OptionsGenerator) uses these to build actual prompts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

# Template for the system message in options generation
OPTIONS_SYSTEM_TEMPLATE = (
    "You are generating diverse, realistic options for each dimension of a synthetic "
    "evaluation schema. For each field, produce a list of concise labels that cover both "
    "common and edge-case values."
)

# Template for the user message in options generation
OPTIONS_USER_TEMPLATE = (
    "Given the following dimensions, generate options for each field. "
    "Provide around {count_per_field} distinct, high-quality options per field.\n\n"
    "{dimension_descriptions}"
)


def format_dimension_descriptions(model: type[BaseModel]) -> str:
    """Format model fields as dimension descriptions for the prompt.

    Args:
        model: The Pydantic model defining the dimensions.

    Returns:
        A formatted string with one dimension per line.
    """
    lines: list[str] = []
    for name, field in model.model_fields.items():
        field_desc = field.description or ""
        field_type = str(field.annotation)
        lines.append(f"- {name} ({field_type}): {field_desc}")
    return "\n".join(lines)


def format_options_prompts(
    model: type[BaseModel],
    count_per_field: int,
    instructions: str | None = None,
) -> tuple[str, str]:
    """Format prompts for options generation.

    Args:
        model: The Pydantic model defining the dimensions.
        count_per_field: Number of options to generate per field.
        instructions: Optional additional instructions for the LLM.

    Returns:
        A tuple of (system_message, user_message).
    """
    system_message = OPTIONS_SYSTEM_TEMPLATE
    if instructions:
        system_message += (
            " Additional instructions:\n"
            f"<instructions>\n{instructions}\n</instructions>\n"
        )

    dimension_descriptions = format_dimension_descriptions(model)
    user_message = OPTIONS_USER_TEMPLATE.format(
        count_per_field=count_per_field,
        dimension_descriptions=dimension_descriptions,
    )

    return system_message, user_message
