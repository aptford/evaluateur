"""Prompt templates for tuple generation.

These templates define the policy for how the LLM should generate tuples
from dimension options. The mechanism (AITupleGenerator) uses these
to build actual prompts.
"""

from __future__ import annotations

import random

# Template for the system message in tuple generation
TUPLES_SYSTEM_TEMPLATE = (
    "You are generating structured synthetic test cases for an evaluation suite. "
    "Each test case is a tuple that selects exactly one value for every dimension."
)

# Template for the user message in tuple generation
TUPLES_USER_TEMPLATE = (
    "Using the following options per dimension, generate diverse tuples. "
    "Return around {count} combinations, preferring realistic and high-value cases.\n\n"
    "{option_lines}\n\n"
    "This is variation {seed}. {variation_hint} Produce a different, distinct set of "
    "tuples than other variations would."
)


def format_option_lines(
    field_names: list[str], value_lists: list[list[object]]
) -> str:
    """Format dimension options as lines for the prompt.

    Args:
        field_names: List of dimension field names.
        value_lists: List of value lists, one per dimension.

    Returns:
        A formatted string with one dimension per line.
    """
    lines: list[str] = []
    for name, values in zip(field_names, value_lists):
        display = ", ".join(map(str, values))
        lines.append(f"- {name}: {display}")
    return "\n".join(lines)


def format_tuples_prompts(
    field_names: list[str],
    value_lists: list[list[object]],
    count: int,
    instructions: str | None = None,
    seed: int = 0,
) -> tuple[str, str]:
    """Format prompts for tuple generation.

    Args:
        field_names: List of dimension field names.
        value_lists: List of value lists, one per dimension.
        count: Number of tuples to generate.
        instructions: Optional additional instructions for the LLM.
        seed: Variation number included in the prompt to encourage diverse outputs.

    Returns:
        A tuple of (system_message, user_message).
    """
    system_message = TUPLES_SYSTEM_TEMPLATE
    if instructions:
        system_message += (
            "\nAdditional instructions:\n"
            f"<instructions>\n{instructions}\n</instructions>\n"
        )

    # Use seed for deterministic variation in prompt hints
    rng = random.Random(seed)
    variation_phrases = [
        "Focus on diverse combinations across the value space.",
        "Prioritize realistic, commonly occurring scenarios.",
        "Include edge cases and boundary conditions.",
        "Balance typical cases with unusual combinations.",
    ]
    variation_hint = rng.choice(variation_phrases)

    option_lines = format_option_lines(field_names, value_lists)
    user_message = TUPLES_USER_TEMPLATE.format(
        count=count,
        option_lines=option_lines,
        seed=seed,
        variation_hint=variation_hint,
    )

    return system_message, user_message
