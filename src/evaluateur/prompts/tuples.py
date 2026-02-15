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

# Template for the user message in tuple generation.
# {anchor_hint} gives the LLM a seed-specific starting point so the first
# tuple differs across seeds.  {variation_hint} steers overall strategy.
TUPLES_USER_TEMPLATE = (
    "Using the following options per dimension, generate diverse tuples. "
    "Return around {count} combinations, preferring realistic and high-value cases.\n\n"
    "{option_lines}\n\n"
    "Your first tuple MUST use these values: {anchor_hint}\n"
    "Then generate {remaining} more diverse tuples.\n"
    "{variation_hint}"
)

# Structurally different variation strategies.  Each phrase steers the LLM
# toward a meaningfully different region of the output space.
VARIATION_PHRASES = [
    "Spread the remaining tuples across as many distinct values as possible.",
    "Prioritize realistic, commonly occurring scenarios for the rest.",
    "Include edge cases and boundary conditions in the remaining tuples.",
    "Balance typical cases with unusual or rare combinations.",
    "Favor combinations where dimension values are least correlated with each other.",
    "Emphasize the last listed value in each dimension for at least one tuple.",
    "Ensure no single value appears in more than half of the tuples.",
    "Maximize coverage: each value from every dimension should appear at least once if possible.",
    "Prefer tuples that a domain expert would find surprising but plausible.",
    "Pair high-frequency values in one dimension with low-frequency values in others.",
]


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


def _select_anchor_values(
    field_names: list[str],
    value_lists: list[list[object]],
    rng: random.Random,
) -> str:
    """Pick one value per dimension as a seed-specific anchor.

    Returns a formatted string like ``payer=Cigna, age=72, ...``.
    """
    parts: list[str] = []
    for name, values in zip(field_names, value_lists):
        chosen = rng.choice(values)
        parts.append(f"{name}={chosen}")
    return ", ".join(parts)


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
        seed: Variation number used to deterministically select anchor values
            and a variation strategy, ensuring different seeds produce
            structurally different prompts.

    Returns:
        A tuple of (system_message, user_message).
    """
    system_message = TUPLES_SYSTEM_TEMPLATE
    if instructions:
        system_message += (
            "\nAdditional instructions:\n"
            f"<instructions>\n{instructions}\n</instructions>\n"
        )

    # Use seed for deterministic variation in anchor selection and hint
    rng = random.Random(seed)
    anchor_hint = _select_anchor_values(field_names, value_lists, rng)
    variation_hint = rng.choice(VARIATION_PHRASES)

    option_lines = format_option_lines(field_names, value_lists)
    user_message = TUPLES_USER_TEMPLATE.format(
        count=count,
        remaining=max(count - 1, 1),
        option_lines=option_lines,
        anchor_hint=anchor_hint,
        variation_hint=variation_hint,
    )

    return system_message, user_message
