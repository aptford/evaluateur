"""Query context composition utilities.

This module provides the mechanism for composing query generation context.
"""

from __future__ import annotations

from evaluateur.constants import (
    CONTEXT_SEPARATOR,
    TAG_INSTRUCTIONS_CLOSE,
    TAG_INSTRUCTIONS_OPEN,
)


def compose_query_context(
    context: str,
    *,
    instructions: str | None = None,
    goal_prompt: str | None = None,
) -> str:
    """Compose evaluator context with optional query instructions and goals.

    This is the single source of truth for how query-generation instructions and
    goal prompts are appended to the domain context across query generators.
    """
    base = (context or "").strip()
    chunks: list[str] = []
    if base:
        chunks.append(base)

    cleaned_instructions = (instructions or "").strip()
    if cleaned_instructions:
        chunks.append(
            f"{TAG_INSTRUCTIONS_OPEN}\n{cleaned_instructions}\n{TAG_INSTRUCTIONS_CLOSE}"
        )

    gp = (goal_prompt or "").strip()
    if gp:
        chunks.append(gp)

    return CONTEXT_SEPARATOR.join(chunks).strip()
