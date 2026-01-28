"""Goal specification parsing.

This module provides the mechanism for parsing free-form text into
structured GoalSpec objects using LLM-based extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from evaluateur.goals.constants import SYSTEM_PROMPT, USER_PROMPT_PREFIX

if TYPE_CHECKING:
    from evaluateur.client import LLMClient
    from evaluateur.goals.models import GoalSpec


class _Message(TypedDict):
    role: str
    content: str


def build_goal_spec_messages(text: str) -> list[_Message]:
    """Build instructor messages for parsing free-form goal guidance."""
    cleaned = text.strip()
    if not cleaned:
        return []
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{USER_PROMPT_PREFIX}```{cleaned}```"},
    ]


async def parse_goal_spec(client: LLMClient, text: str) -> GoalSpec:
    """Parse free-form user text into a structured GoalSpec.

    This is a factory function that uses Instructor + Pydantic parsing
    to extract a stable schema from free-form text.

    Args:
        client: The LLM client to use for parsing.
        text: The free-form text to parse.

    Returns:
        A GoalSpec instance. Returns an empty GoalSpec if text is empty
        or cannot be parsed.
    """
    # Import here to avoid circular imports
    from evaluateur.goals.models import GoalSpec

    cleaned = text.strip()
    if not cleaned:
        return GoalSpec()

    messages = build_goal_spec_messages(cleaned)
    if not messages:
        return GoalSpec()

    inst = client.instructor_client
    parsed: GoalSpec = await inst.chat.completions.create(
        model=client.model_name,
        response_model=GoalSpec,
        messages=messages,
    )
    return parsed
