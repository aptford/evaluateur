"""Goal specification parsing.

Parses free-form text into structured GoalSpec objects using LLM extraction.
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
    """Build instructor messages for parsing goal guidance text."""
    cleaned = text.strip()
    if not cleaned:
        return []
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{USER_PROMPT_PREFIX}```{cleaned}```"},
    ]


async def parse_goal_spec(client: "LLMClient", text: str) -> "GoalSpec":
    """Parse user text into a structured GoalSpec using an LLM.

    Args:
        client: The LLM client for parsing.
        text: The text to parse.

    Returns:
        A GoalSpec instance. Returns an empty GoalSpec if text is empty.
    """
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
