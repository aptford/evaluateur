from __future__ import annotations

from typing import TypedDict

from evaluateur.goals.constants import SYSTEM_PROMPT, USER_PROMPT_PREFIX


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
