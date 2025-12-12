from __future__ import annotations


def compose_context(context: str, goal_prompt: str | None) -> str:
    """Compose evaluator context with optional goal guidance.

    This is the single source of truth for how goal prompts are appended to the
    domain context across query generators.
    """

    base = (context or "").strip()
    if not goal_prompt:
        return base

    gp = goal_prompt.strip()
    if not base:
        return gp

    return f"{base}\n\n{gp}"

