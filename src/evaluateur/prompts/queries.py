"""Prompt templates for query generation.

These templates define the policy for how the LLM should generate natural
language queries from tuples. The mechanism (query generators) use these
to build actual prompts.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluateur.queries.models import GeneratedTuple

# Template for the system message in query generation
QUERY_SYSTEM_TEMPLATE = (
    "You generate ONE realistic user query to evaluate an AI system.\n\n"
    "Hard requirements:\n"
    "- The query value MUST be a single natural-language question a real user might ask.\n"
    "- Do NOT include explanations or preambles.\n"
    "- Do not copy example queries verbatim; write a fresh query.\n\n"
    "Query quality:\n"
    "- Express ALL tuple dimension values as concrete constraints.\n"
    "- Be specific, but keep it as short as possible while satisfying the constraints.\n"
    '- Do not mention "tuple", "dimensions", or the names of these instructions.\n\n'
    "User instructions:\n"
    "- When <instructions> tags appear in the context, they contain "
    "domain constraints and preferences that shape query generation.\n"
    "- Always respect these instructions alongside goals and tuple dimensions.\n\n"
    "Goal-guided generation:\n"
    "- When evaluation goals appear in <evaluation_goals> or <evaluation_goal> tags, "
    "they describe behaviors or failure modes to stress-test.\n"
    "- Craft the query so a real user would naturally trigger the described scenario. "
    "The goal constrains the kind of question, not the literal wording.\n"
    "- If a single goal is provided, make it the primary shaping constraint. "
    "If multiple goals are listed, balance them while favoring higher-weighted ones."
)

# Template for the user message in query generation
QUERY_USER_TEMPLATE = (
    "Context (treat as domain spec + constraints):\n"
    "<context>\n"
    "{context}\n"
    "</context>\n"
    "Tuple (dimension values; include ALL of these in the query):\n"
    "{tuple_lines}\n\n"
    "Task:\n"
    "- Write the single best user question that satisfies the context, "
    "reflects the tuple, and exercises any evaluation goals provided.\n"
    "- Do not mention the tuple, goals, or these instructions."
)


def _format_scalar_for_prompt(value: object) -> str:
    """Format a tuple scalar value for safe, readable inclusion in prompts."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        s = value.strip()
        # Keep the prompt single-line per dimension, but don't silently drop content.
        s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
        return s
    # Defensive fallback: tuples should only contain ScalarValue, but avoid crashing.
    s = str(value).strip()
    return s


def render_tuple_kv_lines(tuple: GeneratedTuple) -> str:
    """Render tuple dimension values as stable, one-per-line key/value pairs.

    Args:
        tuple: The generated tuple to render.

    Returns:
        A formatted string with one dimension per line.
    """
    if not tuple.values:
        return "- (no dimensions)"

    # Preserve insertion order so it matches the upstream dimension order.
    lines = [
        f"- {key}: {_format_scalar_for_prompt(value)}"
        for key, value in tuple.values.items()
    ]
    return "\n".join(lines)


def format_query_prompts(
    tuple: GeneratedTuple,
    context: str,
) -> tuple[str, str]:
    """Format prompts for query generation.

    Args:
        tuple: The generated tuple containing dimension values.
        context: The domain context and constraints.

    Returns:
        A tuple of (system_message, user_message).
    """
    today = datetime.date.today().isoformat()
    system_message = (
        f"{QUERY_SYSTEM_TEMPLATE}\n\n"
        f"Today's date is {today}. "
        "Use this to ground any time-sensitive language in the query."
    )
    tuple_lines = render_tuple_kv_lines(tuple)
    user_message = QUERY_USER_TEMPLATE.format(
        context=(context or "General").strip(),
        tuple_lines=tuple_lines,
    )

    return system_message, user_message
