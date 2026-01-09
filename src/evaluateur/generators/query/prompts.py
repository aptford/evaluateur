from __future__ import annotations

from evaluateur.models import GeneratedTuple


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


def _render_tuple_kv_lines(tuple: GeneratedTuple) -> str:
    """Render tuple dimension values as stable, one-per-line key/value pairs."""

    if not tuple.values:
        return "- (no dimensions)"

    # Preserve insertion order so it matches the upstream dimension order.
    lines = [
        f"- {key}: {_format_scalar_for_prompt(value)}"
        for key, value in tuple.values.items()
    ]
    return "\n".join(lines)


def build_instructor_messages_for_tuple(
    *, tuple: GeneratedTuple, context: str
) -> list[dict[str, str]]:
    """Build Instructor chat messages for single tuple -> query generation."""

    system_message = (
        "You generate ONE realistic user query to evaluate an AI system.\n\n"
        "Hard requirements:\n"
        "- The query value MUST be a single natural-language question a real user might ask.\n"
        "- Do NOT include explanations or preambles.\n\n"
        "- Do not copy example queries verbatim; write a fresh query.\n\n"
        "Query quality:\n"
        "- Express ALL tuple dimension values as concrete constraints.\n"
        "- Be specific, but keep it as short as possible while satisfying the constraints.\n"
        '- Do not mention "tuple", "dimensions", or the names of these instructions.'
    )

    tuple_lines = _render_tuple_kv_lines(tuple)
    user_message = (
        "Context (treat as domain spec + constraints):\n"
        "<context>\n"
        f"{(context or 'General').strip()}\n"
        "</context>\n"
        "Tuple (dimension values; include ALL of these in the query):\n"
        f"{tuple_lines}\n\n"
        "Task:\n"
        "- Write the single best user question that satisfies the Context constraints and reflects the tuple.\n"
        "- Do not mention the tuple or these instructions."
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
