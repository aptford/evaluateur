"""Query prompt building utilities.

This module provides the mechanism for building query generation prompts.
The prompt templates (policy) are imported from the prompts module.
"""

from __future__ import annotations

from typing import Callable

from evaluateur.prompts.queries import (
    format_query_prompts,
    render_tuple_kv_lines,
)
from evaluateur.tuples.models import GeneratedTuple

# Re-export for backward compatibility
_render_tuple_kv_lines = render_tuple_kv_lines


def build_instructor_messages_for_tuple(
    *,
    tuple: GeneratedTuple,
    context: str,
    prompt_formatter: Callable[
        [GeneratedTuple, str], tuple[str, str]
    ] = format_query_prompts,
) -> list[dict[str, str]]:
    """Build Instructor chat messages for single tuple -> query generation.

    Parameters
    ----------
    tuple
        The generated tuple containing dimension values.
    context
        The domain context and constraints.
    prompt_formatter
        Callable that formats the prompts. Defaults to the standard formatter.
        This allows customizing prompts without changing mechanism code.

    Returns
    -------
    list[dict[str, str]]
        A list of message dicts for the Instructor API.
    """
    system_message, user_message = prompt_formatter(tuple, context)

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
