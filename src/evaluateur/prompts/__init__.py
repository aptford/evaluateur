"""Prompt templates for LLM interactions.

This module centralizes all prompt templates used by evaluateur, separating
policy (what prompts to use) from mechanism (how to execute LLM calls).
"""

from __future__ import annotations

from evaluateur.prompts.options import (
    OPTIONS_SYSTEM_TEMPLATE,
    OPTIONS_USER_TEMPLATE,
    format_options_prompts,
)
from evaluateur.prompts.queries import (
    QUERY_SYSTEM_TEMPLATE,
    QUERY_USER_TEMPLATE,
    format_query_prompts,
)
from evaluateur.prompts.tuples import (
    TUPLES_SYSTEM_TEMPLATE,
    TUPLES_USER_TEMPLATE,
    format_tuples_prompts,
)

__all__ = [
    # Options prompts
    "OPTIONS_SYSTEM_TEMPLATE",
    "OPTIONS_USER_TEMPLATE",
    "format_options_prompts",
    # Tuples prompts
    "TUPLES_SYSTEM_TEMPLATE",
    "TUPLES_USER_TEMPLATE",
    "format_tuples_prompts",
    # Queries prompts
    "QUERY_SYSTEM_TEMPLATE",
    "QUERY_USER_TEMPLATE",
    "format_query_prompts",
]
