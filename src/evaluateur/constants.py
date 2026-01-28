"""Shared constants for evaluateur.

This module centralizes magic strings and constants used across the codebase,
making them easier to maintain and modify.
"""

from __future__ import annotations

# XML-style tags used in prompts
TAG_INSTRUCTIONS_OPEN = "<instructions>"
TAG_INSTRUCTIONS_CLOSE = "</instructions>"
TAG_CONTEXT_OPEN = "<context>"
TAG_CONTEXT_CLOSE = "</context>"

# Separator used to join context chunks
CONTEXT_SEPARATOR = "\n\n"
