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
TAG_GOALS_OPEN = "<evaluation_goals>"
TAG_GOALS_CLOSE = "</evaluation_goals>"
TAG_GOAL_OPEN = "<evaluation_goal>"
TAG_GOAL_CLOSE = "</evaluation_goal>"

# Separator used to join context chunks
CONTEXT_SEPARATOR = "\n\n"
