from __future__ import annotations

from enum import Enum


class QueryMode(str, Enum):
    """Available strategies for turning tuples into natural language queries."""

    INSTRUCTOR = "instructor"
