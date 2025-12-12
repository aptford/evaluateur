from __future__ import annotations

from enum import Enum


class TupleStrategy(str, Enum):
    """Strategies for generating dimension tuples."""

    CROSS_PRODUCT = "cross_product"
    DIRECT_LLM = "direct_llm"

