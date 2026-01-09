from __future__ import annotations

from evaluateur.generators.tuple.protocols import TupleGenerator
from evaluateur.generators.tuple.strategy import TupleStrategy
from evaluateur.generators.tuple.cross_product import CrossProductTupleGenerator
from evaluateur.generators.tuple.direct_llm import DirectLLMTupleGenerator

__all__ = [
    "CrossProductTupleGenerator",
    "DirectLLMTupleGenerator",
    "TupleGenerator",
    "TupleStrategy",
]
