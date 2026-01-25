from __future__ import annotations

from dataclasses import dataclass, field

from evaluateur.configs.options import OptionsConfig
from evaluateur.configs.query import QueryConfig
from evaluateur.configs.tuple import TupleConfig


@dataclass(frozen=True)
class RunConfig:
    """Unified configuration for a full evaluation run.

    Combines options, tuple, and query configuration into a single object
    for the ``Evaluator.run()`` method.
    """

    options: OptionsConfig = field(default_factory=OptionsConfig)
    tuples: TupleConfig = field(default_factory=TupleConfig)
    queries: QueryConfig = field(default_factory=QueryConfig)
