from __future__ import annotations

import itertools
import logging
import math
from collections.abc import AsyncIterator

from pydantic import BaseModel

from evaluateur.client import LLMClient
from evaluateur.generators.tuple.options_adapter import extract_dimension_values
from evaluateur.models import GeneratedTuple

log = logging.getLogger(__name__)


class CrossProductTupleGenerator:
    """Generate tuples via cross product as an async iterator.

    This mirrors the "cross product then filter" approach from Hamel Husain's
    FAQ: it guarantees coverage of the dimension space at the cost of volume.
    """

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client

    async def generate(self, options: BaseModel, count: int) -> AsyncIterator[GeneratedTuple]:
        field_names, value_lists = extract_dimension_values(options)

        total = math.prod((len(v) for v in value_lists), start=1) if field_names else 1
        log.debug(
            "CrossProductTupleGenerator: ~%d total combinations from %d fields",
            total,
            len(field_names),
        )

        combos = itertools.product(*value_lists) if value_lists else iter([()])
        if count > 0:
            combos = itertools.islice(combos, count)

        yielded = 0
        for combo in combos:
            yielded += 1
            yield GeneratedTuple(values={name: value for name, value in zip(field_names, combo)})

        log.debug("CrossProductTupleGenerator: yielded %d tuples", yielded)

