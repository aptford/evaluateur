from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from evaluateur.client import LLMClient
from evaluateur.generators.query.context import compose_context
from evaluateur.generators.query.dspy_backend import build_tuple_to_query_module, configure_lm, require_dspy
from evaluateur.generators.query.protocols import DSpyOptimizer
from evaluateur.goals import GoalSpec
from evaluateur.models import GeneratedQuery, GeneratedTuple

log = logging.getLogger(__name__)


def _default_query_metric(ex: Any, pred: Any, *_: Any, **__: Any) -> float:
    """Heuristic query metric (no external judging).

    This intentionally favors specificity and checks that tuple values appear
    in the generated query.
    """

    query = getattr(pred, "query", "") or ""
    q = str(query).lower()

    score = 0.0
    score += 0.1 if "?" in q else 0.0
    score += 0.1 if 25 <= len(q) <= 240 else 0.0

    tuple_json = getattr(ex, "tuple_json", None)
    if isinstance(tuple_json, str):
        try:
            tup = json.loads(tuple_json)
        except Exception:
            tup = None
        if isinstance(tup, dict):
            values = tup.get("values") or {}
            if isinstance(values, dict) and values:
                tokens = [str(v).lower() for v in values.values()]
                hits = sum(1 for tok in tokens if tok and tok in q)
                score += 0.7 * (hits / max(1, len(tokens)))

    return max(0.0, min(1.0, score))


class DSpyQueryGenerator:
    """Use DSPy to turn tuples into queries, optionally with optimization.

    DSPy module calls are synchronous internally, but this class exposes an
    async interface for consistency with the rest of the library.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        optimize: bool = False,
        optimizer: DSpyOptimizer | None = None,
        trainset: Sequence[Any] | None = None,
        valset: Sequence[Any] | None = None,
        goal_spec: GoalSpec | None = None,
        goal_prompt: str | None = None,
    ) -> None:
        # Fail fast if requested but unavailable.
        require_dspy()

        self._client = client
        configure_lm(client)

        self._goal_prompt = goal_prompt or (
            goal_spec.render_prompt() if goal_spec is not None else None
        )

        self._base_module = build_tuple_to_query_module()
        self._compiled_module: Any | None = None

        if optimizer is not None:
            if trainset is None and valset is None:
                # Explicitly do *not* attempt implicit lazy compilation.
                log.info(
                    "DSpyQueryGenerator: optimizer provided but no train/val set; skipping compilation"
                )
            else:
                log.info("DSpyQueryGenerator: compiling with custom optimizer")
                self._compiled_module = optimizer.compile(
                    student=self._base_module,
                    trainset=trainset,
                    valset=valset,
                )
        elif optimize:
            dspy = require_dspy()
            log.info("DSpyQueryGenerator: compiling with default MIPROv2 optimizer")
            try:
                default_optimizer = dspy.MIPROv2(  # type: ignore[attr-defined]
                    metric=_default_query_metric,
                    auto="light",
                )
                self._compiled_module = default_optimizer.compile(
                    student=self._base_module,
                    trainset=list(trainset or []),
                    valset=list(valset or []),
                )
            except Exception:
                log.warning("DSpyQueryGenerator: default optimizer failed, using base module")
                self._compiled_module = None

    async def generate(self, tuples: list[GeneratedTuple], context: str) -> list[GeneratedQuery]:
        results: list[GeneratedQuery] = []

        module = self._compiled_module or self._base_module
        using_compiled = self._compiled_module is not None
        log.info(
            "DSpyQueryGenerator: generating queries for %d tuples (compiled=%s)",
            len(tuples),
            using_compiled,
        )

        effective_context = compose_context(context, self._goal_prompt)

        for i, t in enumerate(tuples):
            tuple_json = t.model_dump_json()
            log.debug("DSpyQueryGenerator: processing tuple %d/%d", i + 1, len(tuples))
            pred = module(context=effective_context, tuple_json=tuple_json)
            results.append(
                GeneratedQuery(
                    query=pred.query,
                    source_tuple=t,
                    metadata={"goal_guided": bool(self._goal_prompt)},
                )
            )

        return results

