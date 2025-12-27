from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal

from evaluateur.client import LLMClient
from evaluateur.goals import GoalSpec
from evaluateur.integrations.dspy.backend import build_tuple_to_query_module, configure_lm
from evaluateur.integrations.dspy.imports import require_dspy
from evaluateur.integrations.dspy.types import DSpyOptimizer, TupleToQueryModule
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


def _resolve_default_dspy_optimizer(
    dspy: Any,
    *,
    optimizer_name: Literal["gepa", "miprov2"],
    metric: Any,
    auto: str,
    reflection_lm: Any | None,
) -> Any:
    """Build the default DSPy optimizer (GEPA by default, MiProV2 fallback)."""

    gepa_cls = getattr(dspy, "GEPA", None)
    miprov2_cls = getattr(dspy, "MIPROv2", None)

    if optimizer_name == "gepa":
        if gepa_cls is not None:
            if reflection_lm is None:
                raise RuntimeError(
                    "DSPy GEPA requires a reflection LM. Configure DSPy settings (configure_lm) "
                    "or pass a client with dspy_lm set."
                )
            return gepa_cls(metric=metric, auto=auto, reflection_lm=reflection_lm)
        if miprov2_cls is not None:
            return miprov2_cls(metric=metric, auto=auto)
        raise RuntimeError("No supported DSPy optimizer found (GEPA/MIPROv2 unavailable).")

    # optimizer_name == "miprov2"
    if miprov2_cls is not None:
        return miprov2_cls(metric=metric, auto=auto)
    if gepa_cls is not None:
        if reflection_lm is None:
            raise RuntimeError(
                "DSPy GEPA requires a reflection LM. Configure DSPy settings (configure_lm) "
                "or pass a client with dspy_lm set."
            )
        return gepa_cls(metric=metric, auto=auto, reflection_lm=reflection_lm)
    raise RuntimeError("No supported DSPy optimizer found (GEPA/MIPROv2 unavailable).")


class DSpyQueryGenerator:
    """Use DSPy to turn tuples into queries, optionally with optimization.

    DSPy module calls are synchronous internally, but this class exposes an
    async interface for consistency with the rest of the library.

    When `optimize=True` and a train/val set is provided, Evaluateur will use
    GEPA by default (falling back to MiProV2 if needed).
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        optimize: bool = False,
        optimizer_name: Literal["gepa", "miprov2"] = "gepa",
        optimizer: DSpyOptimizer | None = None,
        trainset: Sequence[object] | None = None,
        valset: Sequence[object] | None = None,
        goal_spec: GoalSpec | None = None,
        goal_prompt: str | None = None,
    ) -> None:
        # Fail fast if requested but unavailable.
        require_dspy()

        # goal_spec/goal_prompt are currently unused by the base DSPy generator,
        # but are kept for forward compatibility with goal-conditioned DSPy modules.
        _ = goal_spec, goal_prompt

        self._client = client
        self._dspy_lm = configure_lm(client)
        self._optimizer_name = optimizer_name

        self._base_module: TupleToQueryModule = build_tuple_to_query_module()
        self._compiled_module: TupleToQueryModule | None = None

        if optimizer is not None:
            if trainset is None and valset is None:
                # Explicitly do *not* attempt implicit lazy compilation.
                log.info(
                    "DSpyQueryGenerator: optimizer provided but no train/val set; skipping compilation"
                )
            else:
                log.info("DSpyQueryGenerator: compiling with custom optimizer")
                compiled = optimizer.compile(
                    student=self._base_module,
                    trainset=list(trainset) if trainset is not None else None,
                    valset=list(valset) if valset is not None else None,
                )
                self._compiled_module = compiled  # type: ignore[assignment]
        elif optimize:
            if trainset is None and valset is None:
                # Do not attempt implicit compilation with no data.
                log.info(
                    "DSpyQueryGenerator: optimize requested but no train/val set; skipping compilation"
                )
            else:
                dspy = require_dspy()
                auto = "light"
                log.info(
                    "DSpyQueryGenerator: compiling with default optimizer=%s (auto=%s)",
                    self._optimizer_name,
                    auto,
                )
                try:
                    default_optimizer = _resolve_default_dspy_optimizer(
                        dspy,
                        optimizer_name=self._optimizer_name,
                        metric=_default_query_metric,
                        auto=auto,
                        reflection_lm=self._dspy_lm,
                    )
                    compiled = default_optimizer.compile(
                        student=self._base_module,
                        trainset=list(trainset) if trainset is not None else None,
                        valset=list(valset) if valset is not None else None,
                    )
                    self._compiled_module = compiled  # type: ignore[assignment]
                except Exception:
                    log.warning(
                        "DSpyQueryGenerator: default optimizer failed, using base module"
                    )
                    self._compiled_module = None

    async def generate(
        self, tuples: AsyncIterator[GeneratedTuple], context: str
    ) -> AsyncIterator[GeneratedQuery]:
        module = self._compiled_module or self._base_module
        using_compiled = self._compiled_module is not None
        log.info(
            "DSpyQueryGenerator: generating queries (compiled=%s)",
            using_compiled,
        )
        i = 0
        async for t in tuples:
            i += 1
            tuple_json = t.model_dump_json()
            log.debug("DSpyQueryGenerator: processing tuple %d", i)
            pred = module(context=context, tuple_json=tuple_json)
            yield GeneratedQuery(query=pred.query, source_tuple=t)


