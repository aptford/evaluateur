from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Iterable, Sequence

from evaluateur.goals import GoalItem, GoalSpec

log = logging.getLogger(__name__)

try:
    import dspy  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    dspy = None  # type: ignore[assignment]


class JudgeBackend(str, Enum):
    """How query quality is scored during optimization."""

    HEURISTIC = "heuristic"
    LLM = "llm"


def _safe_json_loads(maybe_json: str) -> dict[str, Any] | None:
    try:
        loaded = json.loads(maybe_json)
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None


def _extract_predicted_query(pred: Any) -> str:
    for key in ("query", "refined_query"):
        val = getattr(pred, key, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return str(pred).strip()


def _iter_goal_items(spec: GoalSpec) -> Iterable[tuple[str, GoalItem]]:
    for layer_name, layer in (
        ("components", spec.components),
        ("trajectories", spec.trajectories),
        ("outcomes", spec.outcomes),
    ):
        for it in layer.items:
            yield layer_name, it


@dataclass(frozen=True)
class HeuristicGoalJudge:
    """Deterministic goal scoring.

    This judge is designed for:
    - offline tests
    - fast iteration
    - fallback when LLM scoring is unavailable
    """

    goal_spec: GoalSpec

    def score(self, *, query: str, tuple_json: str | None = None, context: str | None = None) -> float:
        q = (query or "").lower()

        # Basic query quality heuristics.
        base = 0.0
        base += 0.1 if "?" in q else 0.0
        base += 0.1 if len(q) >= 25 else 0.0
        base += 0.1 if len(q) <= 240 else 0.0

        # Encourage including tuple values (drives specificity).
        if tuple_json:
            tup = _safe_json_loads(tuple_json)
            if tup and isinstance(tup.get("values"), dict):
                values = [str(v).lower() for v in tup["values"].values()]
                if values:
                    hits = sum(1 for v in values if v and v in q)
                    base += 0.2 * (hits / max(1, len(values)))

        # Encourage leveraging context.
        if context:
            ctx_tokens = [t.strip().lower() for t in context.split() if len(t.strip()) >= 6]
            if ctx_tokens:
                hits = sum(1 for t in ctx_tokens[:10] if t in q)
                base += 0.1 * min(1.0, hits / 3.0)

        # Goal satisfaction: must_include / avoid across all layers.
        weighted = 0.0
        denom = 0.0
        for _layer, it in _iter_goal_items(self.goal_spec):
            if it.weight <= 0:
                continue
            denom += it.weight

            inc = 1.0
            if it.must_include:
                present = sum(1 for tok in it.must_include if tok.strip().lower() in q)
                inc = present / max(1, len(it.must_include))

            avoid_penalty = 0.0
            if it.avoid:
                present_avoid = sum(1 for tok in it.avoid if tok.strip().lower() in q)
                avoid_penalty = present_avoid / max(1, len(it.avoid))

            # We emphasize inclusion over avoidance.
            item_score = 0.75 * inc + 0.25 * (1.0 - avoid_penalty)
            weighted += it.weight * item_score

        goal_score = weighted / denom if denom > 0 else 0.5

        # Combine. Bound to [0, 1].
        combined = 0.45 * base + 0.55 * goal_score
        return max(0.0, min(1.0, combined))


class GoalGuidedQueryOptimizer:
    """DSPy optimizer that steers query generation to match a `GoalSpec`.

    This wraps a DSPy teleprompter (default: MIPROv2) with a goal-aware metric.

    Notes
    -----
    - The metric is intentionally *query-centric*: it scores the final query string.
    - For goal-driven systems, this tends to produce a better UX than optimizing
      purely on tuple resemblance.
    """

    def __init__(
        self,
        goal_spec: GoalSpec,
        *,
        judge_backend: JudgeBackend = JudgeBackend.LLM,
        auto: str = "light",
        teleprompter_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.goal_spec = goal_spec
        self.judge_backend = judge_backend
        self.auto = auto
        self._teleprompter_factory = teleprompter_factory

        if self.judge_backend == JudgeBackend.HEURISTIC:
            self._heuristic = HeuristicGoalJudge(goal_spec)
        else:
            self._heuristic = None

        self._goal_prompt = goal_spec.render_prompt()

    def _build_metric(self) -> Callable[..., float]:
        if self.judge_backend == JudgeBackend.HEURISTIC:

            def metric(ex: Any, pred: Any, *_: Any, **__: Any) -> float:
                query = _extract_predicted_query(pred)
                tuple_json = getattr(ex, "tuple_json", None)
                context = getattr(ex, "context", None)
                assert self._heuristic is not None
                return float(self._heuristic.score(query=query, tuple_json=tuple_json, context=context))

            return metric

        if dspy is None:  # pragma: no cover
            raise RuntimeError("DSPy is not installed but LLM judge backend was requested.")

        # LLM-based judge: ask for a strict, rubric-driven score in [0, 1].
        class _GoalJudgeSignature(dspy.Signature):  # type: ignore[valid-type]
            """Score how well the query matches the optimization goals."""

            goal_prompt = dspy.InputField(desc="Rubric and priorities")
            tuple_json = dspy.InputField(desc="Tuple JSON for specificity", default="")
            query = dspy.InputField(desc="Candidate query")
            score = dspy.OutputField(desc="A float in [0, 1]")
            rationale = dspy.OutputField(desc="One short sentence explaining the score")

        judge = dspy.Predict(_GoalJudgeSignature)

        @lru_cache(maxsize=2048)
        def _cached(goal_prompt: str, tuple_json: str, query: str) -> float:
            pred = judge(goal_prompt=goal_prompt, tuple_json=tuple_json, query=query)
            raw = getattr(pred, "score", None)
            try:
                score_val = float(str(raw).strip())
            except Exception:
                score_val = 0.0
            return max(0.0, min(1.0, score_val))

        def metric(ex: Any, pred: Any, *_: Any, **__: Any) -> float:
            query = _extract_predicted_query(pred)
            tuple_json = getattr(ex, "tuple_json", "") or ""
            return float(_cached(self._goal_prompt, str(tuple_json), query))

        return metric

    def compile(
        self,
        student: Any,
        trainset: Sequence[Any] | None = None,
        valset: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Compile/optimize a DSPy module using a goal-aware metric."""

        if dspy is None:  # pragma: no cover - defensive
            raise RuntimeError("DSPy is not installed but a DSPy optimizer was requested.")

        metric = self._build_metric()

        teleprompter_factory = self._teleprompter_factory
        if teleprompter_factory is None:
            teleprompter_factory = getattr(dspy, "MIPROv2", None)

        if teleprompter_factory is None:  # pragma: no cover
            raise RuntimeError("DSPy MIPROv2 is not available in this DSPy version.")

        teleprompter = teleprompter_factory(metric=metric, auto=self.auto)

        log.info(
            "GoalGuidedQueryOptimizer: compiling (judge=%s, auto=%s)",
            self.judge_backend.value,
            self.auto,
        )

        compiled = teleprompter.compile(
            student=student,
            trainset=list(trainset or []),
            valset=list(valset or []),
            **kwargs,
        )
        return compiled
