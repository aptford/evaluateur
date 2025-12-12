from __future__ import annotations

from typing import Any

from evaluateur.client import LLMClient


def import_dspy() -> Any | None:
    """Import DSPy if available, otherwise return None.

    This function is the only place that imports DSPy directly. All other
    modules should call this (or `require_dspy`) to keep DSPy optional.
    """

    try:
        import dspy  # type: ignore[import]
    except Exception:  # pragma: no cover - optional dependency
        return None
    return dspy


def require_dspy() -> Any:
    """Return the DSPy module, raising a clear error if not installed."""

    dspy = import_dspy()
    if dspy is None:  # pragma: no cover
        raise RuntimeError("DSPy is not installed but DSPy query mode was requested.")
    return dspy


def configure_lm(client: LLMClient) -> None:
    """Configure DSPy settings with the client's LM (or a default)."""

    dspy = require_dspy()
    lm = client.dspy_lm or dspy.LM(f"{client.provider}/{client.model_name}")
    dspy.settings.configure(lm=lm)


def build_tuple_to_query_module() -> Any:
    """Build the DSPy module for tuple -> query generation."""

    dspy = require_dspy()

    class TupleToQuerySignature(dspy.Signature):  # type: ignore[valid-type]
        """Convert a structured tuple into a natural language query."""

        context = dspy.InputField(desc="Domain or product context for the query.")
        tuple_json = dspy.InputField(desc="JSON representation of the tuple values.")
        query = dspy.OutputField(desc="A realistic natural language query.")

    return dspy.ChainOfThought(TupleToQuerySignature)


def build_refiner_module() -> Any:
    """Build a lightweight DSPy module to refine a draft query."""

    dspy = require_dspy()

    class RefineSignature(dspy.Signature):  # type: ignore[valid-type]
        """Refine an existing query for better evaluation coverage."""

        context = dspy.InputField()
        original_query = dspy.InputField()
        refined_query = dspy.OutputField()

    return dspy.ChainOfThought(RefineSignature)

