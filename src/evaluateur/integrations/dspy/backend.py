from __future__ import annotations

from typing import Any

from evaluateur.client import LLMClient
from evaluateur.integrations.dspy.imports import require_dspy
from evaluateur.integrations.dspy.types import RefinerModule, TupleToQueryModule


def configure_lm(client: LLMClient) -> Any:
    """Configure DSPy settings with the client's LM (or a default).

    Returns the resolved DSPy LM instance used for configuration.
    """

    dspy = require_dspy()
    lm = client.dspy_lm or dspy.LM(f"{client.provider}/{client.model_name}")
    dspy.settings.configure(lm=lm)
    return lm


def build_tuple_to_query_module() -> TupleToQueryModule:
    """Build the DSPy module for tuple -> query generation."""

    dspy = require_dspy()

    class TupleToQuerySignature(dspy.Signature):  # type: ignore[valid-type]
        """Convert a structured tuple into a natural language query."""

        context = dspy.InputField(desc="Domain or product context for the query.")
        tuple_json = dspy.InputField(desc="JSON representation of the tuple values.")
        query = dspy.OutputField(desc="A realistic natural language query.")

    return dspy.ChainOfThought(TupleToQuerySignature)


def build_refiner_module() -> RefinerModule:
    """Build a lightweight DSPy module to refine a draft query."""

    dspy = require_dspy()

    class RefineSignature(dspy.Signature):  # type: ignore[valid-type]
        """Refine an existing query for better evaluation coverage."""

        context = dspy.InputField()
        original_query = dspy.InputField()
        refined_query = dspy.OutputField()

    return dspy.ChainOfThought(RefineSignature)


