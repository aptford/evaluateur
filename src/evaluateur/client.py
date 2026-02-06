"""Internal client resolution for evaluateur.

This module is not part of the public API. It provides a minimal data
bundle (``LLMClient``) and a resolver function that translates user-facing
parameters into that bundle.
"""

from __future__ import annotations

import os
from typing import Any, NamedTuple

import instructor
from dotenv import load_dotenv

DEFAULT_MODEL = os.getenv("EVALUATEUR_MODEL", "openai/gpt-4.1-mini")


class LLMClient(NamedTuple):
    """Internal bundle: async instructor client + model name."""

    instructor_client: Any
    model_name: str


def resolve_client(
    *,
    llm: str | None = None,
    client: Any | None = None,
    model_name: str | None = None,
) -> LLMClient:
    """Build an LLMClient from user-facing parameters.

    Parameters
    ----------
    llm
        A ``"provider/model-name"`` string, e.g. ``"openai/gpt-4.1-mini"``
        or ``"anthropic/claude-3-5-sonnet-latest"``.  Mutually exclusive
        with *client*.
    client
        A pre-configured async Instructor client.  Must be paired with
        *model_name*.  Mutually exclusive with *llm*.
    model_name
        The model identifier passed to ``chat.completions.create(model=...)``.
        Required when *client* is provided, ignored otherwise.

    Returns
    -------
    LLMClient
        A ``(instructor_client, model_name)`` pair ready for internal use.

    Raises
    ------
    ValueError
        If both *llm* and *client* are given, if *client* is given without
        *model_name*, or if the *llm* string is not in ``"provider/model"``
        format.
    """
    if llm and client:
        raise ValueError("Provide 'llm' or 'client', not both.")

    if client is not None:
        if model_name is None:
            raise ValueError(
                "'model_name' is required when passing a pre-configured client."
            )
        return LLMClient(instructor_client=client, model_name=model_name)

    load_dotenv()
    model_str = llm or DEFAULT_MODEL

    if "/" not in model_str:
        raise ValueError(
            f"Expected 'provider/model-name' format, got: {model_str!r}. "
            f"Examples: 'openai/gpt-4.1-mini', 'anthropic/claude-3-5-sonnet-latest'"
        )

    _, name = model_str.split("/", 1)
    return LLMClient(
        instructor_client=instructor.from_provider(model_str, async_client=True),
        model_name=name,
    )
