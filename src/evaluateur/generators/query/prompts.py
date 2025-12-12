from __future__ import annotations

from collections.abc import Sequence

from evaluateur.models import GeneratedTuple


def build_tuple_lines(tuples: Sequence[GeneratedTuple]) -> list[str]:
    """Render tuples as compact bullet lines for prompts."""

    return [f"- {t.values}" for t in tuples]


def build_instructor_messages(*, tuples: Sequence[GeneratedTuple], context: str) -> list[dict[str, str]]:
    """Build Instructor chat messages for tuple -> query generation."""

    system_message = (
        "You create realistic user queries for evaluating an AI system. "
        "Each query should reflect the intent encoded in the tuple of dimension values."
    )

    tuple_descriptions = build_tuple_lines(tuples)
    user_message = (
        f"Domain/context:\n{context or 'General'}\n\n"
        "For each of the following tuples (dimension combinations), write a natural language "
        "query that a user might ask. Be specific and include enough detail to fully express "
        "the combination.\n\n"
        + "\n".join(tuple_descriptions)
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

