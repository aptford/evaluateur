from __future__ import annotations

from evaluateur.models import GeneratedTuple


def build_instructor_messages_for_tuple(
    *, tuple: GeneratedTuple, context: str
) -> list[dict[str, str]]:
    """Build Instructor chat messages for single tuple -> query generation."""

    system_message = (
        "You create realistic user queries for evaluating an AI system. "
        "The query should reflect the intent encoded in the tuple of dimension values."
    )

    user_message = (
        f"Domain/context:\n{context or 'General'}\n\n"
        "Given this tuple (dimension combination), write a natural language query that a user "
        "might ask. Be specific and include enough detail to fully express the combination.\n\n"
        f"Tuple:\n- {tuple.values}"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

