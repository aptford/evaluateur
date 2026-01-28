"""Query metadata merge policy.

This module separates the merge policy from the QueryMetadata data model,
following the principle of separating policy from mechanism.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluateur.queries.models import QueryMetadata


def merge_query_metadata(
    *,
    run_metadata: QueryMetadata,
    per_query_metadata: QueryMetadata,
) -> QueryMetadata:
    """Merge evaluator run metadata with per-query metadata.

    This function implements the merge policy: run metadata provides defaults,
    and per-query metadata wins on conflicts.

    Args:
        run_metadata: Metadata from the evaluator run (defaults).
        per_query_metadata: Metadata from the query generator (overrides).

    Returns:
        A new QueryMetadata instance with merged values.
    """
    # Import here to avoid circular imports
    from evaluateur.queries.models import QueryMetadata

    merged = {
        **run_metadata.model_dump(exclude_none=True),
        **per_query_metadata.model_dump(exclude_none=True, exclude_unset=True),
    }
    return QueryMetadata.model_validate(merged)
