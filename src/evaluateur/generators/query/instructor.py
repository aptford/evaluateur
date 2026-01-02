from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from inspect import isawaitable

from pydantic import BaseModel

from evaluateur.client import LLMClient
from evaluateur.generators.query.prompts import build_instructor_messages_for_tuple
from evaluateur.generators.query.protocols import ContextBuilder
from evaluateur.models import GeneratedQuery, GeneratedTuple
from evaluateur.models import QueryMetadata

log = logging.getLogger(__name__)


class _QueryModel(BaseModel):
    query: str


class InstructorQueryGenerator:
    """Use Instructor directly to synthesize queries from tuples asynchronously."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def generate(
        self,
        tuples: AsyncIterator[GeneratedTuple],
        context: str,
        *,
        context_builder: ContextBuilder | None = None,
    ) -> AsyncIterator[GeneratedQuery]:
        """Generate queries from tuples.

        Parameters
        ----------
        tuples
            Async stream of tuples to turn into natural language queries.
        context
            Default domain/context to use when no per-tuple context is provided.
        context_builder
            Optional callable that can vary context per tuple and attach per-query
            metadata. The callable may be sync or async and must return
            ``(context: str, metadata: Mapping[str, object])``.
        """

        client = self._client.instructor_client

        async def _resolve_context_and_metadata(
            t: GeneratedTuple,
        ) -> tuple[str, QueryMetadata]:
            if context_builder is None:
                return context, QueryMetadata()

            maybe = context_builder(t)
            built = await maybe if isawaitable(maybe) else maybe
            ctx, meta = built
            if isinstance(meta, QueryMetadata):
                return ctx, meta
            return ctx, QueryMetadata.model_validate(meta)

        i = 0
        async for t in tuples:
            i += 1
            effective_context, metadata = await _resolve_context_and_metadata(t)
            messages = build_instructor_messages_for_tuple(
                tuple=t, context=effective_context
            )
            log.debug("InstructorQueryGenerator prompt (tuple=%d):\n%s", i, messages[-1]["content"])

            result: _QueryModel = await client.chat.completions.create(
                model=self._client.model_name,
                response_model=_QueryModel,
                messages=messages,
            )
            yield GeneratedQuery(
                query=result.query,
                source_tuple=t,
                metadata=metadata,
            )

