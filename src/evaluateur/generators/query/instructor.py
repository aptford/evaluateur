from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from pydantic import BaseModel

from evaluateur.client import LLMClient
from evaluateur.generators.query.prompts import build_instructor_messages_for_tuple
from evaluateur.models import GeneratedQuery, GeneratedTuple

log = logging.getLogger(__name__)


class InstructorQueryGenerator:
    """Use Instructor directly to synthesize queries from tuples asynchronously."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def generate(
        self, tuples: AsyncIterator[GeneratedTuple], context: str
    ) -> AsyncIterator[GeneratedQuery]:
        client = self._client.instructor_client

        class QueryModel(BaseModel):
            query: str

        i = 0
        async for t in tuples:
            i += 1
            messages = build_instructor_messages_for_tuple(tuple=t, context=context)
            log.debug("InstructorQueryGenerator prompt (tuple=%d):\n%s", i, messages[-1]["content"])

            result: QueryModel = await client.chat.completions.create(
                model=self._client.model_name,
                response_model=QueryModel,
                messages=messages,
            )
            yield GeneratedQuery(query=result.query, source_tuple=t)

    async def generate_with_context_builder(
        self,
        tuples: AsyncIterator[GeneratedTuple],
        context_builder,
    ) -> AsyncIterator[GeneratedQuery]:
        """Generate queries, computing context per tuple.

        This is used by sampling modes where each query should receive a
        different goal focus area (e.g., components vs trajectories vs outcomes).
        """

        client = self._client.instructor_client

        class QueryModel(BaseModel):
            query: str

        i = 0
        async for t in tuples:
            i += 1
            context, meta = context_builder(t)
            messages = build_instructor_messages_for_tuple(tuple=t, context=context)
            log.debug(
                "InstructorQueryGenerator prompt (tuple=%d):\n%s",
                i,
                messages[-1]["content"],
            )

            result: QueryModel = await client.chat.completions.create(
                model=self._client.model_name,
                response_model=QueryModel,
                messages=messages,
            )
            yield GeneratedQuery(query=result.query, source_tuple=t, metadata=meta)

