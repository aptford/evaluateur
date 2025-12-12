from __future__ import annotations

import logging

from pydantic import BaseModel

from evaluateur.client import LLMClient
from evaluateur.generators.query.prompts import build_instructor_messages
from evaluateur.models import GeneratedQuery, GeneratedTuple

log = logging.getLogger(__name__)


class InstructorQueryGenerator:
    """Use Instructor directly to synthesize queries from tuples asynchronously."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def generate(self, tuples: list[GeneratedTuple], context: str) -> list[GeneratedQuery]:
        client = self._client.instructor_client
        log.info("InstructorQueryGenerator: generating queries for %d tuples", len(tuples))

        class QueryListModel(BaseModel):
            queries: list[GeneratedQuery]

        messages = build_instructor_messages(tuples=tuples, context=context)
        log.debug("InstructorQueryGenerator prompt:\n%s", messages[-1]["content"])

        result: QueryListModel = await client.chat.completions.create(
            model=self._client.model_name,
            response_model=QueryListModel,
            messages=messages,
        )
        log.debug("InstructorQueryGenerator: received %d queries", len(result.queries))
        return result.queries

