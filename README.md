# Evaluateur

Synthetic evaluation helper for LLM applications, built around the
**dimensions → tuples → queries** flow described in [Hamel Husain's FAQ](https://hamel.dev/blog/posts/evals-faq/what-is-the-best-approach-for-generating-synthetic-data.html).

## Installation

The project is packaged as a normal Python library. With `uv`:

```bash
uv add evaluateur
```

## Basic usage

Define a Pydantic model that represents the dimensions of your evaluation
space, then use the `Evaluator` to generate options and queries:

```python
import asyncio
from pydantic import BaseModel, Field

from evaluateur import Evaluator, TupleStrategy


class Query(BaseModel):
    payer: str = Field(..., description="insurance payer, like Cigna")
    age: str = Field(..., description="patient age category, like 'adult' or 'pediatric'")
    complexity: str = Field(
        ...,
        description="complexity of the query to account for the edge cases, like 'off-label', 'comorbidities', etc",
    )
    geography: str = Field(..., description="geography indicator, like a zip code, specific state or county")


async def main() -> None:
    evaluator = Evaluator(Query)

    # Step 1: generate options for each dimension using Instructor
    options = await evaluator.options(
        instructions="Focus on common US payers and edge-case clinical scenarios.",
        count_per_field=5,
    )

    # Step 2: stream tuples -> natural language queries
    async for q in evaluator.run(
        options=options,
        tuple_strategy=TupleStrategy.CROSS_PRODUCT,
        tuple_count=50,
        seed=0,
        instructions="""
Write realistic user questions.
Keep them short but specific.
Don't include any extra explanation outside the query itself.
""",
    ):
        print(q.source_tuple.values, "->", q.query)


asyncio.run(main())
```

The evaluator uses environment variables (for example `OPENAI_API_KEY`)
and supports any provider that `instructor` supports. You can customise the
provider and model via the `LLMClient` helper if needed.

If your input model already uses iterator fields (for example
`payer: list[str] = ["Cigna", "Aetna"]`), those lists are treated as fixed
options and are not modified by `generate_options()`. Scalar fields of any
basic type (`str`, `int`, `float`, and so on) are turned into lists of
options automatically.

### Instructions

Evaluateur accepts instructions at each stage:

- **Option generation**: use `Evaluator.options(instructions=...)` to guide what
  _dimension values_ to propose (e.g. "Focus on common US payers.").
- **Tuple generation**: use `Evaluator.tuples(instructions=...)` to guide how
  tuples are sampled when the generator supports it.
- **Query generation**: use `Evaluator.queries(instructions=...)` to guide how the
  _natural language query_ should be written (e.g. "Keep the question short
  and specific.").
- **Query mode**: use `Evaluator.queries(query_mode=...)` or `Evaluator.run(query_mode=...)`
  to select a query generator mode (currently `QueryMode.INSTRUCTOR`).

`Evaluator.run(instructions=...)` shares the same instruction string across all
three stages.

## Tuple generation: seeded sampling for cross product

When `TupleStrategy.CROSS_PRODUCT` is used and `0 < count < total_combinations`,
Evaluateur returns a **seeded randomized sample** of the cartesian product
(_uniform without replacement_). This helps avoid always taking the "first N"
combinations when the space is large.

- To get reproducible results, set `seed=...`.
- Changing the seed gives you a different randomized subset.

## Goal-guided query optimization

You can guide query generation by providing a `GoalSpec` (structured) or
free-form text (which is parsed into a `GoalSpec`).

Goals are flat and optionally categorized using the CTO framework
(Components / Trajectories / Outcomes) or any custom categories.

Structured text (numbered/bulleted lists) is parsed without an LLM call.
CTO section headers like `Components:` are auto-detected. Free-form text
falls back to LLM enrichment.

### Sampling goals per query (diversity mode)

By default, Evaluateur picks a single goal _per generated query_.
This helps ensure one run produces a mix of different stress-test styles.

```python
import asyncio
from pydantic import BaseModel, Field

from evaluateur import Evaluator, Goal, GoalSpec


class Query(BaseModel):
    payer: str = Field(...)
    age: str = Field(...)
    complexity: str = Field(...)
    geography: str = Field(...)


async def main() -> None:
    evaluator = Evaluator(Query)

    goals = GoalSpec(goals=[
        Goal(name="freshness checks", text="Test data currency", category="components"),
        Goal(name="conflict handling", text="Test conflicting sources", category="trajectories"),
        Goal(name="checklist-ready", text="Request structured output", category="outcomes"),
    ])

    async for q in evaluator.run(
        seed=0,
        instructions="Make the question sound like a real user.",
        goals=goals,
    ):
        print(q.metadata.goal_focus, "->", q.query)
        break


asyncio.run(main())
```

## Context builders (advanced)

A **context builder** is a callable used by query generators to vary the prompt
**per tuple**, instead of using one shared `context` string for the whole run.

It returns two things:

- A `context` string to include in the prompt for that tuple
- Optional per-query `metadata` (extra keys are allowed) that will be merged into `q.metadata`

Evaluateur uses a context builder internally when you enable goal sampling
(`goal_mode="sample"`) so that each generated query can focus on a
different goal.

If you write a custom query generator, accept `context_builder` and fall back to
the base `context` when it is not provided.

```python
from __future__ import annotations

from collections.abc import AsyncIterator

from evaluateur.queries import ContextBuilder, GeneratedQuery, GeneratedTuple


class MyQueryGenerator:
    async def generate(
        self,
        tuples: AsyncIterator[GeneratedTuple],
        context: str,
        *,
        context_builder: ContextBuilder | None = None,
    ) -> AsyncIterator[GeneratedQuery]:
        async for t in tuples:
            if context_builder is None:
                effective_context, meta = context, {}
            else:
                effective_context, meta = context_builder(t)

            # Use effective_context to build your prompt, and attach meta if you want.
            yield GeneratedQuery(query=f"ctx={effective_context}", source_tuple=t, metadata=meta)
```

## Generator factories (advanced)

If you want direct access to the built-in generators, use the public factories:

```python
from evaluateur import LLMClient, build_query_generator, build_tuple_generator
from evaluateur import QueryMode, TupleStrategy

client = LLMClient.from_env()
tuple_gen = build_tuple_generator(client=client, strategy=TupleStrategy.CROSS_PRODUCT)
query_gen = build_query_generator(client=client, mode=QueryMode.INSTRUCTOR)
```

Structured goals:

```python
import asyncio
from pydantic import BaseModel, Field

from evaluateur import Evaluator, Goal, GoalSpec


class Query(BaseModel):
    payer: str = Field(..., description="insurance payer, like Cigna")
    age: str = Field(..., description="patient age category, like 'adult' or 'pediatric'")
    complexity: str = Field(..., description="complexity bucket, e.g. comorbidities, off-label")
    geography: str = Field(..., description="geography indicator, like state or zip code")


async def main() -> None:
    evaluator = Evaluator(Query)

    goals = GoalSpec(goals=[
        Goal(
            name="freshness checks",
            text="Queries should ask about effective dates and latest policy versions",
            category="components",
        ),
        Goal(
            name="grounded claims",
            text="Queries should request citations and policy section references",
            category="components",
        ),
        Goal(
            name="conflict integration",
            text="Test behavior when payer policy conflicts with FDA label",
            category="trajectories",
        ),
        Goal(
            name="checklist-ready",
            text="Queries should request structured lists of requirements",
            category="outcomes",
        ),
    ])

    async for q in evaluator.run(
        goals=goals,
    ):
        print(q.metadata.query_goals.model_dump() if q.metadata.query_goals else None)
        break


asyncio.run(main())
```

Free-form goals (parsed without LLM when structured):

```python
import asyncio
from pydantic import BaseModel, Field

from evaluateur import Evaluator


class Query(BaseModel):
    payer: str = Field(...)
    age: str = Field(...)
    complexity: str = Field(...)
    geography: str = Field(...)


async def main() -> None:
    evaluator = Evaluator(Query)

    i = 0
    async for q in evaluator.run(
        goals="""
Components:
- Prioritize freshness checks and grounded citations
- Missing-source detection (don't proceed silently)

Trajectories:
- Conflict handling and recovery behavior
- Re-try, switch tools, or escalate when evidence conflicts

Outcomes:
- Produce checklist-ready outputs
- Easy to review and hard to misuse
""",
    ):
        print(q.query)
        i += 1
        if i >= 3:
            break


asyncio.run(main())
```
