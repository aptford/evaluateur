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

from evaluateur import Evaluator, QueryMode, TupleStrategy
from evaluateur.configs import QueryConfig, TupleConfig


class Query(BaseModel):
    payer: str = Field(..., description="insurance payer, like Cigna")
    age: str = Field(..., description="patient age category, like 'adult' or 'pediatric'")
    complexity: str = Field(
        ...,
        description="complexity of the query to account for the edge cases, like 'off-label', 'comorbidities', etc",
    )
    geography: str = Field(..., description="geography indicator, like a zip code, specific state or county")


async def main() -> None:
    evaluator = Evaluator(Query, context="Healthcare prior authorization")

    # Step 1: generate options for each dimension using Instructor
    options = await evaluator.options(
        instructions="Focus on common US payers and edge-case clinical scenarios.",
    )

    # Step 2: stream tuples -> natural language queries
    async for q in evaluator.run(
        options=options,
        tuple_config=TupleConfig(strategy=TupleStrategy.CROSS_PRODUCT, count=50, seed=0),
        query_config=QueryConfig(mode=QueryMode.INSTRUCTOR),
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

## Tuple generation: seeded sampling for cross product

When `TupleStrategy.CROSS_PRODUCT` is used and `0 < count < total_combinations`,
Evaluateur returns a **seeded randomized sample** of the cartesian product
(*uniform without replacement*). This helps avoid always taking the “first N”
combinations when the space is large.

- To get reproducible results, set `TupleConfig(seed=...)`.
- Changing the seed gives you a different randomized subset.

## Goal-guided query optimization (Components / Trajectories / Outcomes)

You can guide query generation using the three-layer framework by providing a
`GoalSpec` (structured) or free-form text (which is normalized into a `GoalSpec`).

Goals are used to **condition queries per run**, so you can iterate quickly.
If you provide `GoalItem.examples`, they are included in the internal goal prompt passed to the query generator.

### Sampling goals per query (diversity mode)

By default, Evaluateur picks a single focus area
(**components**, **trajectories**, or **outcomes**) *per generated query*.
This helps ensure one run produces a mix of different stress-test styles.

```python
import asyncio
from pydantic import BaseModel, Field

from evaluateur import Evaluator, GoalItem, GoalLayer, GoalSpec, QueryMode
from evaluateur.configs import QueryConfig


class Query(BaseModel):
    payer: str = Field(...)
    age: str = Field(...)
    complexity: str = Field(...)
    geography: str = Field(...)


async def main() -> None:
    evaluator = Evaluator(Query, context="Healthcare prior authorization")

    goals = GoalSpec(
        components=GoalLayer(items=[GoalItem(name="freshness checks")]),
        trajectories=GoalLayer(items=[GoalItem(name="conflict handling")]),
        outcomes=GoalLayer(items=[GoalItem(name="checklist-ready")]),
    )

    async for q in evaluator.run(
        query_config=QueryConfig(mode=QueryMode.INSTRUCTOR, goal_seed=0),
        instructions="Make the question sound like a real user.",
        goals=goals,
    ):
        print(q.metadata.goal_focus_area, "->", q.query)
        break


asyncio.run(main())
```

Structured goals:

```python
import asyncio
from pydantic import BaseModel, Field

from evaluateur import Evaluator, GoalItem, GoalLayer, GoalSpec, QueryMode
from evaluateur.configs import QueryConfig


class Query(BaseModel):
    payer: str = Field(..., description="insurance payer, like Cigna")
    age: str = Field(..., description="patient age category, like 'adult' or 'pediatric'")
    complexity: str = Field(..., description="complexity bucket, e.g. comorbidities, off-label")
    geography: str = Field(..., description="geography indicator, like state or zip code")


async def main() -> None:
    evaluator = Evaluator(Query, context="Healthcare prior authorization")

    goals = GoalSpec(
        components=GoalLayer(
            summary="Stress freshness, missing-document detection, and citation traceability.",
            items=[
                GoalItem(
                    name="freshness checks",
                    must_include=["effective date", "latest policy", "as of"],
                    avoid=["undated", "last year"],
                ),
                GoalItem(
                    name="grounded claims",
                    must_include=["cite", "policy section"],
                ),
            ],
        ),
        trajectories=GoalLayer(
            items=[
                GoalItem(
                    name="conflict integration",
                    must_include=["conflicting", "payer policy", "FDA label"],
                )
            ]
        ),
        outcomes=GoalLayer(
            items=[
                GoalItem(
                    name="checklist-ready",
                    must_include=["payer", "age", "diagnosis"],
                )
            ]
        ),
    )

    async for q in evaluator.run(
        query_config=QueryConfig(
            mode=QueryMode.INSTRUCTOR,
        ),
        goals=goals,
    ):
        print(q.metadata.query_goals.model_dump() if q.metadata.query_goals else None)
        break


asyncio.run(main())
```

Free-form goals (normalized with Instructor):

```python
import asyncio
from pydantic import BaseModel, Field

from evaluateur import Evaluator, QueryMode
from evaluateur.configs import QueryConfig


class Query(BaseModel):
    payer: str = Field(...)
    age: str = Field(...)
    complexity: str = Field(...)
    geography: str = Field(...)


async def main() -> None:
    evaluator = Evaluator(Query, context="Healthcare prior authorization")

    i = 0
    async for q in evaluator.run(
        query_config=QueryConfig(
            mode=QueryMode.INSTRUCTOR,
        ),
        goals="""
Components: prioritize freshness checks, grounded citations, and missing-source detection (don’t proceed silently).
Trajectories: include conflict handling and recovery behavior (re-try, switch tools, or escalate when evidence conflicts).
Outcomes: produce checklist-ready outputs that are easy to review and hard to misuse.
""",
    ):
        print(q.query)
        i += 1
        if i >= 3:
            break


asyncio.run(main())
```
