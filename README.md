## Evaluateur

Synthetic evaluation helper for LLM applications, built around the
**dimensions → tuples → queries** flow described in [Hamel Husain's FAQ](https://hamel.dev/blog/posts/evals-faq/what-is-the-best-approach-for-generating-synthetic-data.html).

### Installation

The project is packaged as a normal Python library. With `uv`:

```bash
uv add evaluateur
```

DSPy support is optional. If you want to use `QueryMode.DSPY` or DSPy refinement in
`QueryMode.HYBRID`, install the extra:

```bash
uv add "evaluateur[dspy]"
```

### Basic usage

Define a Pydantic model that represents the dimensions of your evaluation
space, then use the `Evaluator` to generate options and queries:

```python
import asyncio
from pydantic import BaseModel, Field

from evaluateur import Evaluator, QueryConfig, QueryMode, TupleConfig, TupleStrategy


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

    # Step 2: turn options into tuples and natural language queries
    output = await evaluator.run(
        options=options,
        tuple_config=TupleConfig(strategy=TupleStrategy.CROSS_PRODUCT, count=50, seed=0),
        query_config=QueryConfig(mode=QueryMode.HYBRID),
    )

    for q in output.queries:
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

### Tuple generation: seeded sampling for cross product

When `TupleStrategy.CROSS_PRODUCT` is used and `0 < count < total_combinations`,
Evaluateur returns a **seeded randomized sample** of the cartesian product
(*uniform without replacement*). This helps avoid always taking the “first N”
combinations when the space is large.

- To get reproducible results, set `TupleConfig(seed=...)`.
- Changing the seed gives you a different randomized subset.

### Goal-guided query optimization (Components / Trajectories / Outcomes)

You can guide query generation using the three-layer framework by providing a
`GoalSpec` (structured) or free-form text (which is normalized into a `GoalSpec`).

The goals are used in two ways:

- Queries are **conditioned per run** on your goals (so you can iterate quickly).
- In DSPy/HYBRID modes you can also enable **compile-time optimization** with a
  goal-aware DSPy optimizer.

#### Which DSPy optimizer is used?

When compile-time optimization is enabled, Evaluateur uses **GEPA by default** (it tends to outperform MiProV2, but can cost more because it does reflective optimization).

- **GEPA**: best quality in many cases, but usually slower and more expensive.
- **MiProV2**: often faster/cheaper, and a good baseline when you want quick iteration.

To force MiProV2:

```python
QueryConfig(
    mode=QueryMode.DSPY,
    dspy=DSpyConfig(
        optimize=True,
        optimizer_name="miprov2",
        trainset=[...],  # required for optimization
        # valset=[...],  # optional
    ),
)
```

Note: compile-time optimization only runs when you provide a `trainset` and/or `valset`. If you omit both, Evaluateur will skip compilation and just run the base DSPy module.

Structured goals:

```python
import asyncio
from pydantic import BaseModel, Field

from evaluateur import DSpyConfig, Evaluator, GoalItem, GoalLayer, GoalSpec, JudgeBackend, QueryConfig, QueryMode


class Query(BaseModel):
    payer: str = Field(..., description="insurance payer, like Cigna")
    age: str = Field(..., description="patient age category, like 'adult' or 'pediatric'")
    complexity: str = Field(..., description="complexity bucket, e.g. comorbidities, off-label")
    geography: str = Field(..., description="geography indicator, like state or zip code")


async def main() -> None:
    evaluator = Evaluator(Query, context="Healthcare prior authorization")

    goals = GoalSpec(
        title="PA letter search failures",
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

    output = await evaluator.run(
        query_config=QueryConfig(
            mode=QueryMode.DSPY,
            dspy=DSpyConfig(
                goal_guided=True,
                judge_backend=JudgeBackend.LLM,  # use HEURISTIC for offline tests
            ),
        ),
        goals=goals,
    )

    print(output.metadata.get("query_goals"))


asyncio.run(main())
```

Free-form goals (normalized with Instructor):

```python
import asyncio
from pydantic import BaseModel, Field

from evaluateur import DSpyConfig, Evaluator, QueryConfig, QueryMode


class Query(BaseModel):
    payer: str = Field(...)
    age: str = Field(...)
    complexity: str = Field(...)
    geography: str = Field(...)


async def main() -> None:
    evaluator = Evaluator(Query, context="Healthcare prior authorization")

    output = await evaluator.run(
        query_config=QueryConfig(
            mode=QueryMode.DSPY,
            dspy=DSpyConfig(goal_guided=True),
        ),
        goals="""
Components: force freshness (effective date, latest policy) and grounded citations.
Trajectories: include conflicting evidence and require resolving or escalating.
Outcomes: short, checklist-friendly queries that reveal missing inputs.
""",
    )

    for q in output.queries[:3]:
        print(q.query)


asyncio.run(main())
```
