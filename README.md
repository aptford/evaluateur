# Evaluateur

[![PyPI](https://img.shields.io/pypi/v/evaluateur)](https://pypi.org/project/evaluateur/)
[![Python](https://img.shields.io/pypi/pyversions/evaluateur)](https://pypi.org/project/evaluateur/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Generate diverse, realistic test queries for LLM applications. Define your
evaluation space as dimensions, sample combinations, and convert them to
natural language -- with optional goal-guided optimization to target specific
failure modes.

## Why

Evaluations require test data. Early on, you don't have any.

If you ask an LLM to "generate 50 test queries," you get repetitive inputs.
The model gravitates toward the same phrasing, the same scenarios, the same
level of complexity. Manual test cases fare no better: they reflect what
the author thought to test, not what actually breaks.

Evaluateur solves this with structure. You define **dimensions** -- the axes
along which your system's behavior varies -- and the library generates
combinations that cover the space systematically, including edge cases that
neither a human nor an LLM would produce on its own.

The approach follows the **dimensions → tuples → queries** pattern described
in [Hamel Husain's evaluation FAQ](https://hamel.dev/blog/posts/evals-faq/what-is-the-best-approach-for-generating-synthetic-data.html).

## How it works

```
Dimensions        Options          Tuples              Queries
                                   (combinations)      (natural language)
┌──────────┐    ┌────────────┐    ┌───────────────┐    ┌──────────────────────┐
│ payer     │───▶│ Cigna      │    │ Cigna, adult, │    │ "Does Cigna cover    │
│ age       │    │ Aetna      │───▶│ off-label, TX │───▶│  off-label Dupixent  │
│ complexity│    │ BCBS       │    │               │    │  for adults in TX?"  │
│ geography │    │ ...        │    │ ...           │    │ ...                  │
└──────────┘    └────────────┘    └───────────────┘    └──────────────────────┘
```

1. **Dimensions → Options.** Define a Pydantic model with the axes of variation.
   The LLM generates diverse values for each field.
2. **Options → Tuples.** Sample combinations. The default cross-product strategy
   uses Farthest Point Sampling to maximize diversity across dimensions. An
   AI strategy is also available for semantically coherent combinations.
3. **Tuples → Queries.** Each combination is converted into a natural language
   query, ready to feed to your agent.

## Installation

```bash
uv add evaluateur
```

or with pip:

```bash
pip install evaluateur
```

## Quick start

```python
import asyncio
from pydantic import BaseModel, Field

from evaluateur import Evaluator, TupleStrategy


class Query(BaseModel):
    payer: str = Field(..., description="insurance payer, like Cigna")
    age: str = Field(..., description="patient age category, like 'adult' or 'pediatric'")
    complexity: str = Field(
        ...,
        description="query complexity, like 'off-label', 'comorbidities', etc",
    )
    geography: str = Field(..., description="geography indicator, like a state or zip code")


async def main() -> None:
    evaluator = Evaluator(Query)

    async for q in evaluator.run(
        tuple_strategy=TupleStrategy.CROSS_PRODUCT,
        tuple_count=50,
        seed=0,
        instructions="Focus on common US payers and edge-case clinical scenarios.",
    ):
        print(q.source_tuple.model_dump(), "->", q.query)


asyncio.run(main())
```

The `run()` method handles the full pipeline: generating options, sampling
tuples, and converting each tuple to a natural language query.

For step-by-step control, call `evaluator.options()`, `evaluator.tuples()`,
and `evaluator.queries()` separately.

## Goal-guided optimization

The first batch of queries gives you a baseline. After running them through
your agent and analyzing the failures, you can feed those observations back
as **goals** to bias the next round of query generation toward specific
failure modes.

Goals can be categorized using the **CTO framework**:

- **Components** -- system internals: retrieval freshness, citation accuracy, tool reliability.
- **Trajectories** -- decision sequences: tool selection order, conflict resolution, retry behavior.
- **Outcomes** -- what the user sees: output format, actionability, appropriate uncertainty.

Pass goals as free-form text. Structured lists with `Components:`,
`Trajectories:`, and `Outcomes:` headers are parsed directly without an
LLM call:

```python
import asyncio
from pydantic import BaseModel, Field

from evaluateur import Evaluator


class Query(BaseModel):
    payer: str = Field(..., description="insurance payer, like Cigna")
    age: str = Field(..., description="patient age category")
    complexity: str = Field(..., description="query complexity, like 'off-label'")
    geography: str = Field(..., description="geography indicator, like a state")


async def main() -> None:
    evaluator = Evaluator(Query)

    async for q in evaluator.run(
        seed=0,
        goals="""
Components:
- The system must cite current policy versions; stale guidelines are a compliance risk
- Every clinical claim needs a traceable source from retrieved documents

Trajectories:
- Prefer formulary API over generic web search for drug lists
- Surface conflicts between sources instead of silently picking one

Outcomes:
- Produce structured checklists that reviewers can sign off on
- Flag uncertainty instead of guessing
""",
        instructions="Write realistic questions from a doctor's perspective.",
    ):
        print(f"[{q.metadata.goal_focus}] {q.query}")


asyncio.run(main())
```

Each generated query targets a single goal by default (cycling through them),
so one run produces a mix of stress-test styles. You can also pass goals
as a `GoalSpec` with structured `Goal` objects for programmatic control.

See the [custom goals guide](https://evaluateur.aptford.com/guides/custom-goals/)
for goal modes (`sample`, `cycle`, `full`) and advanced usage.

## The iteration loop

The core workflow is a feedback loop:

1. **Generate** queries across your dimensions.
2. **Run** them through your agent and collect traces.
3. **Analyze** failures -- write freeform notes about what went wrong.
4. **Turn notes into goals** -- group observations into Components, Trajectories, and Outcomes.
5. **Generate again** with those goals to stress-test the failure modes you found.

Each cycle tightens coverage. The first round catches obvious failures. By the
third, you're stress-testing edge cases that real traffic won't hit for months.
When production traffic arrives, feed those traces back into the loop.

## Features

- **Pydantic-based dimensions.** Define your evaluation space with standard Pydantic models. Field descriptions guide option generation.
- **Farthest Point Sampling.** When sampling from the cross product, tuples are selected to maximize Hamming distance, ensuring broad coverage instead of clustered combinations.
- **Seeded, reproducible sampling.** Set `seed=` to get deterministic results. Change the seed for a different subset.
- **Goal-guided generation.** Bias queries toward specific failure modes using the CTO framework or custom categories.
- **Async streaming.** All generators yield results as async iterators for memory-efficient processing.
- **Provider-agnostic.** Works with any LLM provider supported by [Instructor](https://python.useinstructor.com/) -- OpenAI, Anthropic, and others.
- **Traceability.** Every generated query links back to its source tuple via `q.source_tuple`, making it easy to understand why a query was generated.
- **Mixed options.** Fixed lists (`state: list[str] = ["CA", "NY", "TX"]`) coexist with LLM-generated options in the same model.

## Configuration

By default, evaluateur reads the `EVALUATEUR_MODEL` environment variable
(defaults to `openai/gpt-4.1-mini`). You can override this per evaluator:

```python
from evaluateur import Evaluator

evaluator = Evaluator(Query, llm="anthropic/claude-haiku-4-5")
```

For advanced setups (observability wrappers, custom providers), pass a
pre-configured Instructor client directly:

```python
import instructor
from openai import AsyncOpenAI

from evaluateur import Evaluator

client = instructor.from_openai(AsyncOpenAI())
evaluator = Evaluator(Query, client=client, model_name="gpt-4.1-mini")
```

See the [provider configuration guide](https://evaluateur.aptford.com/guides/provider-configuration/)
for details.

## Documentation

Full documentation is available at [evaluateur.aptford.com](https://evaluateur.aptford.com).

- [Getting started](https://evaluateur.aptford.com/getting-started/) -- installation and environment setup
- [Dimensions, tuples, and queries](https://evaluateur.aptford.com/concepts/dimensions-tuples-queries/) -- core concepts
- [Goal-guided optimization](https://evaluateur.aptford.com/concepts/goal-guided-optimization/) -- the CTO framework
- [Walkthrough notebook](https://evaluateur.aptford.com/guides/walkthrough/) -- end-to-end example
- [API reference](https://evaluateur.aptford.com/api/evaluator/) -- full API docs
