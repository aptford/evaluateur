# Evaluateur

Synthetic evaluation helper for LLM applications, built around the **dimensions → tuples → queries** flow described in [Hamel Husain's FAQ](https://hamel.dev/blog/posts/evals-faq/what-is-the-best-approach-for-generating-synthetic-data.html).

## What is Evaluateur?

Evaluateur helps you generate diverse, realistic test queries for evaluating LLM systems. Instead of manually writing test cases, you define the **dimensions** of your evaluation space (like payer, age, complexity) and let the library generate meaningful combinations.

The library follows a simple three-step flow:

1. **Dimensions → Options**: Define what varies in your queries and generate diverse values
2. **Options → Tuples**: Create combinations of dimension values
3. **Tuples → Queries**: Convert combinations into natural language queries

## Quick Install

=== "uv"

    ```bash
    uv add evaluateur
    ```

=== "pip"

    ```bash
    pip install evaluateur
    ```

## Quick Start

```python
import asyncio
from pydantic import BaseModel, Field
from evaluateur import Evaluator, TupleStrategy


class Query(BaseModel):
    payer: str = Field(..., description="insurance payer, like Cigna")
    age: str = Field(..., description="patient age category")
    complexity: str = Field(..., description="query complexity level")
    geography: str = Field(..., description="geographic region")


async def main() -> None:
    evaluator = Evaluator(Query)

    # Generate options for each dimension
    options = await evaluator.options(
        instructions="Focus on common US payers and edge-case scenarios.",
        count_per_field=5,
    )

    # Stream tuples as natural language queries
    async for q in evaluator.run(
        options=options,
        tuple_strategy=TupleStrategy.CROSS_PRODUCT,
        tuple_count=50,
        seed=0,
        instructions="Write realistic user questions. Keep them short.",
    ):
        print(q.source_tuple.values, "->", q.query)


asyncio.run(main())
```

## Key Features

- **Pydantic-based**: Define dimensions using familiar Pydantic models
- **Async-first**: All operations use async iterators for efficient streaming
- **Goal-guided generation**: Shape queries using the Components/Trajectories/Outcomes framework
- **Seeded sampling**: Reproducible results with configurable random seeds
- **Provider-agnostic**: Works with any LLM provider supported by Instructor

## Next Steps

- [Getting Started](getting-started.md) - Installation and environment setup
- [Concepts](concepts/dimensions-tuples-queries.md) - Understand the core workflow
- [Basic Usage](guides/basic-usage.md) - Complete working examples
- [API Reference](api/evaluator.md) - Full API documentation
