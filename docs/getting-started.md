# Getting Started

This guide covers installation, environment setup, and running your first evaluation.

## Installation

Evaluateur requires Python 3.10 or higher.

=== "uv (recommended)"

    ```bash
    uv add evaluateur
    ```

=== "pip"

    ```bash
    pip install evaluateur
    ```

=== "pipx (for CLI tools)"

    ```bash
    pipx install evaluateur
    ```

## Environment Setup

Evaluateur uses environment variables for LLM provider configuration. The simplest setup uses OpenAI.

### OpenAI (Default)

Create a `.env` file in your project root:

```bash
OPENAI_API_KEY=sk-your-api-key-here
```

Or export directly:

```bash
export OPENAI_API_KEY=sk-your-api-key-here
```

### Model Selection

By default, Evaluateur uses `openai/gpt-4.1-mini`. Override with the `EVALUATEUR_MODEL` environment variable:

```bash
export EVALUATEUR_MODEL=anthropic/claude-4-5-haiku-latest
```

Or configure programmatically:

```python
from evaluateur import Evaluator

evaluator = Evaluator(MyModel, llm="anthropic/claude-4-5-haiku-latest")
```

## Your First Evaluation

Here's a complete example that generates synthetic queries for a healthcare prior authorization system:

```python
import asyncio
from pydantic import BaseModel, Field
from evaluateur import Evaluator, TupleStrategy


class PriorAuthQuery(BaseModel):
    """Dimensions for prior authorization queries."""

    payer: str = Field(..., description="Insurance payer (e.g., Cigna, Aetna)")
    age_group: str = Field(..., description="Patient age category")
    procedure_type: str = Field(..., description="Type of medical procedure")
    urgency: str = Field(..., description="Urgency level of the request")


async def main() -> None:
    # Create evaluator with your dimension model
    evaluator = Evaluator(PriorAuthQuery)

    # Step 1: Generate diverse options for each dimension
    options = await evaluator.options(
        instructions="Include common US payers and varied clinical scenarios.",
        count_per_field=5,
    )

    print("Generated options:")
    print(options.model_dump())

    # Step 2: Generate queries from option combinations
    print("\nGenerated queries:")
    async for query in evaluator.run(
        options=options,
        tuple_strategy=TupleStrategy.CROSS_PRODUCT,
        tuple_count=10,
        seed=42,
        instructions="Write realistic patient questions about prior authorization.",
    ):
        print(f"  {query.query}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Understanding the Output

Each generated query includes:

- **`query`**: The natural language query text
- **`source_tuple`**: The dimension values used to generate this query
- **`metadata`**: Additional information like goal focus area (when using goals)

```python
async for q in evaluator.run(...):
    print(f"Query: {q.query}")
    print(f"From tuple: {q.source_tuple.values}")
    print(f"Metadata: {q.metadata.model_dump()}")
```

## Fixed Options

If your model already has specific values you want to use, define them as lists:

```python
class Query(BaseModel):
    # Fixed options - won't be modified by options generation
    payer: list[str] = ["Cigna", "Aetna", "UnitedHealthcare"]

    # Dynamic options - will be generated
    age_group: str = Field(..., description="Patient age category")
```

The `options()` method preserves list fields and only generates options for scalar fields.

## Next Steps

- [Dimensions, Tuples, Queries](concepts/dimensions-tuples-queries.md) - Understand the core concepts
- [Goal-Guided Optimization](concepts/goal-guided-optimization.md) - Shape queries with goals
- [Provider Configuration](guides/provider-configuration.md) - Use different LLM providers
