# Tuples

Tuple generation strategies and related types.

## Overview

Tuples are combinations of dimension values generated from options. Evaluateur supports multiple strategies for creating tuples.

```python
from evaluateur import TupleStrategy, build_tuple_generator
```

## TupleStrategy

Enum for tuple generation strategy selection.

```python
from evaluateur import TupleStrategy

# Available strategies
TupleStrategy.CROSS_PRODUCT  # Cartesian product with seeded sampling
TupleStrategy.AI             # LLM-generated tuples
```

| Strategy | Description | Best For |
|----------|-------------|----------|
| `CROSS_PRODUCT` | Cartesian product with diversity-maximizing sampling | Systematic coverage |
| `AI` | LLM generates coherent combinations | Realistic scenarios |

---

## Cross Product Strategy

The default strategy. Generates tuples from the Cartesian product of all options.

### How It Works

Given options:

```python
{
    "payer": ["Cigna", "Aetna"],
    "age": ["adult", "pediatric"],
}
```

The full cross product is:

```
("Cigna", "adult"), ("Cigna", "pediatric"),
("Aetna", "adult"), ("Aetna", "pediatric")
```

### Diversity-Maximizing Sampling

When `count < total_combinations`, uses Farthest Point Sampling (FPS) to select tuples that are maximally diverse across all dimensions:

```python
async for t in evaluator.tuples(
    options,
    strategy=TupleStrategy.CROSS_PRODUCT,
    count=50,      # Sample 50 from the full space
    seed=42,       # Reproducible results
):
    print(t.model_dump())
```

**How it works:**

1. Select a random first point (seeded for determinism)
2. For each subsequent selection, choose the candidate that differs from all already-selected tuples on the most dimensions
3. Distance is measured using Hamming distance (count of dimensions where values differ)

**Properties:**

- Reproducible with same seed
- Maximizes diversity: each sample differs from previous samples on as many dimensions as possible
- No replacement (each tuple appears at most once)
- Greedy algorithm with strong spread guarantees

### Example

```python
import asyncio
from pydantic import BaseModel, Field
from evaluateur import Evaluator, TupleStrategy


class Query(BaseModel):
    category: str = Field(..., description="content category")
    tone: str = Field(..., description="writing tone")
    length: str = Field(..., description="content length")


async def main() -> None:
    evaluator = Evaluator(Query)

    options = await evaluator.options(count_per_field=5)
    # 5 × 5 × 5 = 125 possible tuples

    async for t in evaluator.tuples(
        options,
        strategy=TupleStrategy.CROSS_PRODUCT,
        count=20,  # Sample 20 from 125
        seed=42,
    ):
        print(t.model_dump())


asyncio.run(main())
```

---

## AI Strategy

Asks the LLM to generate tuples directly, which can produce more coherent combinations.

### When to Use

- When dimension values have semantic relationships
- When you want "realistic" combinations
- When cross product would include nonsensical pairs

### Example

```python
async for t in evaluator.tuples(
    options,
    strategy=TupleStrategy.AI,
    count=20,
    instructions="Generate realistic patient scenarios",
):
    print(t.model_dump())
```

!!! note
    `AI` requires LLM calls and is slower than `CROSS_PRODUCT`.

---

## build_tuple_generator()

Factory function to create tuple generators directly.

```python
from evaluateur import TupleStrategy, build_tuple_generator
from evaluateur.client import resolve_client

def build_tuple_generator(
    client: LLMClient,
    strategy: TupleStrategy,
) -> TupleGenerator
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `client` | `LLMClient` | Internal client bundle (from `evaluateur.client`) for LLM-based strategies |
| `strategy` | `TupleStrategy` | Generation strategy |

**Returns:** A `TupleGenerator` instance.

**Example:**

```python
from evaluateur import TupleStrategy, build_tuple_generator
from evaluateur.client import resolve_client

client = resolve_client(llm="openai/gpt-4.1-mini")
generator = build_tuple_generator(client=client, strategy=TupleStrategy.CROSS_PRODUCT)

# Use directly
async for t in generator.generate(options, count=50, seed=42):
    print(t.model_dump())
```

---

## TupleGenerator Protocol

Interface for tuple generators.

```python
from evaluateur.tuples import TupleGenerator

class TupleGenerator(Protocol):
    async def generate(
        self,
        options: BaseModel,
        count: int,
        *,
        seed: int = 0,
        instructions: str | None = None,
    ) -> AsyncIterator[GeneratedTuple]:
        ...
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `options` | `BaseModel` | Options model with dimension values |
| `count` | `int` | Number of tuples to generate |
| `seed` | `int` | Random seed (for sampling strategies) |
| `instructions` | `str | None` | Instructions (for LLM strategies) |

---

## Built-in Generators

### CrossProductTupleGenerator

Generates tuples from the Cartesian product.

```python
from evaluateur.tuples import CrossProductTupleGenerator

generator = CrossProductTupleGenerator()

async for t in generator.generate(options, count=50, seed=42):
    print(t.model_dump())
```

### AITupleGenerator

Generates tuples using an LLM.

```python
from evaluateur.tuples import AITupleGenerator

generator = AITupleGenerator(client)

async for t in generator.generate(
    options,
    count=20,
    instructions="Create realistic combinations",
):
    print(t.model_dump())
```

---

## Sampling Behavior

### Full Enumeration

When `count >= total_combinations`:

```python
# All 8 tuples are returned (2 × 2 × 2 = 8)
async for t in evaluator.tuples(options, count=100):
    print(t.model_dump())
```

### Diversity Sampling

When `count < total_combinations`, Farthest Point Sampling ensures maximum spread:

```python
# Same seed = same 10 diverse tuples
async for t in evaluator.tuples(options, count=10, seed=42):
    print(t.model_dump())

# Different seed = different 10 diverse tuples (different starting point)
async for t in evaluator.tuples(options, count=10, seed=43):
    print(t.model_dump())
```

Unlike uniform random sampling, consecutive samples will differ on multiple dimensions rather than potentially differing on just one.

### Reproducibility

```python
# These produce identical results
run1 = [t async for t in evaluator.tuples(options, count=20, seed=42)]
run2 = [t async for t in evaluator.tuples(options, count=20, seed=42)]
assert run1 == run2
```

---

## Complete Example

```python
import asyncio
from pydantic import BaseModel, Field
from evaluateur import Evaluator, TupleStrategy


class CustomerScenario(BaseModel):
    industry: str = Field(..., description="business industry")
    company_size: str = Field(..., description="company size")
    use_case: str = Field(..., description="primary use case")
    urgency: str = Field(..., description="urgency level")


async def main() -> None:
    evaluator = Evaluator(CustomerScenario)

    # Generate options
    options = await evaluator.options(
        instructions="Focus on B2B software scenarios",
        count_per_field=6,
    )

    # 6^4 = 1296 possible combinations
    print(f"Total space: {6**4} combinations")

    # Sample 50 with cross product
    print("\nCross product sample:")
    async for t in evaluator.tuples(
        options,
        strategy=TupleStrategy.CROSS_PRODUCT,
        count=50,
        seed=42,
    ):
        print(f"  {t.model_dump()}")

    # Or use LLM for coherent combinations
    print("\nLLM-generated tuples:")
    async for t in evaluator.tuples(
        options,
        strategy=TupleStrategy.AI,
        count=10,
        instructions="Create realistic enterprise scenarios",
    ):
        print(f"  {t.model_dump()}")


asyncio.run(main())
```

## See Also

- [Dimensions, Tuples, Queries](../concepts/dimensions-tuples-queries.md) - Conceptual overview
- [Evaluator](evaluator.md) - Using tuples with the evaluator
- [Queries](queries.md) - Converting tuples to queries
