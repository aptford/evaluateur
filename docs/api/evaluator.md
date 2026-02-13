# Evaluator

The main entry point for synthetic evaluation generation.

## Overview

The `Evaluator` class orchestrates the dimensions → tuples → queries pipeline. It's parameterized by a Pydantic model that describes the dimensions of your evaluation space.

```python
from pydantic import BaseModel, Field
from evaluateur import Evaluator


class Query(BaseModel):
    topic: str = Field(..., description="subject area")
    difficulty: str = Field(..., description="complexity level")


evaluator = Evaluator(Query)
```

## Class Reference

::: evaluateur.Evaluator
    options:
      show_source: true
      members:
        - __init__
        - options
        - tuples
        - queries
        - run

## Constructor

### `__init__(model, *, llm=None, client=None, model_name=None, config=None)`

Create an evaluator for the given dimension model.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `model` | `Type[BaseModel]` | Pydantic model defining evaluation dimensions |
| `llm` | `str | None` | A `"provider/model-name"` string (e.g. `"openai/gpt-4.1-mini"`). Mutually exclusive with `client`. Defaults to `EVALUATEUR_MODEL` env var. |
| `client` | `Any | None` | Pre-configured async Instructor client. Must be paired with `model_name`. Mutually exclusive with `llm`. |
| `model_name` | `str | None` | Model name for `chat.completions.create()`. Required when `client` is provided. |
| `config` | `EvaluatorConfig | None` | Configuration for default values. If `None`, uses `DEFAULT_CONFIG` |

**Example:**

```python
from evaluateur import Evaluator, EvaluatorConfig

# Default (reads EVALUATEUR_MODEL env var)
evaluator = Evaluator(Query)

# Explicit model
evaluator = Evaluator(Query, llm="openai/gpt-4.1-mini")

# Switch providers
evaluator = Evaluator(Query, llm="anthropic/claude-3-5-sonnet-latest")

# Advanced: bring your own Instructor client
import instructor
from openai import AsyncOpenAI
inst = instructor.from_openai(AsyncOpenAI())
evaluator = Evaluator(Query, client=inst, model_name="gpt-4o")

# Custom config with different defaults
config = EvaluatorConfig(
    instructions="Focus on US healthcare scenarios.",
    tuples_count=50,
    options_count_per_field=10,
    goal_mode="full",
)
evaluator = Evaluator(Query, llm="openai/gpt-4o", config=config)
```

## EvaluatorConfig

Configuration class for setting default parameter values.

```python
from evaluateur import EvaluatorConfig, DEFAULT_CONFIG

# View default values
print(DEFAULT_CONFIG.tuples_count)  # 20
print(DEFAULT_CONFIG.options_count_per_field)  # 5

# Create custom config
config = EvaluatorConfig(
    instructions="Focus on edge cases in US healthcare.",
    options_count_per_field=10,
    tuples_count=50,
    tuples_seed=42,
    tuples_strategy="cross_product",
    goal_mode="sample",
    query_mode="instructor",
)
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `instructions` | `str | None` | `None` | Default instructions shared across options, tuples, and queries. Method-level `instructions` override this value. |
| `options_count_per_field` | `int` | `5` | Default options per field |
| `tuples_count` | `int` | `20` | Default number of tuples |
| `tuples_seed` | `int` | `0` | Default random seed |
| `tuples_strategy` | `str` | `"cross_product"` | Default tuple strategy |
| `goal_mode` | `str` | `"sample"` | Default goal mode |
| `query_mode` | `str` | `"instructor"` | Default query mode |

## Methods

### `options()`

Generate option values for each dimension field.

```python
async def options(
    self,
    *,
    instructions: str | None = None,
    count_per_field: int | None = None,
) -> BaseModel
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `instructions` | `str | None` | config | Instructions for option generation (defaults to config) |
| `count_per_field` | `int | None` | config | Number of options per scalar field (defaults to config) |

**Returns:** A dynamically created Pydantic model instance where each scalar field is converted to a list of options.

**Example:**

```python
options = await evaluator.options(
    instructions="Focus on edge cases",
    count_per_field=10,
)
print(options.topic)  # ["AI", "Healthcare", "Finance", ...]
```

**Notes:**

- Scalar fields (`str`, `int`, `float`) are converted to lists
- Iterator fields (`list`, `tuple`) are preserved as-is
- The returned model has the same field names as the input model

---

### `tuples()`

Generate dimension value combinations as an async iterator.

```python
async def tuples(
    self,
    options: BaseModel,
    *,
    strategy: TupleStrategy | None = None,
    count: int | None = None,
    seed: int | None = None,
    instructions: str | None = None,
) -> AsyncIterator[GeneratedTuple]
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `options` | `BaseModel` | - | Options model from `options()` |
| `strategy` | `TupleStrategy | None` | config | Tuple generation strategy (defaults to config) |
| `count` | `int | None` | config | Number of tuples to generate (defaults to config) |
| `seed` | `int | None` | config | Random seed for sampling (defaults to config) |
| `instructions` | `str | None` | config | Instructions for LLM-based strategies (defaults to config) |

**Yields:** `GeneratedTuple` objects containing dimension value combinations.

**Example:**

```python
async for t in evaluator.tuples(
    options,
    strategy=TupleStrategy.CROSS_PRODUCT,
    count=50,
    seed=42,
):
    print(t.values)  # {"topic": "AI", "difficulty": "hard"}
```

---

### `queries()`

Generate natural language queries from tuples.

```python
async def queries(
    self,
    *,
    tuples: Sequence[GeneratedTuple] | AsyncIterator[GeneratedTuple],
    instructions: str | None = None,
    goal_mode: GoalMode | None = None,
    query_mode: QueryMode | None = None,
    seed: int | None = None,
    goals: GoalSpec | str | None = None,
) -> AsyncIterator[GeneratedQuery]
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tuples` | `Sequence | AsyncIterator` | - | Tuples to convert to queries |
| `instructions` | `str | None` | config | Query generation instructions (defaults to config) |
| `goal_mode` | `GoalMode | None` | config | Goal guidance mode (defaults to config) |
| `query_mode` | `QueryMode | None` | config | Query generator to use (defaults to config) |
| `seed` | `int | None` | config | Seed for goal sampling (defaults to config) |
| `goals` | `GoalSpec | str | None` | `None` | Goal specification |

**Yields:** `GeneratedQuery` objects with the query text and metadata.

**Example:**

```python
tuples = [GeneratedTuple(values={"topic": "AI", "difficulty": "easy"})]

async for q in evaluator.queries(
    tuples=tuples,
    instructions="Write as a curious student",
    goals="Focus on practical applications",
):
    print(q.query)
```

---

### `run()`

Convenience method that runs the full pipeline: options → tuples → queries.

```python
async def run(
    self,
    *,
    options: BaseModel | None = None,
    instructions: str | None = None,
    count_per_field: int | None = None,
    tuple_strategy: TupleStrategy | None = None,
    tuple_count: int | None = None,
    seed: int | None = None,
    goal_mode: GoalMode | None = None,
    query_mode: QueryMode | None = None,
    goals: GoalSpec | str | None = None,
) -> AsyncIterator[GeneratedQuery]
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `options` | `BaseModel | None` | `None` | Pre-generated options (skips generation if provided) |
| `instructions` | `str | None` | config | Shared instructions for all stages (defaults to config) |
| `count_per_field` | `int | None` | config | Options per field (defaults to config) |
| `tuple_strategy` | `TupleStrategy | None` | config | Tuple generation strategy (defaults to config) |
| `tuple_count` | `int | None` | config | Number of tuples (defaults to config) |
| `seed` | `int | None` | config | Random seed (defaults to config) |
| `goal_mode` | `GoalMode | None` | config | Goal guidance mode (defaults to config) |
| `query_mode` | `QueryMode | None` | config | Query generator (defaults to config) |
| `goals` | `GoalSpec | str | None` | `None` | Goal specification |

**Yields:** `GeneratedQuery` objects.

**Example:**

```python
async for q in evaluator.run(
    instructions="Generate diverse questions",
    tuple_count=100,
    seed=42,
    goals="Test edge cases and error handling",
):
    print(q.query)
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `Type[BaseModel]` | The dimension model |
| `config` | `EvaluatorConfig` | Configuration with default values |

## See Also

- [Provider Configuration](../guides/provider-configuration.md) - Provider setup
- [Goals](goals.md) - Goal specification
- [Queries](queries.md) - Query data models
- [Tuples](tuples.md) - Tuple strategies
