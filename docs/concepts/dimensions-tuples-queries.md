# Dimensions, Tuples, and Queries

Evaluateur implements the **dimensions → tuples → queries** workflow for generating synthetic evaluation data. This approach, described in [Hamel Husain's FAQ on synthetic data](https://hamel.dev/blog/posts/evals-faq/what-is-the-best-approach-for-generating-synthetic-data.html), provides systematic coverage of your evaluation space.

## The Three-Step Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Dimensions │ ──▶ │   Tuples    │ ──▶ │   Queries   │
│  (Options)  │     │ (Combos)    │     │ (NL Text)   │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Step 1: Dimensions → Options

A **dimension** is an axis of variation in your evaluation space. For a healthcare chatbot, dimensions might include:

- Payer (Cigna, Aetna, UnitedHealthcare)
- Age group (pediatric, adult, geriatric)
- Procedure type (routine, surgical, specialty)
- Geographic region (urban, rural, specific states)

You define dimensions using a Pydantic model:

```python
from pydantic import BaseModel, Field


class Query(BaseModel):
    payer: str = Field(..., description="insurance payer")
    age_group: str = Field(..., description="patient age category")
    procedure_type: str = Field(..., description="type of medical procedure")
    geography: str = Field(..., description="geographic region")
```

The `options()` method generates diverse values for each dimension:

```python
evaluator = Evaluator(Query)
options = await evaluator.options(
    instructions="Focus on US healthcare scenarios",
    count_per_field=5,
)
# options.payer might be ["Cigna", "Aetna", "UnitedHealthcare", "Blue Cross", "Humana"]
# options.age_group might be ["pediatric", "young adult", "middle-aged", "senior", "elderly"]
```

### Step 2: Options → Tuples

A **tuple** is a specific combination of dimension values. From the options above, one tuple might be:

```python
{
    "payer": "Cigna",
    "age_group": "pediatric",
    "procedure_type": "specialty",
    "geography": "Texas"
}
```

Evaluateur supports two tuple generation strategies:

#### Cross Product (Default)

Generates combinations from the Cartesian product of all options. With 5 options per field and 4 fields, you have 5⁴ = 625 possible tuples.

```python
async for t in evaluator.tuples(
    options,
    strategy=TupleStrategy.CROSS_PRODUCT,
    count=50,
    seed=42,
):
    print(t.values)
```

When `count` is less than total combinations, Evaluateur uses Farthest Point Sampling (FPS) to select a maximally diverse subset. Each selected tuple differs from all previously selected tuples on as many dimensions as possible, ensuring broad coverage of the evaluation space.

#### Direct LLM

Asks the LLM to generate tuples directly, which can produce more coherent combinations:

```python
async for t in evaluator.tuples(
    options,
    strategy=TupleStrategy.DIRECT_LLM,
    count=50,
    instructions="Generate realistic patient scenarios",
):
    print(t.values)
```

### Step 3: Tuples → Queries

Each tuple is converted into a natural language query using the configured query generator:

```python
tuples = [...]  # from step 2

async for q in evaluator.queries(
    tuples=tuples,
    instructions="Write questions a patient might ask about prior authorization",
):
    print(q.query)
    # "I'm a pediatric patient with Cigna in Texas. What's the prior auth
    #  process for specialty procedures?"
```

## The Full Pipeline

The `run()` method combines all three steps:

```python
async for q in evaluator.run(
    instructions="Generate realistic healthcare questions",
    tuple_strategy=TupleStrategy.CROSS_PRODUCT,
    tuple_count=100,
    count_per_field=5,
    seed=42,
):
    print(q.query)
```

## Why This Approach?

### Systematic Coverage

By defining dimensions explicitly, you ensure coverage of important combinations that might be missed when writing test cases manually.

### Maximum Diversity

When sampling a subset, Evaluateur uses Farthest Point Sampling to maximize diversity. This ensures that sampled tuples differ from each other on as many dimensions as possible, rather than clustering around similar combinations.

For example, with healthcare scenarios, you avoid getting multiple samples that only differ by age while keeping payer, indication, and state the same. Instead, each sample explores a different region of the evaluation space.

### Reproducibility

Seeded sampling means you can regenerate the same test set. Change the seed to get a different subset.

### Scalability

The space of 625 combinations from 4 dimensions × 5 options each is easy to sample from. Add more dimensions or options to increase diversity without manual effort.

### Traceability

Each query links back to its source tuple, making it easy to understand why a particular query was generated:

```python
async for q in evaluator.run(...):
    print(f"Query: {q.query}")
    print(f"Generated from: {q.source_tuple.values}")
```

## Instructions at Each Stage

You can provide different instructions at each stage of the pipeline:

```python
# Guide option generation
options = await evaluator.options(
    instructions="Focus on edge cases and unusual scenarios",
    count_per_field=10,
)

# Guide tuple selection (for DIRECT_LLM strategy)
tuples = evaluator.tuples(
    options,
    instructions="Prefer combinations that stress-test the system",
)

# Guide query phrasing
queries = evaluator.queries(
    tuples=tuples,
    instructions="Write questions in casual, conversational English",
)
```

The `run()` method shares the same `instructions` across all stages, but you can call each method separately for fine-grained control.
