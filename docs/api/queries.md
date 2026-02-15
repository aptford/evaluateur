# Queries

Data models for generated queries and related types.

## Overview

The queries module contains models for representing generated queries, their source tuples, and associated metadata.

```python
from evaluateur import GeneratedQuery, GeneratedTuple
from evaluateur.queries import QueryMetadata
```

## GeneratedQuery

A natural language query with full traceability.

::: evaluateur.GeneratedQuery
    options:
      show_source: true

### Constructor

```python
GeneratedQuery(
    query: str,
    source_tuple: GeneratedTuple,
    metadata: QueryMetadata = QueryMetadata(),
)
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `query` | `str` | The generated natural language query |
| `source_tuple` | `GeneratedTuple` | The tuple that produced this query |
| `metadata` | `QueryMetadata` | Associated metadata |

**Example:**

```python
from evaluateur import GeneratedQuery, GeneratedTuple
from evaluateur.queries import QueryMetadata

query = GeneratedQuery(
    query="What's the prior auth process for specialty procedures?",
    source_tuple=GeneratedTuple({
        "payer": "Cigna",
        "procedure": "specialty",
    }),
    metadata=QueryMetadata(goal_guided=True),
)

print(query.query)
print(query.source_tuple.model_dump())
print(query.metadata.goal_guided)
```

---

## GeneratedTuple

A concrete combination of dimension values. Implemented as a Pydantic RootModel with dict-like access.

::: evaluateur.GeneratedTuple
    options:
      show_source: true

### Constructor

```python
GeneratedTuple(dict[str, ScalarValue])
```

**Type:** `ScalarValue = str | int | float | bool`

**Dict-like access:** supports `t["key"]`, `t.get("key", default)`, `t.items()`, `t.keys()`, `"key" in t`, `len(t)`, `bool(t)`.

**Example:**

```python
from evaluateur import GeneratedTuple

t = GeneratedTuple({
    "payer": "Cigna",
    "age_group": "adult",
    "complexity": "high",
    "geography": "Texas",
})

print(t["payer"])  # "Cigna"
print(t.model_dump())  # full dict
```

---

## QueryMetadata

Metadata associated with a generated query.

::: evaluateur.queries.QueryMetadata
    options:
      show_source: true

### Constructor

```python
QueryMetadata(
    goal_guided: bool = False,
    query_goals: GoalSpec | None = None,
    goal_mode: GoalMode | None = None,
    goal_focus: str | None = None,
    goal_category: str | None = None,
    **extra_fields,  # Extra fields allowed
)
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `goal_guided` | `bool` | `False` | Whether goals were used |
| `query_goals` | `GoalSpec \| None` | `None` | The goal spec used |
| `goal_mode` | `GoalMode \| None` | `None` | `"sample"`, `"cycle"`, or `"full"` |
| `goal_focus` | `str \| None` | `None` | Name of the focused goal (sample/cycle mode) |
| `goal_category` | `str \| None` | `None` | Category of the focused goal |

Extra fields are allowed for custom metadata.

**Example:**

```python
from evaluateur.queries import QueryMetadata

meta = QueryMetadata(
    goal_guided=True,
    goal_mode="sample",
    goal_focus="freshness checks",
    goal_category="components",
    custom_field="custom_value",  # Extra fields allowed
)

print(meta.goal_guided)    # True
print(meta.goal_focus)     # "freshness checks"
print(meta.goal_category)  # "components"
print(meta.model_dump())   # Includes custom_field
```

---

## merge_query_metadata()

Standalone function for merging query metadata.

```python
from evaluateur.queries import merge_query_metadata

merged = merge_query_metadata(
    run_metadata=run_meta,
    per_query_metadata=query_meta,
)
```

This function implements the merge policy: run metadata provides defaults,
and per-query metadata wins on conflicts.

---

## QueryMode

Enum for query generator selection.

```python
from evaluateur import QueryMode

QueryMode.INSTRUCTOR  # Default: use Instructor for structured generation
```

| Value | Description |
|-------|-------------|
| `INSTRUCTOR` | Generate queries using Instructor |

---

## ContextBuilder Protocol

Protocol for per-tuple context variation.

```python
from evaluateur.queries import ContextBuilder, GeneratedTuple

class ContextBuilder(Protocol):
    def __call__(self, tuple: GeneratedTuple) -> tuple[str, dict]:
        """Return (context_string, metadata_dict)."""
        ...
```

**Example:**

```python
from evaluateur.queries import GeneratedTuple

def my_builder(t: GeneratedTuple) -> tuple[str, dict]:
    context = f"Focus on {t.get('topic')}"
    metadata = {"builder_used": True}
    return context, metadata
```

---

## Complete Example

```python
import asyncio
from pydantic import BaseModel, Field
from evaluateur import Evaluator, GeneratedQuery


class Query(BaseModel):
    topic: str = Field(..., description="subject")
    level: str = Field(..., description="difficulty")


async def main() -> None:
    evaluator = Evaluator(Query)

    async for q in evaluator.run(
        tuple_count=5,
        goals="- Test edge cases\n- Verify citations",
    ):
        # Access query text
        print(f"Query: {q.query}")

        # Access source tuple
        print(f"Topic: {q.source_tuple['topic']}")
        print(f"Level: {q.source_tuple['level']}")

        # Access metadata
        print(f"Goal-guided: {q.metadata.goal_guided}")
        if q.metadata.goal_focus:
            print(f"Focus: {q.metadata.goal_focus}")
        if q.metadata.goal_category:
            print(f"Category: {q.metadata.goal_category}")

        print("---")


asyncio.run(main())
```

## See Also

- [Evaluator](evaluator.md) - Query generation methods
- [Goals](goals.md) - Goal specification
- [Tuples](tuples.md) - Tuple generation
- [Context Builders](../guides/context-builders.md) - Advanced customization
