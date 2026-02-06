# Goals

Data models for goal-guided query optimization.

## Overview

The goals system uses a three-layer framework:

- **Components**: System internals (freshness, citations, data handling)
- **Trajectories**: User journeys (workflows, error recovery, multi-step)
- **Outcomes**: Output qualities (actionable, clear, accurate)

```python
from evaluateur import GoalItem, GoalLayer, GoalSpec

goals = GoalSpec(
    components=GoalLayer(items=[GoalItem(name="freshness")]),
    trajectories=GoalLayer(items=[GoalItem(name="error recovery")]),
    outcomes=GoalLayer(items=[GoalItem(name="actionable")]),
)
```

## GoalSpec

Top-level container for goal guidance.

::: evaluateur.GoalSpec
    options:
      show_source: true
      members:
        - is_empty
        - render_prompt
        - available_focus_areas
        - focus_weight
        - render_focused_prompt
        - to_metadata

### Constructor

```python
GoalSpec(
    components: GoalLayer = GoalLayer(),
    trajectories: GoalLayer = GoalLayer(),
    outcomes: GoalLayer = GoalLayer(),
)
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `components` | `GoalLayer` | System component goals |
| `trajectories` | `GoalLayer` | User journey goals |
| `outcomes` | `GoalLayer` | Output quality goals |

### Methods

#### `is_empty()`

Check if no goals are specified.

```python
def is_empty(self) -> bool
```

#### `available_focus_areas()`

Get non-empty goal layers.

```python
def available_focus_areas(self) -> list[GoalFocusArea]
# Returns: ["components", "trajectories", "outcomes"] (only non-empty ones)
```

#### `focus_weight()`

Get the effective weight for a focus area.

```python
def focus_weight(self, focus_area: GoalFocusArea) -> float
```

---

## parse_goal_spec()

Parse free-form text into a structured `GoalSpec`.

```python
from evaluateur import parse_goal_spec
from evaluateur.client import resolve_client

client = resolve_client(llm="openai/gpt-4.1-mini")
spec = await parse_goal_spec(
    client,
    "Test freshness and citation accuracy. Include error recovery.",
)
```

In most cases, pass a string directly to `goals=` in `evaluator.run()` instead
of calling this function manually.

This function uses Instructor + Pydantic parsing to extract a stable schema
from free-form text. It returns an empty `GoalSpec` if the text is empty.

---

## GoalLayer

A collection of goals for one framework layer.

::: evaluateur.GoalLayer
    options:
      show_source: true
      members:
        - weight
        - has_content

### Constructor

```python
GoalLayer(
    summary: str | None = None,
    items: list[GoalItem] = [],
)
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `summary` | `str | None` | One-paragraph summary of what matters |
| `items` | `list[GoalItem]` | Individual goals in this layer |

**Example:**

```python
from evaluateur import GoalItem, GoalLayer

layer = GoalLayer(
    summary="Test data freshness and citation accuracy",
    items=[
        GoalItem(name="freshness checks"),
        GoalItem(name="source attribution"),
    ],
)
```

### Methods

#### `weight()`

Calculate the effective weight for this layer.

```python
def weight(self) -> float
```

Returns the sum of item weights, or 1.0 if only a summary exists.

#### `has_content()`

Check if the layer has any active content.

```python
def has_content(self) -> bool
```

Returns `True` if there's a summary or any items with weight > 0.

---

## GoalItem

A single goal for shaping query generation.

::: evaluateur.GoalItem
    options:
      show_source: true

### Constructor

```python
GoalItem(
    name: str,
    description: str | None = None,
    weight: float = 1.0,
    must_include: list[str] = [],
    avoid: list[str] = [],
    examples: list[str] = [],
)
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Short goal name |
| `description` | `str | None` | `None` | Detailed description |
| `weight` | `float` | `1.0` | Relative importance (0 disables) |
| `must_include` | `list[str]` | `[]` | Required terms/phrases |
| `avoid` | `list[str]` | `[]` | Terms to avoid |
| `examples` | `list[str]` | `[]` | Example queries |

**Example:**

```python
from evaluateur import GoalItem

item = GoalItem(
    name="citation accuracy",
    description="Ensure all claims reference specific sources",
    weight=1.5,
    must_include=["cite", "source", "reference"],
    avoid=["probably", "might be"],
    examples=[
        "Which study supports this recommendation?",
        "Can you cite the policy section?",
    ],
)
```

### Weight Behavior

- `weight > 0`: Normal goal, sampled proportionally
- `weight = 0`: Disabled (excluded from sampling)
- Higher weight = more likely to be sampled

---

## Type Aliases

### GoalFocusArea

```python
GoalFocusArea = Literal["components", "trajectories", "outcomes"]
```

### GoalMode

```python
GoalMode = Literal["full", "sample", "cycle"]
```

| Mode | Behavior |
|------|----------|
| `"sample"` | Pick one focus area per query at random (diverse) |
| `"cycle"` | Rotate through focus areas consecutively (even coverage) |
| `"full"` | Include all goals in every query |

---

## Complete Example

```python
import asyncio
from pydantic import BaseModel, Field
from evaluateur import Evaluator, GoalItem, GoalLayer, GoalSpec


class Query(BaseModel):
    topic: str = Field(..., description="subject area")


async def main() -> None:
    evaluator = Evaluator(Query)

    goals = GoalSpec(
        components=GoalLayer(
            summary="Test system reliability",
            items=[
                GoalItem(
                    name="freshness",
                    must_include=["current", "latest", "updated"],
                    weight=2.0,
                ),
                GoalItem(
                    name="citations",
                    must_include=["source", "reference"],
                ),
            ],
        ),
        trajectories=GoalLayer(
            items=[
                GoalItem(
                    name="error handling",
                    description="Test graceful degradation",
                    examples=["What if the data is missing?"],
                ),
            ],
        ),
        outcomes=GoalLayer(
            items=[
                GoalItem(
                    name="actionable",
                    must_include=["next steps", "recommendation"],
                    avoid=["unclear", "maybe"],
                ),
            ],
        ),
    )

    async for q in evaluator.run(
        goals=goals,
        goal_mode="sample",
        tuple_count=10,
    ):
        print(f"[{q.metadata.goal_focus_area}] {q.query}")


asyncio.run(main())
```

## See Also

- [Goal-Guided Optimization](../concepts/goal-guided-optimization.md) - Conceptual overview
- [Custom Goals Guide](../guides/custom-goals.md) - Usage patterns
- [Evaluator](evaluator.md) - Using goals with the evaluator
