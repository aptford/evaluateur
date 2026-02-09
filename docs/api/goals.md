# Goals

Data models for goal-guided query optimization.

## Overview

The goals system uses a flat list of `Goal` objects, optionally categorized using the CTO framework:

- **Components**: System internals (freshness, citations, data handling)
- **Trajectories**: User journeys (workflows, error recovery, multi-step)
- **Outcomes**: Output qualities (actionable, clear, accurate)

Categories are optional. You can use any string or skip them entirely.

```python
from evaluateur import Goal, GoalSpec

goals = GoalSpec(goals=[
    Goal(name="freshness", text="Test data currency", category="components"),
    Goal(name="error recovery", text="Test error handling", category="trajectories"),
    Goal(name="actionable", text="Request clear next steps", category="outcomes"),
    Goal(name="edge cases", text="Cover unusual inputs"),  # no category
])
```

## GoalSpec

Top-level container for goal guidance.

::: evaluateur.GoalSpec
    options:
      show_source: true
      members:
        - is_empty
        - available_goals
        - render_prompt
        - to_metadata

### Constructor

```python
GoalSpec(
    goals: list[Goal] = [],
)
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `goals` | `list[Goal]` | Flat list of evaluation goals |

### Methods

#### `is_empty()`

Check if no active goals are specified.

```python
def is_empty(self) -> bool
```

Returns `True` if no goals exist or all have weight <= 0.

#### `available_goals()`

Get goals with positive weight.

```python
def available_goals(self) -> list[Goal]
```

#### `render_prompt()`

Render the spec into a compact prompt string.

```python
def render_prompt(self) -> str
```

---

## Goal

A single evaluation goal for shaping query generation.

::: evaluateur.Goal
    options:
      show_source: true

### Constructor

```python
Goal(
    name: str = "",
    text: str,
    weight: float = 1.0,
    category: str = "",
)
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `""` | Short goal label |
| `text` | `str` | required | Full goal description |
| `weight` | `float` | `1.0` | Relative importance (0 disables) |
| `category` | `str` | `""` | Optional category (CTO or custom) |

**Example:**

```python
from evaluateur import Goal

goal = Goal(
    name="citation accuracy",
    text="Ensure all claims reference specific sources with publication dates",
    weight=1.5,
    category="components",
)
```

### Weight Behavior

- `weight > 0`: Normal goal, sampled proportionally
- `weight = 0`: Disabled (excluded from sampling)
- Higher weight = more likely to be sampled

---

## Evaluator.parse_goals()

Parse free-form text into a structured `GoalSpec`.

```python
from pydantic import BaseModel, Field
from evaluateur import Evaluator

class MyModel(BaseModel):
    topic: str = Field(..., description="subject area")

evaluator = Evaluator(MyModel, llm="openai/gpt-4.1-mini")
spec = await evaluator.parse_goals(
    "Test freshness and citation accuracy. Include error recovery.",
)
```

Structured text (numbered/bulleted lists) is parsed without an LLM call. Free-form text falls back to LLM enrichment.

In most cases, pass a string directly to `goals=` in `evaluator.run()` instead
of calling this method manually.

---

## CTO Constants

Well-known category constants for the CTO framework:

```python
from evaluateur.goals.constants import COMPONENTS, TRAJECTORIES, OUTCOMES, CTO_CATEGORIES

COMPONENTS    # "components"
TRAJECTORIES  # "trajectories"
OUTCOMES      # "outcomes"
CTO_CATEGORIES  # ("components", "trajectories", "outcomes")
```

---

## GoalMode

```python
GoalMode = Literal["full", "sample", "cycle"]
```

| Mode | Behavior |
|------|----------|
| `"sample"` | Pick one goal per query at random (weighted) |
| `"cycle"` | Rotate through goals consecutively (even coverage) |
| `"full"` | Include all goals in every query |

---

## Complete Example

```python
import asyncio
from pydantic import BaseModel, Field
from evaluateur import Evaluator, Goal, GoalSpec


class Query(BaseModel):
    topic: str = Field(..., description="subject area")


async def main() -> None:
    evaluator = Evaluator(Query)

    goals = GoalSpec(goals=[
        Goal(
            name="freshness",
            text="Queries should ask about current data and latest versions",
            category="components",
            weight=2.0,
        ),
        Goal(
            name="citations",
            text="Queries should request source references",
            category="components",
        ),
        Goal(
            name="error handling",
            text="Test graceful degradation when data is missing",
            category="trajectories",
        ),
        Goal(
            name="actionable",
            text="Queries should request next steps and recommendations",
            category="outcomes",
        ),
    ])

    async for q in evaluator.run(
        goals=goals,
        goal_mode="sample",
        tuple_count=10,
    ):
        print(f"[{q.metadata.goal_focus}] {q.query}")


asyncio.run(main())
```

## See Also

- [Goal-Guided Optimization](../concepts/goal-guided-optimization.md) - Conceptual overview
- [Custom Goals Guide](../guides/custom-goals.md) - Usage patterns
- [Evaluator](evaluator.md) - Using goals with the evaluator
