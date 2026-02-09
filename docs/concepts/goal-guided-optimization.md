# Goal-Guided Optimization

Evaluateur supports shaping query generation using goals. Goals are flat, flexible, and optionally categorized using the **CTO framework**: **Components**, **Trajectories**, and **Outcomes**.

## The CTO Framework (Optional)

The CTO framework provides well-known categories you can assign to goals. It is the default convention but not required.

### Components

What system **parts** should be tested? Components focus on internal capabilities:

- Freshness checks (is the data current?)
- Citation accuracy (are sources correctly referenced?)
- Missing document detection (does the system notice gaps?)

### Trajectories

What **user journeys** should be covered? Trajectories focus on interaction patterns:

- Conflict handling (what happens when sources disagree?)
- Multi-step workflows (can the system guide users through processes?)
- Recovery behavior (how does the system handle errors?)

### Outcomes

What **output qualities** matter? Outcomes focus on the final result:

- Checklist-ready responses (easy to verify)
- Actionable recommendations (clear next steps)
- Appropriate uncertainty (honest about limitations)

You can also use your own categories or skip categories entirely.

## Using Goals

### Structured Goals with GoalSpec

For precise control, define goals using the `GoalSpec` model:

```python
from evaluateur import Evaluator, Goal, GoalSpec
from pydantic import BaseModel, Field


class Query(BaseModel):
    payer: str = Field(...)
    age: str = Field(...)
    complexity: str = Field(...)


async def main() -> None:
    evaluator = Evaluator(Query)

    goals = GoalSpec(goals=[
        Goal(
            name="freshness checks",
            text="Queries should ask about effective dates and latest policy versions",
            category="components",
        ),
        Goal(
            name="grounded claims",
            text="Queries should request citations and policy section references",
            category="components",
        ),
        Goal(
            name="conflict handling",
            text="Test behavior when payer policy conflicts with FDA label",
            category="trajectories",
        ),
        Goal(
            name="checklist-ready",
            text="Queries should ask for structured lists of requirements",
            category="outcomes",
        ),
    ])

    async for q in evaluator.run(goals=goals, seed=0):
        print(f"[{q.metadata.goal_focus}] {q.query}")
```

### Goals Without Categories

Categories are optional. You can define goals without them:

```python
goals = GoalSpec(goals=[
    Goal(name="freshness", text="Ask about effective dates"),
    Goal(name="conflict handling", text="Test conflicting source behavior"),
    Goal(name="checklist output", text="Request structured lists"),
])
```

### Free-Form Goals

For quick iteration, provide goals as plain text with numbered or bulleted lists:

```python
async for q in evaluator.run(
    goals="""
    - Prioritize freshness checks and citation accuracy
    - Include conflict handling when sources disagree
    - Produce checklist-ready outputs
    """,
):
    print(q.query)
```

Evaluateur parses structured lists directly without an LLM call. CTO section headers like `Components:` are auto-detected:

```python
goals = """
Components:
- Freshness checks
- Citation accuracy

Trajectories:
- Conflict handling

Outcomes:
- Checklist-ready outputs
"""
```

For truly free-form text (no list structure), Evaluateur falls back to an LLM to parse the text into goals.

## Goal Modes

### Sample Mode (Default)

In sample mode, Evaluateur picks **one goal** per query. This ensures diversity across your generated queries:

```python
async for q in evaluator.run(
    goals=goals,
    goal_mode="sample",  # default
):
    # Each query focuses on one specific goal
    print(q.metadata.goal_focus)     # e.g. "freshness checks"
    print(q.metadata.goal_category)  # e.g. "components"
```

### Cycle Mode

In cycle mode, Evaluateur **rotates through goals** consecutively, guaranteeing even coverage:

```python
async for q in evaluator.run(
    goals=goals,
    goal_mode="cycle",
):
    # Goals rotate: goal[0] → goal[1] → goal[2] → goal[0] → ...
    print(q.metadata.goal_focus)
```

Use cycle mode when you want deterministic, balanced coverage across all goals without randomness.

### Full Mode

In full mode, **all goals** are included in every query prompt:

```python
async for q in evaluator.run(
    goals=goals,
    goal_mode="full",
):
    # Every query considers all goals
    print(q.query)
```

Use full mode when you want every query to satisfy all constraints simultaneously.

## Goal Weights

Control the relative importance of goals with weights:

```python
Goal(
    name="critical check",
    text="...",
    weight=2.0,  # Twice as likely to be sampled
)

Goal(
    name="disabled for now",
    text="...",
    weight=0.0,  # Excluded from sampling
)
```

Weights affect sampling probability in sample mode. A weight of 0 disables the goal without deleting it.

## Accessing Goal Metadata

Generated queries include goal information in their metadata:

```python
async for q in evaluator.run(goals=goals):
    meta = q.metadata

    # Was this query goal-guided?
    print(meta.goal_guided)  # True

    # Which goal mode was used?
    print(meta.goal_mode)  # "sample", "cycle", or "full"

    # Which goal was focused (in sample/cycle mode)?
    print(meta.goal_focus)     # e.g. "freshness checks"
    print(meta.goal_category)  # e.g. "components"

    # The full goal spec used
    if meta.query_goals:
        print(meta.query_goals.model_dump())
```

## Best Practices

### Start Simple

Begin with a bulleted list to explore what works:

```python
goals = """
- Test edge cases around policy conflicts
- Check handling of missing data
- Verify citation accuracy
"""
```

### Add Structure Incrementally

As you refine, convert to structured goals for precision:

```python
goals = GoalSpec(goals=[
    Goal(name="missing data", text="Test behavior when required fields are absent"),
    Goal(name="policy conflicts", text="Test conflicting policy sources", weight=2.0),
])
```

### Use Categories When Helpful

CTO categories help organize goals and make metadata filtering easier:

```python
# Filter results by category
component_queries = [q for q in results if q.metadata.goal_category == "components"]
```
