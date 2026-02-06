# Goal-Guided Optimization

Evaluateur supports shaping query generation using a three-layer goal framework: **Components**, **Trajectories**, and **Outcomes**. This helps you generate queries that stress-test specific aspects of your system.

## The Three Layers

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

## Using Goals

### Structured Goals with GoalSpec

For precise control, define goals using the `GoalSpec` model:

```python
from evaluateur import Evaluator, GoalItem, GoalLayer, GoalSpec
from pydantic import BaseModel, Field


class Query(BaseModel):
    payer: str = Field(...)
    age: str = Field(...)
    complexity: str = Field(...)


async def main() -> None:
    evaluator = Evaluator(Query)

    goals = GoalSpec(
        components=GoalLayer(
            summary="Stress test freshness and citation accuracy",
            items=[
                GoalItem(
                    name="freshness checks",
                    must_include=["effective date", "latest policy"],
                    avoid=["undated references"],
                ),
                GoalItem(
                    name="grounded claims",
                    must_include=["cite", "policy section"],
                ),
            ],
        ),
        trajectories=GoalLayer(
            items=[
                GoalItem(
                    name="conflict handling",
                    description="Test behavior when payer policy conflicts with FDA label",
                    must_include=["conflicting", "payer policy"],
                ),
            ],
        ),
        outcomes=GoalLayer(
            items=[
                GoalItem(
                    name="checklist-ready",
                    must_include=["payer", "age", "diagnosis"],
                    examples=[
                        "List the prior auth requirements for this procedure",
                        "What documents do I need to submit?",
                    ],
                ),
            ],
        ),
    )

    async for q in evaluator.run(goals=goals, seed=0):
        print(f"[{q.metadata.goal_focus_area}] {q.query}")
```

### Free-Form Goals

For quick iteration, provide goals as plain text:

```python
async for q in evaluator.run(
    goals="""
    Components: prioritize freshness checks and citation accuracy.
    Trajectories: include conflict handling when sources disagree.
    Outcomes: produce checklist-ready outputs.
    """,
):
    print(q.query)
```

Evaluateur uses an LLM to parse this into a structured `GoalSpec`.

!!! tip "Writing Effective Free-Form Goals"
    Include concrete examples and measurable criteria:
    
    - "cite policy section" (specific)
    - "ask a clarifying question if payer is missing" (behavioral)
    - "return a checklist of required documents" (output format)

## Goal Modes

### Sample Mode (Default)

In sample mode, Evaluateur picks **one focus area** per query. This ensures diversity across your generated queries:

```python
async for q in evaluator.run(
    goals=goals,
    goal_mode="sample",  # default
):
    # Each query focuses on components, trajectories, OR outcomes
    print(q.metadata.goal_focus_area)  # "components", "trajectories", or "outcomes"
```

### Cycle Mode

In cycle mode, Evaluateur **rotates through focus areas** consecutively, guaranteeing even coverage across all layers:

```python
async for q in evaluator.run(
    goals=goals,
    goal_mode="cycle",
):
    # Focus areas rotate: components → trajectories → outcomes → components → ...
    print(q.metadata.goal_focus_area)
```

Use cycle mode when you want deterministic, balanced coverage across all focus areas without randomness.

### Full Mode

In full mode, **all goals** are included in every query prompt:

```python
async for q in evaluator.run(
    goals=goals,
    goal_mode="full",
):
    # Every query considers all goal layers
    print(q.query)
```

Use full mode when you want every query to satisfy all constraints simultaneously.

## Goal Weights

Control the relative importance of goals with weights:

```python
GoalItem(
    name="critical check",
    weight=2.0,  # Twice as likely to be sampled
)

GoalItem(
    name="disabled for now",
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

    # Which layer was focused (in sample/cycle mode)?
    print(meta.goal_focus_area)  # "components", "trajectories", or "outcomes"

    # The full goal spec used
    if meta.query_goals:
        print(meta.query_goals.model_dump())
```

## Best Practices

### Start Simple

Begin with free-form goals to explore what works:

```python
goals = "Test edge cases around policy conflicts and missing data"
```

### Add Structure Incrementally

As you refine, convert to structured goals for precision:

```python
goals = GoalSpec(
    components=GoalLayer(items=[
        GoalItem(name="missing data handling", must_include=["if missing", "ask for"]),
    ]),
)
```

### Use Examples

Include example queries in your goals to guide the LLM:

```python
GoalItem(
    name="clarification requests",
    examples=[
        "Could you clarify which payer you're asking about?",
        "I need the patient's age to answer accurately.",
    ],
)
```

### Balance Layers

Ensure all three layers have content for diverse query generation in sample mode. Empty layers are skipped during sampling.
