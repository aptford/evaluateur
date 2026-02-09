# Custom Goals

Goals shape query generation by focusing on specific aspects of your system. This guide covers both structured and free-form goal definitions.

## Structured Goals

Use `GoalSpec` for precise control over query generation.

### Basic Structure

```python
from evaluateur import Goal, GoalSpec

goals = GoalSpec(goals=[
    Goal(name="data freshness", text="Test whether queries surface current data"),
    Goal(name="source attribution", text="Ensure citations are requested"),
    Goal(name="multi-step workflows", text="Cover multi-step user journeys"),
    Goal(name="actionable responses", text="Queries should request clear next steps"),
])
```

### Goal Options

Each `Goal` supports several configuration options:

```python
Goal(
    name="citation accuracy",             # Optional: short label
    text="Verify sources are correctly cited",  # Required: full description
    weight=1.5,                            # Optional: sampling weight (default 1.0)
    category="components",                 # Optional: CTO category or custom
)
```

### Using CTO Categories

The CTO framework (Components / Trajectories / Outcomes) provides well-known categories:

```python
goals = GoalSpec(goals=[
    Goal(
        name="evidence grading",
        text="Queries should reference evidence quality",
        category="components",
    ),
    Goal(
        name="contraindication awareness",
        text="Test detection of drug interactions and contraindications",
        category="components",
    ),
    Goal(
        name="shared decision making",
        text="Support patient-provider conversations about treatment options",
        category="trajectories",
    ),
    Goal(
        name="escalation paths",
        text="Know when to refer to specialists",
        category="trajectories",
    ),
    Goal(
        name="patient-friendly language",
        text="Responses should avoid medical jargon and use plain language",
        category="outcomes",
    ),
])
```

### Complete Example

```python
import asyncio
from pydantic import BaseModel, Field
from evaluateur import Evaluator, Goal, GoalSpec


class MedicalQuery(BaseModel):
    condition: str = Field(..., description="medical condition")
    treatment: str = Field(..., description="treatment type")


async def main() -> None:
    evaluator = Evaluator(MedicalQuery)

    goals = GoalSpec(goals=[
        Goal(
            name="evidence grading",
            text="Queries should reference evidence quality and ask about study types",
            category="components",
            weight=2.0,
        ),
        Goal(
            name="contraindication awareness",
            text="Test detection of contraindications and drug interaction risks",
            category="components",
        ),
        Goal(
            name="shared decision making",
            text="Support patient-provider conversations about treatment options",
            category="trajectories",
        ),
        Goal(
            name="escalation paths",
            text="Test when to refer to specialists or flag urgent situations",
            category="trajectories",
        ),
        Goal(
            name="patient-friendly language",
            text="Responses should use plain language and avoid abbreviations",
            category="outcomes",
        ),
    ])

    async for q in evaluator.run(goals=goals, tuple_count=10, seed=42):
        print(f"[{q.metadata.goal_focus}] {q.query}")


asyncio.run(main())
```

## Free-Form Goals

For quick prototyping, provide goals as plain text. Numbered or bulleted lists are parsed directly without an LLM call:

```python
import asyncio
from pydantic import BaseModel, Field
from evaluateur import Evaluator


class Query(BaseModel):
    topic: str = Field(..., description="subject area")


async def main() -> None:
    evaluator = Evaluator(Query)

    goals = """
    - Test data freshness (queries should ask about recent updates)
    - Verify citation accuracy (references should be traceable)
    - Cover multi-step research workflows
    - Include disambiguation when topics are ambiguous
    - Responses should be actionable, not just informational
    - Include clear next steps or recommendations
    """

    async for q in evaluator.run(goals=goals, tuple_count=5):
        print(q.query)


asyncio.run(main())
```

### CTO Headers in Text

CTO section headers are auto-detected and assign categories:

```python
goals = """
Components:
- Test data freshness
- Verify citation accuracy

Trajectories:
- Cover multi-step research workflows
- Include disambiguation when topics are ambiguous

Outcomes:
- Responses should be actionable
- Include clear next steps
"""
```

### Tips for Free-Form Goals

1. **Be specific**: Include concrete terms and phrases

    ```
    # Less effective
    "Make sure responses are good"

    # More effective
    "Responses should cite specific sources and include publication dates"
    ```

2. **Use bullet points**: Structured lists are parsed without an LLM call

    ```
    - Citation accuracy: references should be traceable
    - Freshness: ask about the latest guidelines
    ```

3. **Use CTO headers optionally**: They assign categories automatically

    ```
    Components:
    - Citation accuracy
    Trajectories:
    - Error recovery
    ```

## Goal Weights

Control sampling probability with weights:

```python
goals = GoalSpec(goals=[
    Goal(name="critical feature", text="...", weight=3.0),   # 3x more likely
    Goal(name="nice to have", text="...", weight=0.5),       # Less common
    Goal(name="temporarily disabled", text="...", weight=0), # Excluded
])
```

Weights only affect `goal_mode="sample"` (the default). In `goal_mode="full"`, all goals are included.

## Goal Modes

### Sample Mode (Default)

Picks one goal per query at random (weighted):

```python
async for q in evaluator.run(
    goals=goals,
    goal_mode="sample",
):
    # Each query focuses on ONE goal
    print(q.metadata.goal_focus)
```

This creates diverse test coverage across all goals.

### Cycle Mode

Rotates through goals consecutively:

```python
async for q in evaluator.run(
    goals=goals,
    goal_mode="cycle",
):
    # Goals rotate: goal[0] → goal[1] → goal[2] → ...
    print(q.metadata.goal_focus)
```

### Full Mode

Includes all goals in every query prompt:

```python
async for q in evaluator.run(
    goals=goals,
    goal_mode="full",
):
    # Every query considers ALL goals
    print(q.query)
```

Use this when queries should satisfy multiple constraints simultaneously.

## Accessing Goal Metadata

Every generated query includes goal information:

```python
async for q in evaluator.run(goals=goals):
    print(f"Query: {q.query}")
    print(f"Goal-guided: {q.metadata.goal_guided}")
    print(f"Goal mode: {q.metadata.goal_mode}")
    print(f"Goal focus: {q.metadata.goal_focus}")
    print(f"Goal category: {q.metadata.goal_category}")

    if q.metadata.query_goals:
        # Access the full GoalSpec used
        spec = q.metadata.query_goals
        print(f"Goals: {[g.name for g in spec.goals]}")
```

## Converting Between Formats

Parse free-form text into structured goals using `evaluator.parse_goals()`:

```python
from pydantic import BaseModel, Field
from evaluateur import Evaluator


class Query(BaseModel):
    topic: str = Field(..., description="subject area")


evaluator = Evaluator(Query)

# Parse free-form text
spec = await evaluator.parse_goals(
    "Test freshness and citation accuracy. Cover error recovery workflows.",
)

# Now use as structured goals
for goal in spec.goals:
    print(f"{goal.name}: {goal.text} (category: {goal.category})")
```

In most cases, you don't need to call this directly -- pass a string to
`goals=` in `evaluator.run()` and it handles parsing automatically.

## Best Practices

1. **Start with a bulleted list** to explore what works, then convert to structured for production

2. **Use weights** to emphasize important goals and disable irrelevant ones

3. **Add categories when helpful** -- they make metadata filtering easier

4. **Review generated queries** -- adjust goals based on what you see
