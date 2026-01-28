# Custom Goals

Goals shape query generation by focusing on specific aspects of your system. This guide covers both structured and free-form goal definitions.

## Structured Goals

Use `GoalSpec` for precise control over query generation.

### Basic Structure

```python
from evaluateur import GoalItem, GoalLayer, GoalSpec

goals = GoalSpec(
    components=GoalLayer(
        summary="Test system internals",
        items=[
            GoalItem(name="data freshness"),
            GoalItem(name="source attribution"),
        ],
    ),
    trajectories=GoalLayer(
        items=[
            GoalItem(name="multi-step workflows"),
        ],
    ),
    outcomes=GoalLayer(
        items=[
            GoalItem(name="actionable responses"),
        ],
    ),
)
```

### GoalItem Options

Each `GoalItem` supports several configuration options:

```python
GoalItem(
    name="citation accuracy",           # Required: short name
    description="Verify sources are correctly cited",  # Optional: longer description
    weight=1.5,                          # Optional: sampling weight (default 1.0)
    must_include=["cite", "source"],     # Optional: required terms
    avoid=["unverified", "probably"],    # Optional: terms to avoid
    examples=[                           # Optional: example queries
        "Can you cite the source for that claim?",
        "Which policy section covers this?",
    ],
)
```

### Complete Example

```python
import asyncio
from pydantic import BaseModel, Field
from evaluateur import Evaluator, GoalItem, GoalLayer, GoalSpec


class MedicalQuery(BaseModel):
    condition: str = Field(..., description="medical condition")
    treatment: str = Field(..., description="treatment type")


async def main() -> None:
    evaluator = Evaluator(MedicalQuery)

    goals = GoalSpec(
        components=GoalLayer(
            summary="Test clinical accuracy and data handling",
            items=[
                GoalItem(
                    name="evidence grading",
                    description="Queries should reference evidence quality",
                    must_include=["evidence", "study", "trial"],
                    examples=[
                        "What's the evidence level for this treatment?",
                        "Are there randomized trials supporting this?",
                    ],
                ),
                GoalItem(
                    name="contraindication awareness",
                    must_include=["contraindicated", "avoid", "risk"],
                    avoid=["always safe", "no side effects"],
                ),
            ],
        ),
        trajectories=GoalLayer(
            items=[
                GoalItem(
                    name="shared decision making",
                    description="Support patient-provider conversations",
                    must_include=["options", "discuss with doctor"],
                ),
                GoalItem(
                    name="escalation paths",
                    description="Know when to refer to specialists",
                    must_include=["specialist", "refer", "urgent"],
                ),
            ],
        ),
        outcomes=GoalLayer(
            items=[
                GoalItem(
                    name="patient-friendly language",
                    avoid=["medical jargon", "abbreviations"],
                    examples=[
                        "Explain this in simple terms",
                        "What does this mean for my daily life?",
                    ],
                ),
            ],
        ),
    )

    async for q in evaluator.run(goals=goals, tuple_count=10, seed=42):
        print(f"[{q.metadata.goal_focus_area}] {q.query}")


asyncio.run(main())
```

## Free-Form Goals

For quick prototyping, provide goals as plain text:

```python
import asyncio
from pydantic import BaseModel, Field
from evaluateur import Evaluator


class Query(BaseModel):
    topic: str = Field(..., description="subject area")


async def main() -> None:
    evaluator = Evaluator(Query)

    goals = """
    Components:
    - Test data freshness (queries should ask about recent updates)
    - Verify citation accuracy (references should be traceable)

    Trajectories:
    - Cover multi-step research workflows
    - Include disambiguation when topics are ambiguous

    Outcomes:
    - Responses should be actionable, not just informational
    - Include clear next steps or recommendations
    """

    async for q in evaluator.run(goals=goals, tuple_count=5):
        print(q.query)


asyncio.run(main())
```

### Tips for Free-Form Goals

1. **Be specific**: Include concrete terms and phrases

    ```
    # Less effective
    "Make sure responses are good"

    # More effective
    "Responses should cite specific sources and include publication dates"
    ```

2. **Include examples**: Help the LLM understand what you want

    ```
    Components:
    - Citation accuracy: "Which study supports this claim?"
    - Freshness: "Is this based on the latest guidelines?"
    ```

3. **Use the three layers**: Structure helps with parsing

    ```
    Components: [internal capabilities]
    Trajectories: [user journeys]
    Outcomes: [output qualities]
    ```

## Goal Weights

Control sampling probability with weights:

```python
goals = GoalSpec(
    components=GoalLayer(
        items=[
            GoalItem(name="critical feature", weight=3.0),   # 3x more likely
            GoalItem(name="nice to have", weight=0.5),       # Less common
            GoalItem(name="temporarily disabled", weight=0),  # Excluded
        ],
    ),
)
```

Weights only affect `goal_mode="sample"` (the default). In `goal_mode="full"`, all goals are included.

## Goal Modes

### Sample Mode (Default)

Picks one focus area (components, trajectories, or outcomes) per query:

```python
async for q in evaluator.run(
    goals=goals,
    goal_mode="sample",
):
    # Each query focuses on ONE layer
    print(q.metadata.goal_focus_area)
```

This creates diverse test coverage across all goal types.

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
    print(f"Focus area: {q.metadata.goal_focus_area}")

    if q.metadata.query_goals:
        # Access the full GoalSpec used
        spec = q.metadata.query_goals
        print(f"Components: {[g.name for g in spec.components.items]}")
```

## Converting Between Formats

Parse free-form text into structured goals:

```python
from evaluateur import LLMClient, GoalSpec

client = LLMClient.from_env()

# Parse free-form text
spec = await GoalSpec.from_text(
    client,
    "Test freshness and citation accuracy. Cover error recovery workflows.",
)

# Now use as structured goals
print(spec.components.items)
print(spec.trajectories.items)
print(spec.outcomes.items)
```

## Best Practices

1. **Start with free-form** to explore what works, then convert to structured for production

2. **Balance the layers** - ensure each has content for diverse sampling

3. **Use `must_include` sparingly** - too many constraints make generation harder

4. **Add examples** - they significantly improve query quality

5. **Review generated queries** - adjust goals based on what you see
