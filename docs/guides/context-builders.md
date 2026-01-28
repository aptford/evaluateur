# Context Builders

Context builders provide advanced customization for query generation by varying the prompt context per tuple.

## What is a Context Builder?

A **context builder** is a callable that takes a tuple and returns:

1. A context string to include in the query generation prompt
2. Optional metadata to attach to the generated query

```python
from evaluateur.queries import GeneratedTuple

def my_context_builder(tuple: GeneratedTuple) -> tuple[str, dict]:
    context = f"Focus on {tuple.values.get('topic', 'general')} queries"
    metadata = {"custom_field": "value"}
    return context, metadata
```

## How Context Builders Work

Without a context builder, all queries in a run share the same `instructions`:

```
┌─────────┐     ┌─────────────────┐     ┌─────────┐
│ Tuple 1 │ ──▶ │ Same context    │ ──▶ │ Query 1 │
└─────────┘     └─────────────────┘     └─────────┘
┌─────────┐     ┌─────────────────┐     ┌─────────┐
│ Tuple 2 │ ──▶ │ Same context    │ ──▶ │ Query 2 │
└─────────┘     └─────────────────┘     └─────────┘
```

With a context builder, each tuple gets a customized prompt:

```
┌─────────┐     ┌─────────────────┐     ┌─────────┐
│ Tuple 1 │ ──▶ │ Context for T1  │ ──▶ │ Query 1 │
└─────────┘     └─────────────────┘     └─────────┘
┌─────────┐     ┌─────────────────┐     ┌─────────┐
│ Tuple 2 │ ──▶ │ Context for T2  │ ──▶ │ Query 2 │
└─────────┘     └─────────────────┘     └─────────┘
```

## Built-in Use: Goal Sampling

Evaluateur uses context builders internally for goal sampling. When you set `goal_mode="sample"`, a context builder selects a different goal focus area for each query:

```python
async for q in evaluator.run(
    goals=goals,
    goal_mode="sample",  # Uses context builder internally
):
    # Each query has different goal focus
    print(q.metadata.goal_focus_area)  # "components", "trajectories", or "outcomes"
```

## Custom Query Generators

If you implement a custom query generator, accept the `context_builder` parameter:

```python
from collections.abc import AsyncIterator
from evaluateur.queries import (
    ContextBuilder,
    GeneratedQuery,
    GeneratedTuple,
    QueryMetadata,
)


class CustomQueryGenerator:
    """Example custom query generator with context builder support."""

    async def generate(
        self,
        tuples: AsyncIterator[GeneratedTuple],
        context: str,
        *,
        context_builder: ContextBuilder | None = None,
    ) -> AsyncIterator[GeneratedQuery]:
        async for t in tuples:
            # Use context builder if provided, otherwise use base context
            if context_builder is not None:
                effective_context, meta = context_builder(t)
            else:
                effective_context, meta = context, {}

            # Generate query using effective_context
            query_text = await self._generate_query(t, effective_context)

            yield GeneratedQuery(
                query=query_text,
                source_tuple=t,
                metadata=QueryMetadata(**meta),
            )

    async def _generate_query(
        self,
        tuple: GeneratedTuple,
        context: str,
    ) -> str:
        # Your query generation logic here
        return f"Query about {tuple.values} with context: {context}"
```

## Protocol Definition

The `ContextBuilder` protocol is defined as:

```python
from typing import Protocol
from evaluateur.queries import GeneratedTuple


class ContextBuilder(Protocol):
    def __call__(self, tuple: GeneratedTuple) -> tuple[str, dict]:
        """Return (context_string, metadata_dict) for this tuple."""
        ...
```

## Example: Difficulty-Based Context

Vary query generation based on a dimension value:

```python
from evaluateur.queries import GeneratedTuple


def difficulty_context_builder(t: GeneratedTuple) -> tuple[str, dict]:
    difficulty = t.values.get("difficulty", "medium")

    if difficulty == "easy":
        context = "Generate a simple, straightforward question."
        style = "beginner"
    elif difficulty == "hard":
        context = "Generate a complex question requiring deep expertise."
        style = "expert"
    else:
        context = "Generate a moderately challenging question."
        style = "intermediate"

    return context, {"difficulty_style": style}
```

## Example: Persona-Based Context

Generate queries from different user perspectives:

```python
import random
from evaluateur.queries import GeneratedTuple

PERSONAS = [
    ("curious beginner", "Ask basic questions as someone new to the topic."),
    ("skeptical expert", "Challenge assumptions and ask for evidence."),
    ("busy professional", "Ask practical, action-oriented questions."),
]


def persona_context_builder(t: GeneratedTuple) -> tuple[str, dict]:
    persona_name, persona_prompt = random.choice(PERSONAS)
    return persona_prompt, {"persona": persona_name}
```

## Example: Domain-Specific Context

Add domain knowledge based on tuple values:

```python
from evaluateur.queries import GeneratedTuple

DOMAIN_CONTEXT = {
    "healthcare": "Use medical terminology appropriately. Consider HIPAA.",
    "finance": "Reference relevant regulations. Use precise financial terms.",
    "legal": "Consider jurisdiction. Use proper legal terminology.",
}


def domain_context_builder(t: GeneratedTuple) -> tuple[str, dict]:
    domain = t.values.get("domain", "general")
    context = DOMAIN_CONTEXT.get(domain, "Ask a clear, well-formed question.")
    return context, {"domain_applied": domain}
```

## Combining Context with Base Instructions

Context builders complement, not replace, base instructions. The generated context is typically combined with any base instructions:

```python
# In a custom generator
base_instructions = "Keep questions concise."
builder_context, meta = context_builder(tuple)

# Combine both
full_context = f"{base_instructions}\n\n{builder_context}"
```

## Best Practices

1. **Keep context focused**: Return only the per-tuple variation, not general instructions

2. **Include useful metadata**: Help downstream processing by tagging queries

3. **Handle missing values**: Check if expected tuple keys exist

4. **Be deterministic when needed**: Use tuple values to seed randomness for reproducibility

```python
def deterministic_builder(t: GeneratedTuple) -> tuple[str, dict]:
    # Use tuple hash for reproducible "random" selection
    seed = hash(frozenset(t.values.items()))
    random.seed(seed)
    choice = random.choice(OPTIONS)
    return choice, {"selected": choice}
```
