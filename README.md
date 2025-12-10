## Evaluator

Synthetic evaluation helper for LLM applications, built around the
**dimensions → tuples → queries** flow described in [Hamel Husain's FAQ](https://hamel.dev/blog/posts/evals-faq/what-is-the-best-approach-for-generating-synthetic-data.html).

### Installation

The project is packaged as a normal Python library. With `uv`:

```bash
uv add evaluator
```

### Basic usage

Define a Pydantic model that represents the dimensions of your evaluation
space, then use the `Evaluator` to generate options and queries:

```python
from pydantic import BaseModel, Field

from evaluator import Evaluator, QueryMode, TupleStrategy


class Query(BaseModel):
    payer: str = Field(..., description="insurance payer, like Cigna")
    age: str = Field(..., description="patient age category, like 'adult' or 'pediatric'")
    complexity: str = Field(
        ...,
        description="complexity of the query to account for the edge cases, like 'off-label', 'comorbidities', etc",
    )
    geography: str = Field(..., description="geography indicator, like a zip code, specific state or county")


evaluator = Evaluator(Query, context="Healthcare prior authorization")

# Step 1: generate options for each dimension using Instructor
options = evaluator.generate_options(
    instructions="Focus on common US payers and edge-case clinical scenarios.",
)

# Step 2: turn options into tuples and natural language queries
output = evaluator.generate_queries(
    options=options,
    mode=QueryMode.HYBRID,
    tuple_strategy=TupleStrategy.CROSS_PRODUCT,
    tuple_count=50,
)

for q in output.queries:
    print(q.source_tuple.values, "->", q.query)
```

The evaluator uses environment variables (for example `OPENAI_API_KEY`)
and supports any provider that `instructor` supports. You can customise the
provider and model via the `LLMClient` helper if needed.

If your input model already uses iterator fields (for example
`payer: list[str] = ["Cigna", "Aetna"]`), those lists are treated as fixed
options and are not modified by `generate_options()`. Scalar fields of any
basic type (`str`, `int`, `float`, and so on) are turned into lists of
options automatically.
