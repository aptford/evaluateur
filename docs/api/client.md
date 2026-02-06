# Client Configuration

Provider configuration is handled directly through the `Evaluator` constructor. See the [Provider Configuration](../guides/provider-configuration.md) guide for full details.

## Quick Reference

```python
from evaluateur import Evaluator

# Simple: pass a "provider/model-name" string
evaluator = Evaluator(MyModel, llm="openai/gpt-4.1-mini")
evaluator = Evaluator(MyModel, llm="anthropic/claude-4-5-haiku-latest")

# Default: reads EVALUATEUR_MODEL env var
evaluator = Evaluator(MyModel)

# Advanced: bring your own Instructor client
import instructor
from openai import AsyncOpenAI

inst = instructor.from_openai(AsyncOpenAI())
evaluator = Evaluator(MyModel, client=inst, model_name="claude-4-5-haiku-latest")
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EVALUATEUR_MODEL` | `openai/gpt-4.1-mini` | Default model in `"provider/model-name"` format |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |

## See Also

- [Evaluator](evaluator.md) - Full constructor reference
- [Provider Configuration](../guides/provider-configuration.md) - Detailed provider setup
