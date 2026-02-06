# Provider Configuration

Evaluateur uses [Instructor](https://python.useinstructor.com/) under the hood, so it works with any provider that Instructor supports — OpenAI, Anthropic, Google, Mistral, Groq, Ollama, and more.

## Quick Start

Pass a `"provider/model-name"` string to the `llm` parameter:

```python
from evaluateur import Evaluator

# OpenAI
evaluator = Evaluator(MyModel, llm="openai/gpt-4.1-mini")

# Anthropic
evaluator = Evaluator(MyModel, llm="anthropic/claude-3-5-sonnet-latest")

# Google
evaluator = Evaluator(MyModel, llm="google/gemini-2.0-flash")
```

Set the API key for your chosen provider as an environment variable
(e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`).
Evaluateur loads `.env` files automatically.

## Default from Environment

When `llm` is omitted, Evaluateur reads the `EVALUATEUR_MODEL` environment
variable (default: `"openai/gpt-4.1-mini"`):

```bash
export EVALUATEUR_MODEL=anthropic/claude-3-5-sonnet-latest
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
from evaluateur import Evaluator

evaluator = Evaluator(MyModel)  # uses anthropic/claude-3-5-sonnet-latest
```

## Provider Examples

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
```

```python
evaluator = Evaluator(MyModel, llm="openai/gpt-4.1-mini")
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
evaluator = Evaluator(MyModel, llm="anthropic/claude-3-5-sonnet-latest")
```

### Google Gemini

```bash
export GEMINI_API_KEY=...
```

```python
evaluator = Evaluator(MyModel, llm="google/gemini-2.0-flash")
```

### Groq

```bash
export GROQ_API_KEY=...
```

```python
evaluator = Evaluator(MyModel, llm="groq/llama-3.1-70b-versatile")
```

### Ollama (Local)

```python
evaluator = Evaluator(MyModel, llm="ollama/llama3.2")
```

### Mistral

```bash
export MISTRAL_API_KEY=...
```

```python
evaluator = Evaluator(MyModel, llm="mistral/mistral-large-latest")
```

## Advanced: Bring Your Own Client

For observability wrappers, custom base URLs, or any configuration that
`instructor.from_provider()` doesn't cover, pass a pre-configured async
Instructor client directly:

```python
import instructor
from openai import AsyncOpenAI
from evaluateur import Evaluator

# Custom OpenAI client (e.g. proxy, Azure, or wrapped for tracing)
inst = instructor.from_openai(AsyncOpenAI(base_url="https://my-proxy.com/v1"))
evaluator = Evaluator(MyModel, client=inst, model_name="gpt-4o")
```

### Anthropic (Custom Client)

```python
import instructor
from anthropic import AsyncAnthropic
from evaluateur import Evaluator

inst = instructor.from_anthropic(AsyncAnthropic())
evaluator = Evaluator(MyModel, client=inst, model_name="claude-3-5-sonnet-latest")
```

### Azure OpenAI

```python
import instructor
from openai import AsyncAzureOpenAI
from evaluateur import Evaluator

azure = AsyncAzureOpenAI(
    api_key="your-azure-key",
    api_version="2024-02-01",
    azure_endpoint="https://your-resource.openai.azure.com",
)
inst = instructor.from_openai(azure)
evaluator = Evaluator(MyModel, client=inst, model_name="your-deployment-name")
```

### LangSmith Tracing

```python
import instructor
from langsmith import wrappers
from openai import AsyncOpenAI
from evaluateur import Evaluator

traced = wrappers.wrap_openai(AsyncOpenAI())
inst = instructor.from_openai(traced)
evaluator = Evaluator(MyModel, client=inst, model_name="gpt-4o")
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EVALUATEUR_MODEL` | `openai/gpt-4.1-mini` | Default model in `"provider/model-name"` format |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `GROQ_API_KEY` | — | Groq API key |
| `MISTRAL_API_KEY` | — | Mistral API key |

API key variable names are determined by each provider's SDK, not by
evaluateur. Consult the provider's documentation for details.

## Best Practices

1. **Use `llm=` for simple cases** — one string, no boilerplate.

2. **Use `client=` + `model_name=` for advanced cases** — observability
   wrappers, proxies, Azure, or any custom Instructor configuration.

3. **Use cheaper models for development** — option and query generation
   make multiple LLM calls. Use a fast model during iteration.

```python
import os
from evaluateur import Evaluator

model = "openai/gpt-4.1-nano" if os.getenv("ENV") == "dev" else "openai/gpt-4.1-mini"
evaluator = Evaluator(MyModel, llm=model)
```
