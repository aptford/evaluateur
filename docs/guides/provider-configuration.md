# Provider Configuration

Evaluateur uses [Instructor](https://python.useinstructor.com/) for structured LLM outputs, supporting any provider that Instructor supports.

## Quick Setup

### OpenAI (Default)

Set your API key and you're ready:

```bash
export OPENAI_API_KEY=sk-your-key-here
```

```python
from evaluateur import Evaluator

evaluator = Evaluator(MyModel)  # Uses OpenAI by default
```

### Model Selection

Override the default model (`gpt-4o-mini`):

```bash
export EVALUATEUR_MODEL_NAME=gpt-4o
```

Or configure programmatically:

```python
from evaluateur import LLMClient, Evaluator

client = LLMClient.from_env(model_name="gpt-4o")
evaluator = Evaluator(MyModel, client=client)
```

## LLMClient Factory Methods

### from_env()

The simplest approach. Reads configuration from environment variables:

```python
from evaluateur import LLMClient

# OpenAI with defaults
client = LLMClient.from_env()

# OpenAI with custom model
client = LLMClient.from_env(model_name="gpt-4-turbo")

# Other providers via Instructor's from_provider
client = LLMClient.from_env(provider="anthropic", model_name="claude-3-opus")
```

### from_openai()

Use a custom AsyncOpenAI client:

```python
from openai import AsyncOpenAI
from evaluateur import LLMClient

# Custom base URL (e.g., proxy or alternative endpoint)
openai_client = AsyncOpenAI(
    api_key="your-key",
    base_url="https://your-proxy.com/v1",
)
client = LLMClient.from_openai(openai_client)

# With custom timeout
openai_client = AsyncOpenAI(timeout=60.0)
client = LLMClient.from_openai(openai_client, model_name="gpt-4o")
```

### from_instructor()

Use a pre-configured Instructor client:

```python
import instructor
from openai import AsyncOpenAI
from evaluateur import LLMClient

# Custom Instructor configuration
inst = instructor.from_openai(
    AsyncOpenAI(),
    mode=instructor.Mode.JSON,
)
client = LLMClient.from_instructor(inst, model_name="gpt-4o")
```

## Provider Examples

### Anthropic

```python
import instructor
from anthropic import AsyncAnthropic
from evaluateur import LLMClient

anthropic_client = AsyncAnthropic()
inst = instructor.from_anthropic(anthropic_client)

client = LLMClient.from_instructor(
    inst,
    model_name="claude-3-5-sonnet-latest",
    provider="anthropic",
)
```

### Azure OpenAI

```python
from openai import AsyncAzureOpenAI
from evaluateur import LLMClient

azure_client = AsyncAzureOpenAI(
    api_key="your-azure-key",
    api_version="2024-02-01",
    azure_endpoint="https://your-resource.openai.azure.com",
)

client = LLMClient.from_openai(
    azure_client,
    model_name="your-deployment-name",
    provider="azure",
)
```

### Local Models (Ollama)

```python
from openai import AsyncOpenAI
from evaluateur import LLMClient

# Ollama exposes an OpenAI-compatible API
ollama_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Required but not used
)

client = LLMClient.from_openai(
    ollama_client,
    model_name="llama3.2",
    provider="ollama",
)
```

### Together AI

```python
from openai import AsyncOpenAI
from evaluateur import LLMClient

together_client = AsyncOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key="your-together-key",
)

client = LLMClient.from_openai(
    together_client,
    model_name="meta-llama/Llama-3-70b-chat-hf",
    provider="together",
)
```

### Groq

```python
from openai import AsyncOpenAI
from evaluateur import LLMClient

groq_client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="your-groq-key",
)

client = LLMClient.from_openai(
    groq_client,
    model_name="llama-3.1-70b-versatile",
    provider="groq",
)
```

## Observability

### LangSmith

```python
from langsmith import wrappers
from openai import AsyncOpenAI
from evaluateur import LLMClient

# Wrap for LangSmith tracing
traced_client = wrappers.wrap_openai(AsyncOpenAI())
client = LLMClient.from_openai(traced_client)
```

### Custom Logging

Access the underlying Instructor client for custom instrumentation:

```python
from evaluateur import LLMClient

client = LLMClient.from_env()

# Access the Instructor client directly
instructor_client = client.instructor_client

# The provider and model are available for logging
print(f"Using {client.provider}/{client.model_name}")
```

## Client Properties

The `LLMClient` exposes these properties:

| Property | Description |
|----------|-------------|
| `provider` | Provider identifier (e.g., "openai", "anthropic") |
| `model_name` | Model name (e.g., "gpt-4o-mini") |
| `instructor_client` | The underlying Instructor client |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key |
| `EVALUATEUR_MODEL_NAME` | `gpt-4o-mini` | Default model name |

## Best Practices

1. **Use `from_env()` for simple cases**: It handles dotenv loading automatically

2. **Pass wrapped clients for observability**: Wrap before passing to `from_openai()`

3. **Set `provider` for clarity**: Even when using OpenAI-compatible APIs, set the provider for better logging

4. **Consider costs**: Option generation and query generation each make LLM calls. Use efficient models for development.

```python
import os
from evaluateur import LLMClient

# Use cheaper model for development
model = "gpt-4o-mini" if os.getenv("ENV") == "dev" else "gpt-4o"
client = LLMClient.from_env(model_name=model)
```
