# LLMClient

Provider-agnostic wrapper for LLM clients used by the evaluator.

## Overview

`LLMClient` wraps Instructor clients to provide a consistent interface across providers. Create clients using factory methods:

```python
from evaluateur import LLMClient

# From environment variables (simplest)
client = LLMClient.from_env()

# From custom OpenAI client
client = LLMClient.from_openai(AsyncOpenAI())

# From pre-configured Instructor client
client = LLMClient.from_instructor(instructor_client)
```

## Class Reference

::: evaluateur.LLMClient
    options:
      show_source: true
      members:
        - from_env
        - from_openai
        - from_instructor
        - instructor_client

## Factory Methods

### `from_env()`

Create a client from environment variables.

```python
@classmethod
def from_env(
    cls,
    provider: str = "openai",
    model_name: str | None = None,
) -> LLMClient
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `provider` | `str` | `"openai"` | LLM provider name |
| `model_name` | `str | None` | `None` | Model name (defaults to `EVALUATEUR_MODEL_NAME` or `gpt-4o-mini`) |

**Returns:** Configured `LLMClient` instance.

**Environment Variables:**

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (loaded via python-dotenv) |
| `EVALUATEUR_MODEL_NAME` | Default model name |

**Example:**

```python
from evaluateur import LLMClient

# OpenAI with defaults
client = LLMClient.from_env()

# Custom model
client = LLMClient.from_env(model_name="gpt-4o")

# Other provider
client = LLMClient.from_env(provider="anthropic", model_name="claude-3-opus")
```

---

### `from_openai()`

Create a client from a custom AsyncOpenAI instance.

```python
@classmethod
def from_openai(
    cls,
    openai_client: AsyncOpenAI,
    model_name: str | None = None,
    *,
    provider: str = "openai",
    **instructor_kwargs: Any,
) -> LLMClient
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `openai_client` | `AsyncOpenAI` | - | AsyncOpenAI-compatible client |
| `model_name` | `str | None` | `None` | Model name for metadata |
| `provider` | `str` | `"openai"` | Provider identifier |
| `**instructor_kwargs` | `Any` | - | Passed to `instructor.from_openai()` |

**Returns:** Configured `LLMClient` instance.

**Example:**

```python
from openai import AsyncOpenAI
from evaluateur import LLMClient

# Custom base URL
client = LLMClient.from_openai(
    AsyncOpenAI(base_url="https://my-proxy.com/v1"),
    model_name="gpt-4o",
)

# With Instructor mode
import instructor
client = LLMClient.from_openai(
    AsyncOpenAI(),
    mode=instructor.Mode.JSON,
)
```

---

### `from_instructor()`

Create a client from a pre-configured Instructor client.

```python
@classmethod
def from_instructor(
    cls,
    instructor_client: Any,
    model_name: str | None = None,
    *,
    provider: str = "openai",
) -> LLMClient
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `instructor_client` | `Any` | - | Pre-configured async Instructor client |
| `model_name` | `str | None` | `None` | Model name for metadata |
| `provider` | `str` | `"openai"` | Provider identifier |

**Returns:** Configured `LLMClient` instance.

**Example:**

```python
import instructor
from openai import AsyncOpenAI
from evaluateur import LLMClient

# Custom Instructor setup
inst = instructor.from_openai(
    AsyncOpenAI(),
    mode=instructor.Mode.TOOLS,
)
client = LLMClient.from_instructor(inst, model_name="gpt-4o")
```

## Properties

### `instructor_client`

Returns the underlying async Instructor client for direct access.

```python
@property
def instructor_client(self) -> Any
```

**Example:**

```python
client = LLMClient.from_env()
inst = client.instructor_client

# Use directly with Instructor
result = await inst.chat.completions.create(
    model=client.model_name,
    response_model=MyModel,
    messages=[...],
)
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `provider` | `str` | Provider identifier (e.g., "openai") |
| `model_name` | `str` | Model name (e.g., "gpt-4o-mini") |

## Provider Examples

### Anthropic

```python
import instructor
from anthropic import AsyncAnthropic
from evaluateur import LLMClient

client = LLMClient.from_instructor(
    instructor.from_anthropic(AsyncAnthropic()),
    model_name="claude-3-5-sonnet-latest",
    provider="anthropic",
)
```

### Azure OpenAI

```python
from openai import AsyncAzureOpenAI
from evaluateur import LLMClient

client = LLMClient.from_openai(
    AsyncAzureOpenAI(
        api_key="your-key",
        api_version="2024-02-01",
        azure_endpoint="https://your-resource.openai.azure.com",
    ),
    model_name="your-deployment",
    provider="azure",
)
```

### Local (Ollama)

```python
from openai import AsyncOpenAI
from evaluateur import LLMClient

client = LLMClient.from_openai(
    AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    model_name="llama3.2",
    provider="ollama",
)
```

## See Also

- [Evaluator](evaluator.md) - Main evaluator class
- [Provider Configuration](../guides/provider-configuration.md) - Detailed provider setup
