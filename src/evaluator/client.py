from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import instructor
from dotenv import load_dotenv
from openai import OpenAI

try:
    import dspy  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency at runtime
    dspy = None  # type: ignore[assignment]


DEFAULT_MODEL_NAME = os.getenv("EVALUATOR_MODEL_NAME", "gpt-4o-mini")


@dataclass
class LLMClient:
    """Wrapper around the underlying LLM clients used by the evaluator.

    This class is intentionally small and focused on configuration so that
    higher-level code can remain provider-agnostic.

    Factory Methods
    ---------------
    - ``from_env()`` - Create from environment variables (simplest path)
    - ``from_openai()`` - Create from a custom OpenAI client (e.g., Langfuse-wrapped)
    - ``from_instructor()`` - Create from a pre-configured Instructor client

    Examples
    --------
    Simple usage with environment variables::

        client = LLMClient.from_env()

    Langfuse integration::

        from langfuse.openai import OpenAI as LangfuseOpenAI
        client = LLMClient.from_openai(LangfuseOpenAI())

    Pre-configured Instructor client::

        import instructor
        patched = instructor.from_openai(my_custom_openai_client)
        client = LLMClient.from_instructor(patched)
    """

    provider: str
    model_name: str
    _instructor_client: Any
    _dspy_lm: Any | None = None

    @classmethod
    def from_env(
        cls,
        provider: str = "openai",
        model_name: str | None = None,
        *,
        dspy_lm: Any | None = None,
    ) -> "LLMClient":
        """Create an ``LLMClient`` using environment variables.

        By default this looks up ``OPENAI_API_KEY`` via ``python-dotenv`` and
        configures an Instructor client for OpenAI. Additional providers can be
        supported by extending this method while keeping the Evaluator API
        unchanged.

        Parameters
        ----------
        provider
            The LLM provider to use (e.g., "openai", "anthropic").
        model_name
            The model name to use. Defaults to ``EVALUATOR_MODEL_NAME`` env var
            or "gpt-4o-mini".
        dspy_lm
            Optional pre-configured DSPy language model. If not provided and
            DSPy is available, one will be created automatically.

        Returns
        -------
        LLMClient
            A configured client ready for use with the evaluator.
        """
        load_dotenv()
        model = model_name or DEFAULT_MODEL_NAME

        if provider == "openai":
            # Uses the official OpenAI client; Instructor patches it to add
            # ``response_model`` support.
            base_client = OpenAI()
            inst_client = instructor.from_openai(base_client)
        else:
            # Fallback to provider-agnostic configuration supported by
            # Instructor. The string identifier is delegated to Instructor.
            inst_client = instructor.from_provider(f"{provider}/{model}")

        # DSPy integration is optional; if DSPy is not available the evaluator
        # can still operate in non-DSPy modes.
        resolved_dspy_lm = dspy_lm
        if resolved_dspy_lm is None and dspy is not None:
            try:
                resolved_dspy_lm = dspy.LM(f"{provider}/{model}")
            except Exception:
                resolved_dspy_lm = None

        return cls(
            provider=provider,
            model_name=model,
            _instructor_client=inst_client,
            _dspy_lm=resolved_dspy_lm,
        )

    @classmethod
    def from_openai(
        cls,
        openai_client: Any,
        model_name: str | None = None,
        *,
        provider: str = "openai",
        dspy_lm: Any | None = None,
        **instructor_kwargs: Any,
    ) -> "LLMClient":
        """Create an ``LLMClient`` from a custom OpenAI-compatible client.

        Use this method when you need to pass a pre-configured or wrapped
        OpenAI client, such as one instrumented with Langfuse, Helicone,
        or other observability tools.

        Parameters
        ----------
        openai_client
            An OpenAI-compatible client instance. This can be the standard
            ``openai.OpenAI()`` client, or a wrapped version from tools like
            Langfuse (``langfuse.openai.OpenAI``).
        model_name
            The model name to use. Defaults to ``EVALUATOR_MODEL_NAME`` env var
            or "gpt-4o-mini".
        provider
            Provider identifier for metadata purposes. Defaults to "openai".
        dspy_lm
            Optional pre-configured DSPy language model. If not provided,
            DSPy features will be unavailable for this client.
        **instructor_kwargs
            Additional keyword arguments passed to ``instructor.from_openai()``,
            such as ``mode`` for different extraction modes.

        Returns
        -------
        LLMClient
            A configured client ready for use with the evaluator.

        Examples
        --------
        With Langfuse instrumentation::

            from langfuse.openai import OpenAI as LangfuseOpenAI
            client = LLMClient.from_openai(LangfuseOpenAI())

        With custom base URL::

            from openai import OpenAI
            custom = OpenAI(base_url="https://my-proxy.com/v1")
            client = LLMClient.from_openai(custom)

        With Instructor mode::

            import instructor
            from openai import OpenAI
            client = LLMClient.from_openai(
                OpenAI(),
                mode=instructor.Mode.JSON,
            )
        """
        model = model_name or DEFAULT_MODEL_NAME
        inst_client = instructor.from_openai(openai_client, **instructor_kwargs)

        return cls(
            provider=provider,
            model_name=model,
            _instructor_client=inst_client,
            _dspy_lm=dspy_lm,
        )

    @classmethod
    def from_instructor(
        cls,
        instructor_client: Any,
        model_name: str | None = None,
        *,
        provider: str = "openai",
        dspy_lm: Any | None = None,
    ) -> "LLMClient":
        """Create an ``LLMClient`` from a pre-configured Instructor client.

        Use this method when you have already set up an Instructor client
        with custom configuration, hooks, or patching that you want to
        preserve.

        Parameters
        ----------
        instructor_client
            A pre-configured Instructor client, typically created via
            ``instructor.from_openai()``, ``instructor.from_anthropic()``,
            or ``instructor.from_provider()``.
        model_name
            The model name for metadata purposes. Defaults to
            ``EVALUATOR_MODEL_NAME`` env var or "gpt-4o-mini".
        provider
            Provider identifier for metadata purposes. Defaults to "openai".
        dspy_lm
            Optional pre-configured DSPy language model. If not provided,
            DSPy features will be unavailable for this client.

        Returns
        -------
        LLMClient
            A configured client ready for use with the evaluator.

        Examples
        --------
        With custom Instructor setup::

            import instructor
            from openai import OpenAI

            # Custom client with hooks or special configuration
            inst = instructor.from_openai(OpenAI(), mode=instructor.Mode.TOOLS)
            client = LLMClient.from_instructor(inst, model_name="gpt-4o")

        With Anthropic::

            import instructor
            from anthropic import Anthropic

            inst = instructor.from_anthropic(Anthropic())
            client = LLMClient.from_instructor(
                inst,
                provider="anthropic",
                model_name="claude-3-sonnet",
            )
        """
        model = model_name or DEFAULT_MODEL_NAME

        return cls(
            provider=provider,
            model_name=model,
            _instructor_client=instructor_client,
            _dspy_lm=dspy_lm,
        )

    @property
    def instructor_client(self) -> Any:
        """Return the patched Instructor client."""

        return self._instructor_client

    @property
    def dspy_lm(self) -> Any | None:
        """Return the DSPy language model, if configured."""

        return self._dspy_lm

