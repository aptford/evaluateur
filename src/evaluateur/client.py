from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI

DEFAULT_MODEL_NAME = os.getenv("EVALUATEUR_MODEL_NAME", "gpt-4o-mini")


@dataclass
class LLMClient:
    """Async wrapper around the underlying LLM clients used by the evaluator.

    This class is intentionally small and focused on configuration so that
    higher-level code can remain provider-agnostic. All operations are async.

    Factory Methods
    ---------------
    - ``from_env()`` - Create from environment variables (simplest path)
    - ``from_openai()`` - Create from a custom AsyncOpenAI client
    - ``from_instructor()`` - Create from a pre-configured async Instructor client

    Examples
    --------
    Simple usage with environment variables::

        client = LLMClient.from_env()

    With custom AsyncOpenAI client::

        from openai import AsyncOpenAI
        client = LLMClient.from_openai(AsyncOpenAI())

    Pre-configured Instructor client::

        import instructor
        from openai import AsyncOpenAI
        patched = instructor.from_openai(AsyncOpenAI())
        client = LLMClient.from_instructor(patched)
    """

    provider: str
    model_name: str
    _instructor_client: Any

    @classmethod
    def from_env(
        cls,
        provider: str = "openai",
        model_name: str | None = None,
    ) -> LLMClient:
        """Create an ``LLMClient`` using environment variables.

        By default this looks up ``OPENAI_API_KEY`` via ``python-dotenv`` and
        configures an async Instructor client for OpenAI.

        Parameters
        ----------
        provider
            The LLM provider to use (e.g., "openai", "anthropic").
        model_name
            The model name to use. Defaults to ``EVALUATEUR_MODEL_NAME`` env var
            or "gpt-4o-mini".

        Returns
        -------
        LLMClient
            A configured async client ready for use with the evaluator.
        """
        load_dotenv()
        model = model_name or DEFAULT_MODEL_NAME

        if provider == "openai":
            async_client = AsyncOpenAI()
            inst_client = instructor.from_openai(async_client)
        else:
            inst_client = instructor.from_provider(f"{provider}/{model}")

        return cls(
            provider=provider,
            model_name=model,
            _instructor_client=inst_client,
        )

    @classmethod
    def from_openai(
        cls,
        openai_client: AsyncOpenAI,
        model_name: str | None = None,
        *,
        provider: str = "openai",
        **instructor_kwargs: Any,
    ) -> LLMClient:
        """Create an ``LLMClient`` from a custom AsyncOpenAI client.

        Use this method when you need to pass a pre-configured or wrapped
        AsyncOpenAI client for observability or custom configuration.

        Parameters
        ----------
        openai_client
            An AsyncOpenAI-compatible client instance.
        model_name
            The model name to use. Defaults to ``EVALUATEUR_MODEL_NAME`` env var
            or "gpt-4o-mini".
        provider
            Provider identifier for metadata purposes. Defaults to "openai".
        **instructor_kwargs
            Additional keyword arguments passed to ``instructor.from_openai()``.

        Returns
        -------
        LLMClient
            A configured async client ready for use with the evaluator.

        Examples
        --------
        With custom base URL::

            from openai import AsyncOpenAI
            custom = AsyncOpenAI(base_url="https://my-proxy.com/v1")
            client = LLMClient.from_openai(custom)

        With Instructor mode::

            import instructor
            from openai import AsyncOpenAI
            client = LLMClient.from_openai(
                AsyncOpenAI(),
                mode=instructor.Mode.JSON,
            )
        """
        model = model_name or DEFAULT_MODEL_NAME
        inst_client = instructor.from_openai(openai_client, **instructor_kwargs)

        return cls(
            provider=provider,
            model_name=model,
            _instructor_client=inst_client,
        )

    @classmethod
    def from_instructor(
        cls,
        instructor_client: Any,
        model_name: str | None = None,
        *,
        provider: str = "openai",
    ) -> LLMClient:
        """Create an ``LLMClient`` from a pre-configured async Instructor client.

        Use this method when you have already set up an async Instructor client
        with custom configuration, hooks, or patching.

        Parameters
        ----------
        instructor_client
            A pre-configured async Instructor client, typically created via
            ``instructor.from_openai(AsyncOpenAI())``.
        model_name
            The model name for metadata purposes. Defaults to
            ``EVALUATEUR_MODEL_NAME`` env var or "gpt-4o-mini".
        provider
            Provider identifier for metadata purposes. Defaults to "openai".

        Returns
        -------
        LLMClient
            A configured async client ready for use with the evaluator.

        Examples
        --------
        With custom Instructor setup::

            import instructor
            from openai import AsyncOpenAI

            inst = instructor.from_openai(AsyncOpenAI(), mode=instructor.Mode.TOOLS)
            client = LLMClient.from_instructor(inst, model_name="gpt-4o")
        """
        model = model_name or DEFAULT_MODEL_NAME

        return cls(
            provider=provider,
            model_name=model,
            _instructor_client=instructor_client,
        )

    @property
    def instructor_client(self) -> Any:
        """Return the async Instructor client."""
        return self._instructor_client
