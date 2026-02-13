from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from evaluateur.client import LLMClient, resolve_client
from evaluateur.queries.models import GeneratedTuple
from evaluateur.tuples import AITupleGenerator


class SimpleOptions(BaseModel):
    """Options model for testing tuple generation."""

    payer: list[str] = ["Cigna", "Aetna", "UnitedHealth"]
    age: list[str] = ["adult", "pediatric", "geriatric"]


def _make_mock_client(provider: str | None = "openai") -> LLMClient:
    """Create a mock LLM client for unit tests."""
    mock_instructor = MagicMock()
    mock_result = MagicMock()
    mock_result.tuples = [
        MagicMock(model_dump=lambda: {"payer": "Cigna", "age": "adult"}),
        MagicMock(model_dump=lambda: {"payer": "Aetna", "age": "pediatric"}),
    ]
    mock_instructor.chat.completions.create = AsyncMock(return_value=mock_result)
    return LLMClient(
        instructor_client=mock_instructor,
        model_name="test-model",
        provider=provider,
    )


@pytest.fixture
def mock_client() -> LLMClient:
    """Mock LLM client with OpenAI provider."""
    return _make_mock_client(provider="openai")


@pytest.fixture
def real_client() -> LLMClient | None:
    """Create a real LLM client if credentials are available."""
    try:
        # Try to create a real client - will fail if no credentials
        return resolve_client()
    except Exception:
        return None


@pytest.mark.asyncio
async def test_ai_generator_accepts_temperature_parameter(mock_client: LLMClient) -> None:
    """Test that AITupleGenerator accepts temperature parameter."""
    gen = AITupleGenerator(mock_client)
    opts = SimpleOptions()

    # Should not raise an error
    tuples = [t async for t in gen.generate(opts, count=2, seed=1, temperature=0.7)]
    assert len(tuples) == 2


@pytest.mark.asyncio
async def test_ai_generator_passes_temperature_to_llm(mock_client: LLMClient) -> None:
    """Test that temperature is passed to the LLM API call."""
    gen = AITupleGenerator(mock_client)
    opts = SimpleOptions()

    _ = [t async for t in gen.generate(opts, count=2, seed=1, temperature=0.8)]

    # Verify the temperature was passed to the API
    call_kwargs = mock_client.instructor_client.chat.completions.create.call_args[1]
    assert "temperature" in call_kwargs
    assert call_kwargs["temperature"] == 0.8


@pytest.mark.asyncio
async def test_ai_generator_passes_seed_to_openai(mock_client: LLMClient) -> None:
    """Test that seed is passed to the LLM API call for OpenAI."""
    gen = AITupleGenerator(mock_client)
    opts = SimpleOptions()

    _ = [t async for t in gen.generate(opts, count=2, seed=42, temperature=0.5)]

    # Verify the seed was passed to the API
    call_kwargs = mock_client.instructor_client.chat.completions.create.call_args[1]
    assert "seed" in call_kwargs
    assert call_kwargs["seed"] == 42


@pytest.mark.asyncio
async def test_ai_generator_omits_seed_for_anthropic() -> None:
    """Test that seed is NOT passed to the API for Anthropic."""
    client = _make_mock_client(provider="anthropic")
    gen = AITupleGenerator(client)
    opts = SimpleOptions()

    _ = [t async for t in gen.generate(opts, count=2, seed=42, temperature=0.5)]

    call_kwargs = client.instructor_client.chat.completions.create.call_args[1]
    assert "seed" not in call_kwargs
    assert call_kwargs["temperature"] == 0.5


@pytest.mark.asyncio
async def test_ai_generator_omits_seed_for_unknown_provider() -> None:
    """Test that seed is NOT passed when provider is None (pre-configured client)."""
    client = _make_mock_client(provider=None)
    gen = AITupleGenerator(client)
    opts = SimpleOptions()

    _ = [t async for t in gen.generate(opts, count=2, seed=42, temperature=0.5)]

    call_kwargs = client.instructor_client.chat.completions.create.call_args[1]
    assert "seed" not in call_kwargs


@pytest.mark.asyncio
async def test_ai_generator_uses_default_temperature(mock_client: LLMClient) -> None:
    """Test that default temperature is 0.5 when not specified."""
    gen = AITupleGenerator(mock_client)
    opts = SimpleOptions()

    _ = [t async for t in gen.generate(opts, count=2, seed=1)]

    # Verify default temperature was passed
    call_kwargs = mock_client.instructor_client.chat.completions.create.call_args[1]
    assert call_kwargs["temperature"] == 0.5


@pytest.mark.asyncio
async def test_ai_generator_instructions_reach_system_message(mock_client: LLMClient) -> None:
    """Test that instructions are included in the LLM system message."""
    gen = AITupleGenerator(mock_client)
    opts = SimpleOptions()

    _ = [
        t
        async for t in gen.generate(
            opts, count=2, seed=1, instructions="Focus on edge cases."
        )
    ]

    call_kwargs = mock_client.instructor_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    system_msg = messages[0]["content"]
    assert "<instructions>" in system_msg
    assert "Focus on edge cases." in system_msg
    assert "</instructions>" in system_msg


@pytest.mark.asyncio
async def test_ai_generator_no_instructions_tag_when_none(mock_client: LLMClient) -> None:
    """Test that no <instructions> tag appears when instructions are None."""
    gen = AITupleGenerator(mock_client)
    opts = SimpleOptions()

    _ = [t async for t in gen.generate(opts, count=2, seed=1)]

    call_kwargs = mock_client.instructor_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    system_msg = messages[0]["content"]
    assert "<instructions>" not in system_msg


# Integration tests that require real LLM API access
@pytest.mark.env
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires LLM API credentials",
)
@pytest.mark.asyncio
async def test_ai_tuples_different_seeds_produce_different_results(
    real_client: LLMClient,
) -> None:
    """Different seeds should produce noticeably different tuple sets."""
    if real_client is None:
        pytest.skip("No LLM client available")

    gen = AITupleGenerator(real_client)
    opts = SimpleOptions()

    tuples1 = [t async for t in gen.generate(opts, count=5, seed=1, temperature=0.5)]
    tuples2 = [
        t async for t in gen.generate(opts, count=5, seed=999, temperature=0.5)
    ]

    # Extract values for comparison
    values1 = [t.values for t in tuples1]
    values2 = [t.values for t in tuples2]

    assert (
        values1 != values2
    ), "Different seeds should produce different results"


@pytest.mark.env
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OpenAI API for seed reproducibility",
)
@pytest.mark.asyncio
async def test_ai_tuples_same_seed_reproducibility_openai() -> None:
    """Same seed with OpenAI should produce highly similar results."""
    client = resolve_client(llm="openai/gpt-4o-mini")
    gen = AITupleGenerator(client)
    opts = SimpleOptions()

    tuples1 = [t async for t in gen.generate(opts, count=5, seed=42, temperature=0.3)]
    tuples2 = [t async for t in gen.generate(opts, count=5, seed=42, temperature=0.3)]

    # Extract values for comparison
    values1 = [t.values for t in tuples1]
    values2 = [t.values for t in tuples2]

    # Check overlap - OpenAI seed should give high reproducibility
    # We'll check if at least 60% overlap (allowing for some variation)
    overlap = sum(1 for v in values1 if v in values2)
    overlap_pct = overlap / len(values1) * 100

    assert (
        overlap_pct >= 60
    ), f"Expected >=60% overlap with same seed, got {overlap_pct:.1f}%"


@pytest.mark.env
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires LLM API credentials",
)
@pytest.mark.asyncio
async def test_ai_tuples_temperature_affects_output(real_client: LLMClient) -> None:
    """Temperature parameter should affect output diversity."""
    if real_client is None:
        pytest.skip("No LLM client available")

    gen = AITupleGenerator(real_client)
    opts = SimpleOptions()

    # Same seed, different temperatures should produce different results
    low_temp = [t async for t in gen.generate(opts, count=5, seed=1, temperature=0.1)]
    high_temp = [t async for t in gen.generate(opts, count=5, seed=1, temperature=1.5)]

    # Extract values for comparison
    values_low = [t.values for t in low_temp]
    values_high = [t.values for t in high_temp]

    # Results should differ (can't guarantee every tuple is different, but sets should differ)
    assert (
        values_low != values_high
    ), "Different temperatures should produce different results"
