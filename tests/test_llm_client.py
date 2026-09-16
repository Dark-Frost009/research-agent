"""Tests for the provider-neutral LLM contract.

The protocol itself contains no provider implementation, so these tests
stay intentionally small. They verify that application code can depend
on the LLMClient shape without requiring a real LLM provider.
"""

from pydantic import BaseModel

from research_agent.llm.client import (
    LLMClient,
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
    LLMResponseError,
)


class ExampleStructuredResponse(BaseModel):
    """Small structured model used only for contract tests."""

    answer: str
    confidence: int


class FakeLLMClient:
    """Simple fake implementation of the LLMClient contract."""

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return f"{system_prompt} | {user_prompt}"

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> BaseModel:
        return response_model(
            answer=f"{system_prompt} | {user_prompt}",
            confidence=100,
        )


def _use_text_client(client: LLMClient) -> str:
    """Represent how an application component would use the protocol."""
    return client.generate_text(
        system_prompt="system",
        user_prompt="user",
    )


def _use_structured_client(
    client: LLMClient,
) -> ExampleStructuredResponse:
    """Represent structured-output usage by a future graph node."""
    return client.generate_structured(
        system_prompt="planner system",
        user_prompt="research question",
        response_model=ExampleStructuredResponse,
    )


def test_fake_client_can_be_used_through_text_contract():
    client = FakeLLMClient()

    result = _use_text_client(client)

    assert result == "system | user"


def test_fake_client_can_be_used_through_structured_contract():
    client = FakeLLMClient()

    result = _use_structured_client(client)

    assert isinstance(result, ExampleStructuredResponse)
    assert result.answer == "planner system | research question"
    assert result.confidence == 100


def test_configuration_error_inherits_from_llm_error():
    assert issubclass(
        LLMConfigurationError,
        LLMError,
    )


def test_provider_error_inherits_from_llm_error():
    assert issubclass(
        LLMProviderError,
        LLMError,
    )


def test_response_error_inherits_from_llm_error():
    assert issubclass(
        LLMResponseError,
        LLMError,
    )


def test_llm_errors_are_runtime_errors():
    assert issubclass(
        LLMError,
        RuntimeError,
    )