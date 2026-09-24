"""Provider-neutral LLM contract for the Research Agent.

The rest of the application depends on LLMClient rather than directly
depending on a specific provider such as Groq, Amazon Bedrock, or OpenAI.

Concrete provider implementations will live elsewhere and implement
this protocol.
"""

from typing import Protocol, TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Base exception for LLM-related failures."""


class LLMConfigurationError(LLMError):
    """Raised when the LLM provider is configured incorrectly."""


class LLMProviderError(LLMError):
    """Raised when the external LLM provider request fails."""


class LLMUnavailableError(LLMProviderError):
    """The provider is temporarily unavailable or overloaded."""


class LLMRateLimitError(LLMProviderError):
    """The provider rejected the request due to quota or rate limits."""


class LLMResponseError(LLMError):
    """Raised when an LLM response cannot satisfy the expected contract."""


class LLMClient(Protocol):
    """Provider-neutral interface used by Research Agent components.

    Implementations are responsible for provider-specific API calls,
    response parsing, and mapping provider failures into the exception
    hierarchy defined in this module.
    """

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate an ordinary text response."""

        ...

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
    ) -> T:
        """Generate and validate a structured Pydantic response."""

        ...