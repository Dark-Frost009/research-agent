"""Gemini implementation of the provider-neutral LLM contract.

This module owns Gemini-specific SDK interaction and translates provider
responses/errors into the project's generic LLM abstractions.

Application components such as Planner, Synthesizer, and Critic should
depend on LLMClient rather than importing Gemini-specific code directly.
"""

from __future__ import annotations

from typing import Any, TypeVar

from google import genai
from google.genai import errors, types
import httpx
from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout as RequestsTimeout
from pydantic import BaseModel, ValidationError

from research_agent.llm.client import (
    LLMConfigurationError,
    LLMProviderError,
    LLMUnavailableError,
    LLMRateLimitError,
    LLMResponseError,
)


T = TypeVar("T", bound=BaseModel)

# Catch known remote/transport failures only. Unexpected SDK or application
# programming errors must retain their original type and traceback.
_PROVIDER_ERRORS = (errors.APIError, httpx.TransportError, RequestsConnectionError, RequestsTimeout)


class GeminiLLMClient:
    """Concrete LLM client backed by the Google Gemini API."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        clean_model = model.strip()

        if not clean_model:
            raise LLMConfigurationError(
                "A Gemini model name is required."
            )

        self._model = clean_model

        if client is not None:
            self._client = client
            return

        if api_key is None or not api_key.strip():
            raise LLMConfigurationError(
                "A Gemini API key is required when no client is supplied."
            )

        try:
            self._client = genai.Client(
                api_key=api_key.strip(),
            )
        except ValueError as exc:
            raise LLMConfigurationError(
                "Failed to initialize the Gemini client."
            ) from exc

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a non-structured text response."""
        clean_system_prompt, clean_user_prompt = self._validate_prompts(
            system_prompt,
            user_prompt,
        )

        config = types.GenerateContentConfig(system_instruction=clean_system_prompt)
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=clean_user_prompt,
                config=config,
            )
        except errors.UnknownApiResponseError as exc:
            raise LLMResponseError("Gemini returned an unreadable response.") from exc
        except _PROVIDER_ERRORS as exc:
            raise self._provider_error(exc, "text") from exc

        return self._extract_response_text(
            response
        )

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
    ) -> T:
        """Generate JSON and validate it against a Pydantic model."""
        clean_system_prompt, clean_user_prompt = self._validate_prompts(
            system_prompt,
            user_prompt,
        )

        if not isinstance(
            response_model,
            type,
        ) or not issubclass(
            response_model,
            BaseModel,
        ):
            raise TypeError(
                "response_model must be a Pydantic BaseModel subclass."
            )

        response_json_schema = (
            response_model.model_json_schema()
        )

        config = types.GenerateContentConfig(
            system_instruction=clean_system_prompt,
            response_mime_type="application/json",
            response_json_schema=response_json_schema,
        )
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=clean_user_prompt,
                config=config,
            )
        except errors.UnknownApiResponseError as exc:
            raise LLMResponseError("Gemini returned an unreadable response.") from exc
        except _PROVIDER_ERRORS as exc:
            raise self._provider_error(exc, "structured") from exc

        response_text = self._extract_response_text(
            response
        )

        try:
            return response_model.model_validate_json(
                response_text
            )
        except (
            ValidationError,
            ValueError,
            TypeError,
        ) as exc:
            raise LLMResponseError(
                "Gemini returned a response that did not satisfy "
                "the expected structured schema."
            ) from exc

    @staticmethod
    def _provider_error(exc: Exception, operation: str) -> LLMProviderError:
        # Preserve useful categories without exposing provider bodies or keys.
        code = getattr(exc, "code", None)
        if code in (502, 503, 504):
            return LLMUnavailableError("Gemini is temporarily unavailable. Try again later.")
        if code == 429:
            return LLMRateLimitError("Gemini rate or quota limit reached. Check your allowance before retrying.")
        return LLMProviderError(f"Gemini {operation} generation request failed.")

    @staticmethod
    def _validate_prompts(
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, str]:
        """Validate and normalize trusted/user prompt strings."""
        if not isinstance(
            system_prompt,
            str,
        ):
            raise TypeError(
                "system_prompt must be a string."
            )

        if not isinstance(
            user_prompt,
            str,
        ):
            raise TypeError(
                "user_prompt must be a string."
            )

        clean_system_prompt = (
            system_prompt.strip()
        )
        clean_user_prompt = (
            user_prompt.strip()
        )

        if not clean_system_prompt:
            raise ValueError(
                "system_prompt must not be blank."
            )

        if not clean_user_prompt:
            raise ValueError(
                "user_prompt must not be blank."
            )

        return (
            clean_system_prompt,
            clean_user_prompt,
        )

    @staticmethod
    def _extract_response_text(
        response: Any,
    ) -> str:
        """Extract and validate text from a Gemini SDK response."""
        try:
            text = response.text
        except (AttributeError, ValueError) as exc:
            raise LLMResponseError(
                "Gemini response did not expose usable text."
            ) from exc

        if (
            not isinstance(
                text,
                str,
            )
            or not text.strip()
        ):
            raise LLMResponseError(
                "Gemini returned an empty text response."
            )

        return text.strip()
