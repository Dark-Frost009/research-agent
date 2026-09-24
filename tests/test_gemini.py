"""Tests for the Gemini LLM adapter.

All provider interactions are mocked. These tests must never make a real
network request or consume Gemini API quota.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, ConfigDict

from research_agent.llm.client import (
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseError,
)
from research_agent.llm.gemini import GeminiLLMClient


class ExampleResponse(BaseModel):
    answer: str
    confidence: int


class StrictExampleResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    answer: str
    confidence: int


def _fake_client_with_response(text: object):
    """Create an injected fake Gemini client returning response.text."""
    response = SimpleNamespace(
        text=text
    )

    models = Mock()
    models.generate_content.return_value = (
        response
    )

    client = SimpleNamespace(
        models=models
    )

    return client, models


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "",
        " ",
        "   ",
        "\n",
    ],
)
def test_model_must_not_be_blank(
    model,
):
    with pytest.raises(
        LLMConfigurationError
    ):
        GeminiLLMClient(
            model=model,
            client=Mock(),
        )


@pytest.mark.parametrize(
    "api_key",
    [
        None,
        "",
        " ",
        "   ",
    ],
)
def test_api_key_is_required_when_client_is_not_injected(
    api_key,
):
    with pytest.raises(
        LLMConfigurationError
    ):
        GeminiLLMClient(
            model="gemini-test",
            api_key=api_key,
        )


def test_injected_client_does_not_require_api_key():
    injected_client = Mock()

    client = GeminiLLMClient(
        model="gemini-test",
        client=injected_client,
    )

    assert (
        client._client
        is injected_client
    )


def test_model_name_is_stripped():
    fake_client, models = (
        _fake_client_with_response(
            "hello"
        )
    )

    client = GeminiLLMClient(
        model="  gemini-test  ",
        client=fake_client,
    )

    client.generate_text(
        system_prompt="system",
        user_prompt="user",
    )

    assert (
        models.generate_content
        .call_args.kwargs["model"]
        == "gemini-test"
    )


def test_api_key_is_stripped_before_sdk_initialization(
    monkeypatch,
):
    sdk_client = object()

    constructor = Mock(
        return_value=sdk_client
    )

    monkeypatch.setattr(
        "research_agent.llm.gemini.genai.Client",
        constructor,
    )

    client = GeminiLLMClient(
        model="gemini-test",
        api_key="  secret-key  ",
    )

    constructor.assert_called_once_with(
        api_key="secret-key",
    )

    assert (
        client._client
        is sdk_client
    )


def test_sdk_initialization_failure_is_wrapped(
    monkeypatch,
):
    original_error = RuntimeError(
        "SDK initialization failed"
    )

    constructor = Mock(
        side_effect=original_error
    )

    monkeypatch.setattr(
        "research_agent.llm.gemini.genai.Client",
        constructor,
    )

    with pytest.raises(
        LLMConfigurationError
    ) as exc_info:
        GeminiLLMClient(
            model="gemini-test",
            api_key="secret-key",
        )

    assert (
        exc_info.value.__cause__
        is original_error
    )


# ---------------------------------------------------------------------------
# Prompt validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "system_prompt",
    [
        "",
        " ",
        "   ",
        "\n",
    ],
)
def test_blank_system_prompt_is_rejected(
    system_prompt,
):
    client, _ = (
        _fake_client_with_response(
            "hello"
        )
    )

    gemini = GeminiLLMClient(
        model="gemini-test",
        client=client,
    )

    with pytest.raises(
        ValueError
    ):
        gemini.generate_text(
            system_prompt=system_prompt,
            user_prompt="user",
        )


@pytest.mark.parametrize(
    "user_prompt",
    [
        "",
        " ",
        "   ",
        "\n",
    ],
)
def test_blank_user_prompt_is_rejected(
    user_prompt,
):
    client, _ = (
        _fake_client_with_response(
            "hello"
        )
    )

    gemini = GeminiLLMClient(
        model="gemini-test",
        client=client,
    )

    with pytest.raises(
        ValueError
    ):
        gemini.generate_text(
            system_prompt="system",
            user_prompt=user_prompt,
        )


@pytest.mark.parametrize(
    "system_prompt",
    [
        None,
        123,
        [],
        {},
    ],
)
def test_system_prompt_must_be_string(
    system_prompt,
):
    client, _ = (
        _fake_client_with_response(
            "hello"
        )
    )

    gemini = GeminiLLMClient(
        model="gemini-test",
        client=client,
    )

    with pytest.raises(
        TypeError
    ):
        gemini.generate_text(
            system_prompt=system_prompt,
            user_prompt="user",
        )


@pytest.mark.parametrize(
    "user_prompt",
    [
        None,
        123,
        [],
        {},
    ],
)
def test_user_prompt_must_be_string(
    user_prompt,
):
    client, _ = (
        _fake_client_with_response(
            "hello"
        )
    )

    gemini = GeminiLLMClient(
        model="gemini-test",
        client=client,
    )

    with pytest.raises(
        TypeError
    ):
        gemini.generate_text(
            system_prompt="system",
            user_prompt=user_prompt,
        )


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------


def test_generate_text_returns_stripped_response():
    fake_client, _ = (
        _fake_client_with_response(
            "   Gemini answer   "
        )
    )

    client = GeminiLLMClient(
        model="gemini-test",
        client=fake_client,
    )

    result = client.generate_text(
        system_prompt="system",
        user_prompt="question",
    )

    assert result == "Gemini answer"


def test_generate_text_sends_expected_request(
    monkeypatch,
):
    fake_client, models = (
        _fake_client_with_response(
            "answer"
        )
    )

    config_constructor = Mock(
        side_effect=lambda **kwargs: kwargs
    )

    monkeypatch.setattr(
        (
            "research_agent.llm.gemini."
            "types.GenerateContentConfig"
        ),
        config_constructor,
    )

    client = GeminiLLMClient(
        model="gemini-test",
        client=fake_client,
    )

    client.generate_text(
        system_prompt=(
            "  trusted system  "
        ),
        user_prompt=(
            "  user question  "
        ),
    )

    models.generate_content.assert_called_once_with(
        model="gemini-test",
        contents="user question",
        config={
            "system_instruction": (
                "trusted system"
            ),
        },
    )


def test_text_provider_failure_is_wrapped():
    original_error = RuntimeError(
        "provider failed"
    )

    models = Mock()

    models.generate_content.side_effect = (
        original_error
    )

    fake_client = SimpleNamespace(
        models=models
    )

    client = GeminiLLMClient(
        model="gemini-test",
        client=fake_client,
    )

    with pytest.raises(
        LLMProviderError
    ) as exc_info:
        client.generate_text(
            system_prompt="system",
            user_prompt="user",
        )

    assert (
        exc_info.value.__cause__
        is original_error
    )


# ---------------------------------------------------------------------------
# Response text validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response_text",
    [
        None,
        "",
        " ",
        "   ",
    ],
)
def test_empty_or_missing_text_is_rejected(
    response_text,
):
    fake_client, _ = (
        _fake_client_with_response(
            response_text
        )
    )

    client = GeminiLLMClient(
        model="gemini-test",
        client=fake_client,
    )

    with pytest.raises(
        LLMResponseError
    ):
        client.generate_text(
            system_prompt="system",
            user_prompt="user",
        )


def test_response_without_text_attribute_is_rejected():
    models = Mock()

    models.generate_content.return_value = (
        SimpleNamespace()
    )

    fake_client = SimpleNamespace(
        models=models
    )

    client = GeminiLLMClient(
        model="gemini-test",
        client=fake_client,
    )

    with pytest.raises(
        LLMResponseError
    ):
        client.generate_text(
            system_prompt="system",
            user_prompt="user",
        )


def test_non_string_response_text_is_rejected():
    fake_client, _ = (
        _fake_client_with_response(
            123
        )
    )

    client = GeminiLLMClient(
        model="gemini-test",
        client=fake_client,
    )

    with pytest.raises(
        LLMResponseError
    ):
        client.generate_text(
            system_prompt="system",
            user_prompt="user",
        )


# ---------------------------------------------------------------------------
# Structured generation
# ---------------------------------------------------------------------------


def test_generate_structured_returns_pydantic_model():
    fake_client, _ = (
        _fake_client_with_response(
            (
                '{"answer":"structured answer",'
                '"confidence":90}'
            )
        )
    )

    client = GeminiLLMClient(
        model="gemini-test",
        client=fake_client,
    )

    result = client.generate_structured(
        system_prompt="system",
        user_prompt="question",
        response_model=ExampleResponse,
    )

    assert isinstance(
        result,
        ExampleResponse,
    )

    assert (
        result.answer
        == "structured answer"
    )

    assert result.confidence == 90


def test_generate_structured_sends_json_schema_configuration(
    monkeypatch,
):
    fake_client, models = (
        _fake_client_with_response(
            (
                '{"answer":"ok",'
                '"confidence":100}'
            )
        )
    )

    config_constructor = Mock(
        side_effect=lambda **kwargs: kwargs
    )

    monkeypatch.setattr(
        (
            "research_agent.llm.gemini."
            "types.GenerateContentConfig"
        ),
        config_constructor,
    )

    client = GeminiLLMClient(
        model="gemini-test",
        client=fake_client,
    )

    client.generate_structured(
        system_prompt=(
            "  planner instructions  "
        ),
        user_prompt=(
            "  research question  "
        ),
        response_model=ExampleResponse,
    )

    models.generate_content.assert_called_once_with(
        model="gemini-test",
        contents="research question",
        config={
            "system_instruction": (
                "planner instructions"
            ),
            "response_mime_type": (
                "application/json"
            ),
            "response_json_schema": (
                ExampleResponse
                .model_json_schema()
            ),
        },
    )


def test_generate_structured_passes_strict_json_schema_unchanged(
    monkeypatch,
):
    fake_client, models = (
        _fake_client_with_response(
            (
                '{"answer":"ok",'
                '"confidence":100}'
            )
        )
    )

    config_constructor = Mock(
        side_effect=lambda **kwargs: kwargs
    )

    monkeypatch.setattr(
        (
            "research_agent.llm.gemini."
            "types.GenerateContentConfig"
        ),
        config_constructor,
    )

    client = GeminiLLMClient(
        model="gemini-test",
        client=fake_client,
    )

    expected_schema = (
        StrictExampleResponse
        .model_json_schema()
    )

    assert (
        expected_schema[
            "additionalProperties"
        ]
        is False
    )

    client.generate_structured(
        system_prompt="system",
        user_prompt="user",
        response_model=(
            StrictExampleResponse
        ),
    )

    config = (
        models.generate_content
        .call_args.kwargs["config"]
    )

    assert (
        config[
            "response_json_schema"
        ]
        == expected_schema
    )

    assert (
        "response_schema"
        not in config
    )


@pytest.mark.parametrize(
    "invalid_model",
    [
        None,
        dict,
        str,
        object,
    ],
)
def test_response_model_must_be_pydantic_model(
    invalid_model,
):
    fake_client, _ = (
        _fake_client_with_response(
            "{}"
        )
    )

    client = GeminiLLMClient(
        model="gemini-test",
        client=fake_client,
    )

    with pytest.raises(
        TypeError
    ):
        client.generate_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=invalid_model,
        )


@pytest.mark.parametrize(
    "response_text",
    [
        "not json",
        "{",
        "[]",
        "{}",
        (
            '{"answer":'
            '"missing confidence"}'
        ),
        (
            '{"answer":123,'
            '"confidence":"wrong"}'
        ),
    ],
)
def test_invalid_structured_response_is_wrapped(
    response_text,
):
    fake_client, _ = (
        _fake_client_with_response(
            response_text
        )
    )

    client = GeminiLLMClient(
        model="gemini-test",
        client=fake_client,
    )

    with pytest.raises(
        LLMResponseError
    ) as exc_info:
        client.generate_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=ExampleResponse,
        )

    assert (
        exc_info.value.__cause__
        is not None
    )


def test_structured_provider_failure_is_wrapped():
    original_error = RuntimeError(
        "structured provider failed"
    )

    models = Mock()

    models.generate_content.side_effect = (
        original_error
    )

    fake_client = SimpleNamespace(
        models=models
    )

    client = GeminiLLMClient(
        model="gemini-test",
        client=fake_client,
    )

    with pytest.raises(
        LLMProviderError
    ) as exc_info:
        client.generate_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=ExampleResponse,
        )

    assert (
        exc_info.value.__cause__
        is original_error
    )

@pytest.mark.parametrize("code, expected", [(503, "LLMUnavailableError"), (502, "LLMUnavailableError"), (504, "LLMUnavailableError"), (429, "LLMRateLimitError"), (400, "LLMProviderError")])
@pytest.mark.parametrize("structured", [False, True])
def test_provider_status_is_safe_and_does_not_add_unbudgeted_retries(code, expected, structured):
    original = RuntimeError("PRIVATE_PROVIDER_BODY_AND_KEY")
    original.code = code
    fake, models = _fake_client_with_response("unused")
    models.generate_content.side_effect = original
    client = GeminiLLMClient(model="test", client=fake)
    with pytest.raises(LLMProviderError) as caught:
        if structured:
            client.generate_structured(system_prompt="s", user_prompt="u", response_model=ExampleResponse)
        else:
            client.generate_text(system_prompt="s", user_prompt="u")
    assert type(caught.value).__name__ == expected
    assert caught.value.__cause__ is original
    assert "PRIVATE" not in str(caught.value)
    assert models.generate_content.call_count == 1
