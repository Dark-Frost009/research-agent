"""Unit tests for the Tavily web-search adapter.

These tests verify our own search-layer contract:
- input validation
- provider-call behavior
- provider-response normalization
- empty-result handling
- configuration failures
- provider failures
- malformed provider responses

No test makes a real network request.
"""

from unittest.mock import Mock

import pytest

import research_agent.tools.web_search as web_search_module
from research_agent.tools.web_search import (
    SearchConfigurationError,
    SearchProviderError,
    TavilySearchClient,
)


def _client_with_response(response):
    """Return an adapter backed by a fake Tavily client."""
    fake_client = Mock()
    fake_client.search.return_value = response

    adapter = TavilySearchClient(client=fake_client)

    return adapter, fake_client


def _valid_provider_response():
    """Representative normalized-shaped Tavily response."""
    return {
        "results": [
            {
                "title": "First result",
                "url": "https://example.com/first",
                "content": "First search-result snippet.",
            },
            {
                "title": "Second result",
                "url": "https://example.com/second",
                "content": "Second search-result snippet.",
            },
        ]
    }


# ---------------------------------------------------------------------------
# Construction / configuration
# ---------------------------------------------------------------------------


def test_injected_client_does_not_require_api_key():
    fake_client = Mock()

    adapter = TavilySearchClient(client=fake_client)

    assert adapter._client is fake_client


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_missing_api_key_rejected_when_no_client_is_injected(api_key):
    with pytest.raises(SearchConfigurationError):
        TavilySearchClient(api_key=api_key)


def test_api_key_is_trimmed_before_tavily_client_initialization(monkeypatch):
    fake_client = Mock()
    tavily_constructor = Mock(return_value=fake_client)

    monkeypatch.setattr(
        web_search_module,
        "TavilyClient",
        tavily_constructor,
    )

    adapter = TavilySearchClient(api_key="  secret-key  ")

    tavily_constructor.assert_called_once_with(api_key="secret-key")
    assert adapter._client is fake_client


def test_tavily_client_initialization_failure_is_wrapped(monkeypatch):
    def fail_to_initialize(*args, **kwargs):
        raise RuntimeError("SDK initialization failed")

    monkeypatch.setattr(
        web_search_module,
        "TavilyClient",
        fail_to_initialize,
    )

    with pytest.raises(SearchConfigurationError) as exc_info:
        TavilySearchClient(api_key="secret-key")

    assert isinstance(exc_info.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# Successful search / normalization
# ---------------------------------------------------------------------------


def test_search_normalizes_provider_results():
    adapter, _ = _client_with_response(_valid_provider_response())

    results = adapter.search(
        query="AI drug discovery",
        sub_question_id="sq_1",
        max_results=5,
    )

    assert len(results) == 2

    first = results[0]
    assert first.sub_question_id == "sq_1"
    assert first.query == "AI drug discovery"
    assert first.title == "First result"
    assert first.url == "https://example.com/first"
    assert first.snippet == "First search-result snippet."
    assert first.rank == 1
    assert first.provider == "tavily"
    assert first.retrieved_at.utcoffset() is not None

    second = results[1]
    assert second.rank == 2
    assert second.title == "Second result"


def test_search_calls_provider_with_expected_arguments():
    adapter, fake_client = _client_with_response({"results": []})

    adapter.search(
        query="AI drug discovery",
        sub_question_id="sq_1",
        max_results=7,
    )

    fake_client.search.assert_called_once_with(
        query="AI drug discovery",
        max_results=7,
        include_raw_content=False,
    )


def test_search_strips_query_and_sub_question_id():
    adapter, fake_client = _client_with_response({"results": []})

    results = adapter.search(
        query="  AI drug discovery  ",
        sub_question_id="  sq_1  ",
        max_results=5,
    )

    assert results == []

    fake_client.search.assert_called_once_with(
        query="AI drug discovery",
        max_results=5,
        include_raw_content=False,
    )


def test_empty_provider_results_return_empty_list():
    adapter, _ = _client_with_response({"results": []})

    results = adapter.search(
        query="valid query",
        sub_question_id="sq_1",
        max_results=5,
    )

    assert results == []


def test_missing_content_becomes_empty_snippet():
    response = {
        "results": [
            {
                "title": "Example",
                "url": "https://example.com",
            }
        ]
    }

    adapter, _ = _client_with_response(response)

    results = adapter.search(
        query="example",
        sub_question_id="sq_1",
        max_results=5,
    )

    assert results[0].snippet == ""


def test_none_content_becomes_empty_snippet():
    response = {
        "results": [
            {
                "title": "Example",
                "url": "https://example.com",
                "content": None,
            }
        ]
    }

    adapter, _ = _client_with_response(response)

    results = adapter.search(
        query="example",
        sub_question_id="sq_1",
        max_results=5,
    )

    assert results[0].snippet == ""


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", " ", "   "])
def test_blank_query_rejected_before_provider_call(query):
    adapter, fake_client = _client_with_response({"results": []})

    with pytest.raises(ValueError):
        adapter.search(
            query=query,
            sub_question_id="sq_1",
            max_results=5,
        )

    fake_client.search.assert_not_called()


@pytest.mark.parametrize("sub_question_id", ["", " ", "   "])
def test_blank_sub_question_id_rejected_before_provider_call(
    sub_question_id,
):
    adapter, fake_client = _client_with_response({"results": []})

    with pytest.raises(ValueError):
        adapter.search(
            query="valid query",
            sub_question_id=sub_question_id,
            max_results=5,
        )

    fake_client.search.assert_not_called()


@pytest.mark.parametrize("max_results", [0, -1, 21])
def test_max_results_outside_supported_range_rejected(max_results):
    adapter, fake_client = _client_with_response({"results": []})

    with pytest.raises(ValueError):
        adapter.search(
            query="valid query",
            sub_question_id="sq_1",
            max_results=max_results,
        )

    fake_client.search.assert_not_called()


@pytest.mark.parametrize("max_results", [1, 20])
def test_max_results_boundary_values_are_allowed(max_results):
    adapter, fake_client = _client_with_response({"results": []})

    adapter.search(
        query="valid query",
        sub_question_id="sq_1",
        max_results=max_results,
    )

    fake_client.search.assert_called_once()


@pytest.mark.parametrize("max_results", [True, False, 5.0, "5", None])
def test_non_integer_max_results_rejected(max_results):
    adapter, fake_client = _client_with_response({"results": []})

    with pytest.raises(ValueError):
        adapter.search(
            query="valid query",
            sub_question_id="sq_1",
            max_results=max_results,
        )

    fake_client.search.assert_not_called()


# ---------------------------------------------------------------------------
# Provider failures
# ---------------------------------------------------------------------------


def test_provider_exception_is_wrapped_as_search_provider_error():
    fake_client = Mock()
    fake_client.search.side_effect = RuntimeError("provider unavailable")

    adapter = TavilySearchClient(client=fake_client)

    with pytest.raises(SearchProviderError) as exc_info:
        adapter.search(
            query="valid query",
            sub_question_id="sq_1",
            max_results=5,
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# Malformed provider responses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        "unexpected",
        123,
    ],
)
def test_non_dictionary_provider_response_rejected(response):
    adapter, _ = _client_with_response(response)

    with pytest.raises(SearchProviderError):
        adapter.search(
            query="valid query",
            sub_question_id="sq_1",
            max_results=5,
        )


def test_missing_results_field_rejected():
    adapter, _ = _client_with_response({})

    with pytest.raises(SearchProviderError):
        adapter.search(
            query="valid query",
            sub_question_id="sq_1",
            max_results=5,
        )


@pytest.mark.parametrize(
    "results",
    [
        {},
        "not-a-list",
        123,
    ],
)
def test_non_list_results_field_rejected(results):
    adapter, _ = _client_with_response({"results": results})

    with pytest.raises(SearchProviderError):
        adapter.search(
            query="valid query",
            sub_question_id="sq_1",
            max_results=5,
        )


def test_non_dictionary_result_item_rejected():
    adapter, _ = _client_with_response(
        {
            "results": [
                {
                    "title": "Valid",
                    "url": "https://example.com",
                    "content": "Valid snippet",
                },
                "invalid-result",
            ]
        }
    )

    with pytest.raises(SearchProviderError):
        adapter.search(
            query="valid query",
            sub_question_id="sq_1",
            max_results=5,
        )


@pytest.mark.parametrize(
    "provider_result",
    [
        {
            "url": "https://example.com",
            "content": "Missing title.",
        },
        {
            "title": "Missing URL",
            "content": "Missing URL.",
        },
        {
            "title": "",
            "url": "https://example.com",
            "content": "Empty title.",
        },
        {
            "title": "Empty URL",
            "url": "",
            "content": "Empty URL.",
        },
    ],
)
def test_invalid_result_schema_is_wrapped(provider_result):
    adapter, _ = _client_with_response(
        {"results": [provider_result]}
    )

    with pytest.raises(SearchProviderError) as exc_info:
        adapter.search(
            query="valid query",
            sub_question_id="sq_1",
            max_results=5,
        )

    assert exc_info.value.__cause__ is not None