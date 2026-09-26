"""Tests for the production Research Agent composition root.

External provider constructors are replaced with deterministic capturing
doubles. No Gemini, Tavily, HTTP, or other network call is performed.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

import research_agent.bootstrap as bootstrap
from research_agent.config import Settings
from research_agent.graph.budget import (
    BudgetPolicy,
)
from research_agent.graph.nodes.critic import (
    Critic,
)
from research_agent.graph.nodes.evidence import (
    EvidenceExtractor,
)
from research_agent.graph.nodes.evidence_collector import (
    EvidenceCollector,
)
from research_agent.graph.nodes.planner import (
    Planner,
)
from research_agent.graph.nodes.search import (
    SearchNode,
)
from research_agent.graph.nodes.source_fetcher import (
    SourceFetcher,
)
from research_agent.graph.nodes.synthesis import (
    Synthesizer,
)


def _settings(
    **overrides: Any,
) -> Settings:
    values: dict[str, Any] = {
        "llm_provider": "gemini",
        "llm_model": "gemini-test-model",
        "llm_api_key": SecretStr(
            "llm-secret"
        ),
        "search_provider": "tavily",
        "tavily_api_key": SecretStr(
            "search-secret"
        ),
        "max_research_iterations": 3,
        "max_sub_questions": 4,
        "max_search_queries_per_run": 11,
        "max_search_queries_per_iteration": 3,
        "max_search_results_per_query": 6,
        "max_sources_per_run": 13,
        "max_source_fetches_per_run": 9,
        "max_llm_calls_per_run": 30,
        "finalization_llm_reserve": 2,
        "request_timeout_seconds": 7.5,
        "max_response_bytes": 123_456,
        "max_text_chars": 45_678,
        "max_redirects": 4,
    }

    values.update(
        overrides
    )

    return Settings(
        _env_file=None,
        **values
    )


def _patch_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    class CapturingGemini:
        def __init__(
            self,
            *,
            model: str,
            api_key: str | None = None,
            client: Any | None = None,
        ) -> None:
            record = {
                "instance": self,
                "model": model,
                "api_key": api_key,
                "client": client,
            }

            calls.setdefault(
                "gemini_clients",
                [],
            ).append(
                record
            )

            # Preserve the original single-client test interface.
            # When only one Gemini client is built, this is that client.
            # When two clients are built, this points at the most recently
            # constructed one while gemini_clients retains both.
            calls["gemini"] = record

    class CapturingSearchClient:
        def __init__(
            self,
            api_key: str | None = None,
            *,
            client: Any | None = None,
        ) -> None:
            calls["search_client"] = {
                "instance": self,
                "api_key": api_key,
                "client": client,
            }

    class CapturingPageFetcher:
        def __init__(
            self,
            *,
            timeout_seconds: float = 15.0,
            max_response_bytes: int = (
                2 * 1024 * 1024
            ),
            max_text_chars: int = 100_000,
            max_redirects: int = 5,
            session: Any | None = None,
            url_validator: Any = None,
        ) -> None:
            calls["page_fetcher"] = {
                "instance": self,
                "timeout_seconds": (
                    timeout_seconds
                ),
                "max_response_bytes": (
                    max_response_bytes
                ),
                "max_text_chars": (
                    max_text_chars
                ),
                "max_redirects": (
                    max_redirects
                ),
                "session": session,
                "url_validator": url_validator,
            }

    class CapturingBudgetPolicy(
        BudgetPolicy
    ):
        def __init__(
            self,
            *,
            limits,
        ) -> None:
            calls["budget_limits"] = limits

            super().__init__(
                limits=limits
            )

    class CapturingPlanner(
        Planner
    ):
        def __init__(
            self,
            *,
            llm,
            max_sub_questions: int,
            **kwargs: Any,
        ) -> None:
            calls["planner"] = {
                "instance": self,
                "llm": llm,
                "max_sub_questions": (
                    max_sub_questions
                ),
                "extra": kwargs,
            }

    class CapturingSearchNode(
        SearchNode
    ):
        def __init__(
            self,
            *,
            search_client,
            max_results_per_query: int,
        ) -> None:
            calls["search_node"] = {
                "instance": self,
                "search_client": (
                    search_client
                ),
                "max_results_per_query": (
                    max_results_per_query
                ),
            }

    class CapturingSourceFetcher(
        SourceFetcher
    ):
        def __init__(
            self,
            *,
            page_fetcher,
            **kwargs: Any,
        ) -> None:
            calls["source_fetcher"] = {
                "instance": self,
                "page_fetcher": (
                    page_fetcher
                ),
                "extra": kwargs,
            }

    class CapturingEvidenceExtractor(
        EvidenceExtractor
    ):
        def __init__(
            self,
            *,
            llm,
            **kwargs: Any,
        ) -> None:
            calls["evidence_extractor"] = {
                "instance": self,
                "llm": llm,
                "extra": kwargs,
            }

    class CapturingEvidenceCollector(
        EvidenceCollector
    ):
        def __init__(
            self,
            *,
            evidence_extractor,
        ) -> None:
            calls["evidence_collector"] = {
                "instance": self,
                "evidence_extractor": (
                    evidence_extractor
                ),
            }

    class CapturingCritic(
        Critic
    ):
        def __init__(
            self,
            *,
            llm,
            max_follow_up_questions: int,
        ) -> None:
            calls["critic"] = {
                "instance": self,
                "llm": llm,
                "max_follow_up_questions": (
                    max_follow_up_questions
                ),
            }

    class CapturingSynthesizer(
        Synthesizer
    ):
        def __init__(
            self,
            *,
            llm,
            **kwargs: Any,
        ) -> None:
            calls["synthesizer"] = {
                "instance": self,
                "llm": llm,
                "extra": kwargs,
            }

    monkeypatch.setattr(
        bootstrap,
        "GeminiLLMClient",
        CapturingGemini,
    )

    monkeypatch.setattr(
        bootstrap,
        "TavilySearchClient",
        CapturingSearchClient,
    )

    monkeypatch.setattr(
        bootstrap,
        "WebPageFetcher",
        CapturingPageFetcher,
    )

    monkeypatch.setattr(
        bootstrap,
        "BudgetPolicy",
        CapturingBudgetPolicy,
    )

    monkeypatch.setattr(
        bootstrap,
        "Planner",
        CapturingPlanner,
    )

    monkeypatch.setattr(
        bootstrap,
        "SearchNode",
        CapturingSearchNode,
    )

    monkeypatch.setattr(
        bootstrap,
        "SourceFetcher",
        CapturingSourceFetcher,
    )

    monkeypatch.setattr(
        bootstrap,
        "EvidenceExtractor",
        CapturingEvidenceExtractor,
    )

    monkeypatch.setattr(
        bootstrap,
        "EvidenceCollector",
        CapturingEvidenceCollector,
    )

    monkeypatch.setattr(
        bootstrap,
        "Critic",
        CapturingCritic,
    )

    monkeypatch.setattr(
        bootstrap,
        "Synthesizer",
        CapturingSynthesizer,
    )

    return calls


def test_build_research_context_requires_settings() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "settings must be a Settings object"
        ),
    ):
        bootstrap.build_research_context(
            "not-settings"
        )


def test_build_research_context_builds_configured_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_dependencies(
        monkeypatch
    )

    bootstrap.build_research_context(
        _settings()
    )

    assert len(
        calls["gemini_clients"]
    ) == 1

    assert calls["gemini"][
        "model"
    ] == "gemini-test-model"

    assert calls["gemini"][
        "api_key"
    ] == "llm-secret"

    assert calls["search_client"][
        "api_key"
    ] == "search-secret"


def test_role_specific_models_build_two_clients_and_route_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_dependencies(
        monkeypatch
    )

    bootstrap.build_research_context(
        _settings(
            llm_model="legacy-model",
            llm_worker_model="worker-model",
            llm_final_model="final-model",
        )
    )

    gemini_clients = calls[
        "gemini_clients"
    ]

    assert len(
        gemini_clients
    ) == 2

    worker_record = gemini_clients[0]
    final_record = gemini_clients[1]

    assert worker_record[
        "model"
    ] == "worker-model"

    assert final_record[
        "model"
    ] == "final-model"

    assert worker_record[
        "api_key"
    ] == "llm-secret"

    assert final_record[
        "api_key"
    ] == "llm-secret"

    worker_llm = worker_record[
        "instance"
    ]

    final_llm = final_record[
        "instance"
    ]

    assert worker_llm is not final_llm

    assert calls["planner"][
        "llm"
    ] is worker_llm

    assert calls[
        "evidence_extractor"
    ][
        "llm"
    ] is worker_llm

    assert calls["critic"][
        "llm"
    ] is worker_llm

    assert calls["synthesizer"][
        "llm"
    ] is final_llm


def test_worker_model_with_legacy_final_fallback_routes_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_dependencies(
        monkeypatch
    )

    bootstrap.build_research_context(
        _settings(
            llm_model="legacy-final-model",
            llm_worker_model="worker-model",
            llm_final_model=None,
        )
    )

    gemini_clients = calls[
        "gemini_clients"
    ]

    assert len(
        gemini_clients
    ) == 2

    worker_record = gemini_clients[0]
    final_record = gemini_clients[1]

    assert worker_record[
        "model"
    ] == "worker-model"

    assert final_record[
        "model"
    ] == "legacy-final-model"

    worker_llm = worker_record[
        "instance"
    ]

    final_llm = final_record[
        "instance"
    ]

    assert calls["planner"][
        "llm"
    ] is worker_llm

    assert calls[
        "evidence_extractor"
    ][
        "llm"
    ] is worker_llm

    assert calls["critic"][
        "llm"
    ] is worker_llm

    assert calls["synthesizer"][
        "llm"
    ] is final_llm


def test_build_research_context_maps_network_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_dependencies(
        monkeypatch
    )

    bootstrap.build_research_context(
        _settings()
    )

    page_fetcher = calls[
        "page_fetcher"
    ]

    assert page_fetcher[
        "timeout_seconds"
    ] == 7.5

    assert page_fetcher[
        "max_response_bytes"
    ] == 123_456

    assert page_fetcher[
        "max_text_chars"
    ] == 45_678

    assert page_fetcher[
        "max_redirects"
    ] == 4


def test_build_research_context_maps_budget_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_dependencies(
        monkeypatch
    )

    context = (
        bootstrap.build_research_context(
            _settings()
        )
    )

    limits = calls[
        "budget_limits"
    ]

    assert limits.max_research_iterations == 3
    assert limits.max_search_queries_per_run == 11
    assert limits.max_search_queries_per_iteration == 3
    assert limits.max_sources_per_run == 13
    assert limits.max_source_fetches_per_run == 9
    assert limits.max_llm_calls_per_run == 30
    assert limits.finalization_llm_reserve == 2

    assert (
        context.budget_policy
        is not None
    )


def test_build_research_context_maps_node_shaping_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_dependencies(
        monkeypatch
    )

    bootstrap.build_research_context(
        _settings()
    )

    assert calls["planner"][
        "max_sub_questions"
    ] == 4

    assert calls["search_node"][
        "max_results_per_query"
    ] == 6

    assert calls["critic"][
        "max_follow_up_questions"
    ] == 3


def test_legacy_single_model_reuses_one_llm_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_dependencies(
        monkeypatch
    )

    bootstrap.build_research_context(
        _settings()
    )

    assert len(
        calls["gemini_clients"]
    ) == 1

    llm = calls[
        "gemini_clients"
    ][0][
        "instance"
    ]

    assert calls["planner"][
        "llm"
    ] is llm

    assert calls[
        "evidence_extractor"
    ][
        "llm"
    ] is llm

    assert calls["critic"][
        "llm"
    ] is llm

    assert calls["synthesizer"][
        "llm"
    ] is llm


def test_search_and_fetch_nodes_receive_provider_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_dependencies(
        monkeypatch
    )

    bootstrap.build_research_context(
        _settings()
    )

    assert calls[
        "search_node"
    ][
        "search_client"
    ] is calls[
        "search_client"
    ][
        "instance"
    ]

    assert calls[
        "source_fetcher"
    ][
        "page_fetcher"
    ] is calls[
        "page_fetcher"
    ][
        "instance"
    ]


def test_evidence_collector_receives_evidence_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_dependencies(
        monkeypatch
    )

    bootstrap.build_research_context(
        _settings()
    )

    assert calls[
        "evidence_collector"
    ][
        "evidence_extractor"
    ] is calls[
        "evidence_extractor"
    ][
        "instance"
    ]


@pytest.mark.parametrize(
    "provider",
    [
        None,
        "",
        "openai",
        "anthropic",
    ],
)
def test_unsupported_llm_provider_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    provider: str | None,
) -> None:
    _patch_dependencies(
        monkeypatch
    )

    with pytest.raises(
        bootstrap.BootstrapConfigurationError,
        match=(
            "Production currently supports "
            "'gemini'"
        ),
    ):
        bootstrap.build_research_context(
            _settings(
                llm_provider=provider
            )
        )


@pytest.mark.parametrize(
    "model",
    [
        None,
        "",
        "   ",
    ],
)
def test_missing_gemini_worker_model_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    model: str | None,
) -> None:
    _patch_dependencies(
        monkeypatch
    )

    with pytest.raises(
        bootstrap.BootstrapConfigurationError,
        match=(
            "llm_worker_model or llm_model "
            "must be configured"
        ),
    ):
        bootstrap.build_research_context(
            _settings(
                llm_model=model,
                llm_worker_model=None,
                llm_final_model=None,
            )
        )


def test_missing_gemini_final_model_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dependencies(
        monkeypatch
    )

    with pytest.raises(
        bootstrap.BootstrapConfigurationError,
        match=(
            "llm_final_model or llm_model "
            "must be configured"
        ),
    ):
        bootstrap.build_research_context(
            _settings(
                llm_model=None,
                llm_worker_model="worker-model",
                llm_final_model=None,
            )
        )


@pytest.mark.parametrize(
    "provider",
    [
        "",
        "google",
        "bing",
    ],
)
def test_unsupported_search_provider_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    _patch_dependencies(
        monkeypatch
    )

    with pytest.raises(
        bootstrap.BootstrapConfigurationError,
        match=(
            "Production currently supports "
            "'tavily'"
        ),
    ):
        bootstrap.build_research_context(
            _settings(
                search_provider=provider
            )
        )


def test_missing_provider_api_keys_are_passed_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_dependencies(
        monkeypatch
    )

    bootstrap.build_research_context(
        _settings(
            llm_api_key=None,
            tavily_api_key=None,
        )
    )

    assert calls[
        "gemini_clients"
    ][0][
        "api_key"
    ] is None

    assert calls[
        "search_client"
    ][
        "api_key"
    ] is None


def test_build_research_application_returns_graph_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_dependencies(
        monkeypatch
    )

    application = (
        bootstrap.build_research_application(
            _settings()
        )
    )

    assert isinstance(
        application,
        bootstrap.ResearchApplication,
    )

    assert (
        type(
            application.graph
        ).__name__
        == "CompiledStateGraph"
    )

    assert (
        application.context.planner
        is calls["planner"]["instance"]
    )

    assert (
        application.context.search_node
        is calls[
            "search_node"
        ][
            "instance"
        ]
    )

    assert (
        application.context.source_fetcher
        is calls[
            "source_fetcher"
        ][
            "instance"
        ]
    )

    assert (
        application.context.evidence_collector
        is calls[
            "evidence_collector"
        ][
            "instance"
        ]
    )

    assert (
        application.context.critic
        is calls["critic"]["instance"]
    )

    assert (
        application.context.synthesizer
        is calls[
            "synthesizer"
        ][
            "instance"
        ]
    )