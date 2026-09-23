"""Production dependency composition for the Research Agent.

This module is the application composition root.

It converts Settings into concrete infrastructure and domain services, then
pairs those dependencies with the compiled production LangGraph.

No research run is executed while building the application.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import SecretStr

from research_agent.config import Settings
from research_agent.graph.budget import (
    BudgetLimits,
    BudgetPolicy,
)
from research_agent.graph.builder import (
    build_research_graph,
)
from research_agent.graph.context import (
    ResearchGraphContext,
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
from research_agent.graph.nodes.sources import (
    SourceNode,
)
from research_agent.graph.nodes.synthesis import (
    Synthesizer,
)
from research_agent.llm.gemini import (
    GeminiLLMClient,
)
from research_agent.tools.web_extract import (
    WebPageFetcher,
)
from research_agent.tools.web_search import (
    TavilySearchClient,
)


class BootstrapConfigurationError(ValueError):
    """Raised when production dependency configuration is unsupported."""


@dataclass(frozen=True)
class ResearchApplication:
    """Compiled production graph together with its runtime dependencies."""

    graph: Any
    context: ResearchGraphContext


def _secret_value(
    value: SecretStr | None,
) -> str | None:
    """Return a secret's raw value without exposing it through repr/logging."""

    if value is None:
        return None

    return value.get_secret_value()


def _build_budget_policy(
    settings: Settings,
) -> BudgetPolicy:
    """Build the whole-run budget policy from validated Settings."""

    return BudgetPolicy(
        limits=BudgetLimits(
            max_research_iterations=(
                settings.max_research_iterations
            ),
            max_search_queries_per_run=(
                settings.max_search_queries_per_run
            ),
            max_search_queries_per_iteration=(
                settings.max_search_queries_per_iteration
            ),
            max_sources_per_run=(
                settings.max_sources_per_run
            ),
            max_source_fetches_per_run=(
                settings.max_source_fetches_per_run
            ),
            max_llm_calls_per_run=(
                settings.max_llm_calls_per_run
            ),
            finalization_llm_reserve=(
                settings.finalization_llm_reserve
            ),
        )
    )


def _build_llm(
    settings: Settings,
) -> GeminiLLMClient:
    """Build the configured production LLM client."""

    provider = (
        settings.llm_provider or ""
    ).strip().lower()

    if provider != "gemini":
        raise BootstrapConfigurationError(
            "Unsupported LLM provider. "
            "Production currently supports 'gemini'."
        )

    model = (
        settings.llm_model or ""
    ).strip()

    if not model:
        raise BootstrapConfigurationError(
            "llm_model must be configured for Gemini."
        )

    return GeminiLLMClient(
        model=model,
        api_key=_secret_value(
            settings.llm_api_key
        ),
    )


def _build_search_client(
    settings: Settings,
) -> TavilySearchClient:
    """Build the configured production search provider."""

    provider = (
        settings.search_provider or ""
    ).strip().lower()

    if provider != "tavily":
        raise BootstrapConfigurationError(
            "Unsupported search provider. "
            "Production currently supports 'tavily'."
        )

    return TavilySearchClient(
        api_key=_secret_value(
            settings.tavily_api_key
        )
    )


def _build_page_fetcher(
    settings: Settings,
) -> WebPageFetcher:
    """Build the SSRF-protected production webpage fetcher."""

    return WebPageFetcher(
        timeout_seconds=(
            settings.request_timeout_seconds
        ),
        max_response_bytes=(
            settings.max_response_bytes
        ),
        max_text_chars=(
            settings.max_text_chars
        ),
        max_redirects=(
            settings.max_redirects
        ),
    )


def build_research_context(
    settings: Settings,
) -> ResearchGraphContext:
    """Compose every concrete dependency required by the graph."""

    if not isinstance(
        settings,
        Settings,
    ):
        raise TypeError(
            "settings must be a Settings object."
        )

    llm = _build_llm(
        settings
    )

    search_client = _build_search_client(
        settings
    )

    page_fetcher = _build_page_fetcher(
        settings
    )

    budget_policy = _build_budget_policy(
        settings
    )

    planner = Planner(
        llm=llm,
        max_sub_questions=(
            settings.max_sub_questions
        ),
    )

    search_node = SearchNode(
        search_client=search_client,
        max_results_per_query=(
            settings.max_search_results_per_query
        ),
    )

    source_node = SourceNode()

    source_fetcher = SourceFetcher(
        page_fetcher=page_fetcher
    )

    evidence_extractor = (
        EvidenceExtractor(
            llm=llm
        )
    )

    evidence_collector = (
        EvidenceCollector(
            evidence_extractor=(
                evidence_extractor
            )
        )
    )

    critic = Critic(
        llm=llm,
        # Critic follow-ups become next-iteration search queries, so
        # their shaping ceiling is the per-iteration search-query limit.
        max_follow_up_questions=(
            settings.max_search_queries_per_iteration
        ),
    )

    synthesizer = Synthesizer(
        llm=llm
    )

    return ResearchGraphContext(
        budget_policy=budget_policy,
        planner=planner,
        search_node=search_node,
        source_node=source_node,
        source_fetcher=source_fetcher,
        evidence_collector=(
            evidence_collector
        ),
        critic=critic,
        synthesizer=synthesizer,
    )


def build_research_application(
    settings: Settings | None = None,
) -> ResearchApplication:
    """Build the complete production Research Agent application.

    When ``settings`` is omitted, Settings loads configuration using its normal
    environment/.env behavior.

    This function constructs dependencies and compiles the graph only. It does
    not execute a research request or contact external providers.
    """

    resolved_settings = (
        settings
        if settings is not None
        else Settings()
    )

    context = build_research_context(
        resolved_settings
    )

    graph = build_research_graph()

    return ResearchApplication(
        graph=graph,
        context=context,
    )