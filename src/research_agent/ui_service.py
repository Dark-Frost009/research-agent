"""Per-request execution and report exports for the local interface."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from urllib.parse import urlsplit

from research_agent.bootstrap import build_research_application, BootstrapConfigurationError
from research_agent.config import Settings
from research_agent.llm.client import LLMConfigurationError, LLMProviderError, LLMResponseError
from research_agent.main import build_initial_state
from research_agent.models.schemas import Evidence, ResearchReport, Source
from research_agent.tools.web_search import SearchConfigurationError
from pydantic import ValidationError


STAGES = {
    "reserve_initial_iteration": "Planning your research",
    "reserve_follow_up_iteration": "Investigating remaining questions",
    "reserve_search": "Searching the web",
    "reserve_source_fetch": "Reading source pages",
    "reserve_evidence_extraction": "Collecting supporting evidence",
    "reserve_critique": "Checking whether the evidence is sufficient",
    "reserve_finalization": "Writing and verifying the report",
    "assemble_final_report": "Report ready",
}


@dataclass(frozen=True)
class CompletedResearch:
    report: ResearchReport
    evidence: list[Evidence]
    sources: list[Source]
    iterations: int
    searches: int
    warning_count: int

    def citation_sources(self, citation):
        evidence = {item.id: item for item in self.evidence}
        sources = {item.id: item for item in self.sources}
        found = {}
        for evidence_id in citation.evidence_ids:
            item = evidence[evidence_id]
            source = sources[item.source_id]
            found[source.id] = source
        return list(found.values())

    def text_export(self) -> str:
        lines = [self.report.question, "", self.report.content, "", "SOURCES AND SUPPORT"]
        for index, citation in enumerate(self.report.citations, 1):
            lines += ["", f"[{index}] {citation.claim_text}"]
            for source in self.citation_sources(citation):
                url = safe_source_url(source)
                lines.append(f"  {source.title or source.domain}: {url or 'Link unavailable'}")
        if not self.report.citations:
            lines.append("No verified citations were produced.")
        return "\n".join(lines)

    def json_export(self) -> str:
        return json.dumps({
            "report": self.report.model_dump(mode="json"),
            "evidence": [e.model_dump(mode="json") for e in self.evidence],
            "sources": [s.model_dump(mode="json") for s in self.sources],
            "iterations": self.iterations, "searches": self.searches,
        }, indent=2, ensure_ascii=False)


def safe_source_url(source: Source) -> str | None:
    """Only expose ordinary HTTP(S) links, preferring the fetched destination."""
    url = source.final_url or source.url
    try:
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or any(ord(c) < 32 for c in url)):
            return None
        return url
    except ValueError:
        return None


def run_question(question: str, on_progress: Callable[[str], None]) -> CompletedResearch:
    initial = build_initial_state(question)
    # Never cache an application/context: providers and workspace are per request.
    root = Path(__file__).resolve().parents[2]
    application = build_research_application(Settings(_env_file=root / ".env"))
    final_state = None
    for mode, payload in application.graph.stream(
        initial, context=application.context, stream_mode=["updates", "values"],
        config={"recursion_limit": max(100, application.context.budget_policy.limits.max_research_iterations * 20 + 10)},
    ):
        if mode == "updates":
            for node in payload:
                if node in STAGES:
                    on_progress(STAGES[node])
        elif mode == "values":
            final_state = payload
    if final_state is None or not isinstance(final_state.get("final_report"), ResearchReport):
        raise RuntimeError("Research completed without a report.")
    result = CompletedResearch(
        report=final_state["final_report"], evidence=final_state["evidence"],
        sources=final_state["sources"], iterations=final_state["iteration_count"],
        searches=final_state["search_queries_used"], warning_count=len(final_state["errors"]),
    )
    # Do not display a report whose citations cannot be traced to source records.
    for citation in result.report.citations:
        result.citation_sources(citation)
    return result


def friendly_error(exc: Exception) -> str:
    """Do not echo provider responses, configuration values, or credentials."""
    if isinstance(exc, (BootstrapConfigurationError, LLMConfigurationError, SearchConfigurationError, ValidationError)):
        return "The research service needs configuration. Check the provider settings and API keys in the project's .env file."
    if isinstance(exc, LLMProviderError):
        return "The AI service could not complete the request. Check your connection, provider availability, and API allowance, then try again."
    if isinstance(exc, LLMResponseError):
        return "The AI response did not meet the required checks. No unverified report has been released. You can try again."
    return "This research run could not finish. No new report was released. Try again; if it repeats, ask for a diagnostic review."
