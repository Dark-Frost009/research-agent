"""End-to-end tests for the compiled production Research Agent graph.

All external side effects are replaced with deterministic test doubles.

These tests exercise the real production graph topology, reducers,
reservation/execution orchestration, critique routing, follow-up iteration,
finalization, and ResearchReport assembly.

They do not claim crash-durable accounting or checkpoint resumability.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from datetime import (
    datetime,
    timezone,
)

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
    CritiqueCall,
)
from research_agent.graph.nodes.evidence import (
    EvidenceCall,
)
from research_agent.graph.nodes.evidence_collector import (
    EvidenceCollector,
)
from research_agent.graph.nodes.planner import (
    Planner,
    PlannerCall,
)
from research_agent.graph.nodes.search import (
    SearchBatch,
    SearchNode,
)
from research_agent.graph.nodes.source_fetcher import (
    SourceFetchBatch,
    SourceFetchResult,
    SourceFetcher,
)
from research_agent.graph.nodes.sources import (
    SourceNode,
)
from research_agent.graph.nodes.synthesis import (
    ClaimEvidenceSupport,
    SynthesizedClaim,
    SynthesisResponse,
    FinalizationCall,
    SynthesisResult,
    Synthesizer,
)
from research_agent.graph.nodes.synthesis_verifier import (
    ClaimSupportVerdict,
    SynthesisVerificationResponse,
)
from research_agent.graph.state import (
    ResearchState,
)
from research_agent.llm.client import LLMProviderError
from research_agent.models.schemas import (
    Citation,
    CritiqueResult,
    Evidence,
    ResearchReport,
    SearchResult,
    SubQuestion,
)
from research_agent.tools.web_extract import (
    FetchedPage,
)


QUESTION = "What is the evidence for this topic?"


def _policy(
    *,
    max_research_iterations: int = 2,
    max_llm_calls_per_run: int = 10,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=BudgetLimits(
            max_research_iterations=(
                max_research_iterations
            ),
            max_search_queries_per_run=10,
            max_search_queries_per_iteration=5,
            max_sources_per_run=10,
            max_source_fetches_per_run=10,
            max_llm_calls_per_run=(
                max_llm_calls_per_run
            ),
            finalization_llm_reserve=2,
        )
    )


def _initial_state() -> ResearchState:
    return {
        "original_question": QUESTION,
        "sub_questions": [],
        "search_results": [],
        "sources": [],
        "evidence": [],
        "draft_content": None,
        "citations": [],
        "critique": None,
        "iteration_count": 0,
        "search_queries_used": 0,
        "source_fetches_used": 0,
        "llm_calls_used": 0,
        "final_report": None,
        "errors": [],
    }


class GraphPlanner(Planner):
    """Deterministic initial-iteration Planner double."""

    def __init__(
        self,
        *,
        planned: list[SubQuestion],
    ) -> None:
        self.planned = planned
        self.provider_calls = 0
        self.entered_calls: list[
            PlannerCall
        ] = []

    def plan(
        self,
        call: PlannerCall,
    ) -> list[SubQuestion]:
        self.entered_calls.append(
            call
        )

        if not call.authorized:
            return []

        self.provider_calls += 1

        return list(
            self.planned
        )


class GraphSearchNode(SearchNode):
    """Create one deterministic result per authorized SubQuestion."""

    def __init__(self) -> None:
        self.provider_calls = 0
        self.result_counter = 0
        self.seen_batches: list[
            SearchBatch
        ] = []

    def search(
        self,
        batch: SearchBatch,
    ) -> list[SearchResult]:
        self.seen_batches.append(
            batch
        )

        results: list[
            SearchResult
        ] = []

        for sub_question in batch.sub_questions:
            self.provider_calls += 1
            self.result_counter += 1

            results.append(
                SearchResult(
                    sub_question_id=(
                        sub_question.id
                    ),
                    query=(
                        sub_question.question
                    ),
                    title=(
                        f"Source "
                        f"{self.result_counter}"
                    ),
                    url=(
                        "https://source-"
                        f"{self.result_counter}"
                        ".example.com/article"
                    ),
                    snippet=(
                        "Deterministic search "
                        "result."
                    ),
                    rank=1,
                    provider="fake",
                )
            )

        return results


class GraphSourceFetcher(SourceFetcher):
    """Deterministic successful source-fetch double."""

    def __init__(self) -> None:
        self.network_calls = 0
        self.seen_batches: list[
            SourceFetchBatch
        ] = []

    def fetch(
        self,
        batch: SourceFetchBatch,
    ) -> list[SourceFetchResult]:
        self.seen_batches.append(
            batch
        )

        results: list[
            SourceFetchResult
        ] = []

        for source in batch.sources:
            self.network_calls += 1

            page = FetchedPage(
                requested_url=source.url,
                final_url=source.url,
                content_type="text/html",
                text=(
                    "PRIVATE RAW PAGE DATA\n"
                    f"Evidence sentence "
                    f"{self.network_calls}."
                ),
            )

            fetched_source = (
                source.model_copy(
                    update={
                        "fetch_status": "success",
                        "fetched_at": datetime(
                            2026,
                            1,
                            1,
                            12,
                            0,
                            tzinfo=timezone.utc,
                        ),
                        "content_type": (
                            "text/html"
                        ),
                        "final_url": source.url,
                    }
                )
            )

            results.append(
                SourceFetchResult(
                    source=fetched_source,
                    page=page,
                    error=None,
                )
            )

        return results


class GraphEvidenceExtractor:
    """Deterministic grounded Evidence extraction double."""

    def __init__(self) -> None:
        self.calls: list[
            EvidenceCall
        ] = []

    def extract(
        self,
        call: EvidenceCall,
    ) -> list[Evidence]:
        self.calls.append(
            call
        )

        page = call.fetch_result.page

        assert page is not None

        excerpt = ""

        for line in page.text.splitlines():
            if line.startswith(
                "Evidence sentence "
            ):
                excerpt = line
                break

        assert excerpt

        return [
            Evidence(
                id=(
                    f"ev-"
                    f"{len(self.calls)}"
                ),
                source_id=(
                    call.fetch_result
                    .source
                    .id
                ),
                sub_question_id=(
                    call.sub_question.id
                ),
                excerpt=excerpt,
                relevance_note=(
                    "Relevant to the "
                    "research question."
                ),
            )
        ]


class GraphCritic(Critic):
    """Return deterministic critique outcomes in sequence."""

    def __init__(
        self,
        *,
        outcomes: list[CritiqueResult],
    ) -> None:
        self.outcomes = list(
            outcomes
        )
        self.entered_calls: list[
            CritiqueCall
        ] = []
        self.provider_llm_calls = 0

    def critique(
        self,
        call: CritiqueCall,
    ) -> CritiqueResult:
        self.entered_calls.append(
            call
        )

        if not call.requires_llm:
            return CritiqueResult(
                sufficient=False,
                gaps=[
                    "No evidence is available."
                ],
                follow_up_questions=[
                    call.original_question
                ],
            )

        if not call.authorized:
            return CritiqueResult(
                sufficient=False,
                gaps=[
                    "Critique budget is "
                    "unavailable."
                ],
                follow_up_questions=[],
            )

        index = (
            self.provider_llm_calls
        )

        if index >= len(
            self.outcomes
        ):
            raise AssertionError(
                "GraphCritic received more "
                "authorized calls than expected."
            )

        self.provider_llm_calls += 1

        return self.outcomes[
            index
        ]


class GraphSynthesizer(Synthesizer):
    """Deterministic protected finalization double."""

    def __init__(self) -> None:
        self.entered_calls: list[
            FinalizationCall
        ] = []
        self.provider_llm_calls = 0

    def synthesize(
        self,
        call: FinalizationCall,
    ) -> SynthesisResult:
        self.entered_calls.append(
            call
        )

        if not call.requires_llm:
            return SynthesisResult(
                content=(
                    "Insufficient evidence "
                    "to produce a grounded "
                    "answer."
                ),
                citations=[],
            )

        if not call.fully_authorized:
            return SynthesisResult(
                content=(
                    "Finalization could not "
                    "run because the atomic "
                    "LLM pair was unavailable."
                ),
                citations=[],
            )

        self.provider_llm_calls += 2

        evidence = list(
            call.evidence
        )

        assert evidence

        content = (
            "The accumulated evidence "
            "supports the final answer."
        )

        citation = Citation(
            id="cit-final",
            claim_text=content,
            evidence_ids=[
                item.id
                for item in evidence
            ],
        )

        return SynthesisResult(
            content=content,
            citations=[
                citation
            ],
        )


def _context(
    *,
    planner: GraphPlanner,
    search_node: GraphSearchNode,
    source_fetcher: GraphSourceFetcher,
    evidence_extractor: GraphEvidenceExtractor,
    critic: Critic,
    synthesizer: Synthesizer,
    budget_policy: BudgetPolicy | None = None,
) -> ResearchGraphContext:
    return ResearchGraphContext(
        budget_policy=(
            budget_policy
            if budget_policy is not None
            else _policy()
        ),
        planner=planner,
        search_node=search_node,
        source_node=SourceNode(),
        source_fetcher=source_fetcher,
        evidence_collector=EvidenceCollector(
            evidence_extractor=(
                evidence_extractor
            )
        ),
        critic=critic,
        synthesizer=synthesizer,
    )


def _initial_sub_question() -> SubQuestion:
    return SubQuestion(
        id="sq-initial",
        question=(
            "What does the initial "
            "evidence show?"
        ),
        rationale=(
            "Establish the initial "
            "evidence base."
        ),
        created_at_iteration=0,
    )


def test_production_graph_completes_sufficient_first_iteration() -> None:
    planner = GraphPlanner(
        planned=[
            _initial_sub_question()
        ]
    )

    search_node = GraphSearchNode()
    source_fetcher = (
        GraphSourceFetcher()
    )
    evidence_extractor = (
        GraphEvidenceExtractor()
    )

    critic = GraphCritic(
        outcomes=[
            CritiqueResult(
                sufficient=True,
                gaps=[],
                follow_up_questions=[],
                reasoning=(
                    "The evidence is "
                    "sufficient."
                ),
            )
        ]
    )

    synthesizer = (
        GraphSynthesizer()
    )

    context = _context(
        planner=planner,
        search_node=search_node,
        source_fetcher=(
            source_fetcher
        ),
        evidence_extractor=(
            evidence_extractor
        ),
        critic=critic,
        synthesizer=synthesizer,
    )

    result = (
        build_research_graph()
        .invoke(
            _initial_state(),
            context=context,
        )
    )

    assert result[
        "iteration_count"
    ] == 1

    assert result[
        "search_queries_used"
    ] == 1

    assert result[
        "source_fetches_used"
    ] == 1

    # Planner 1 + evidence 1 + critique 1
    # + protected finalization pair 2.
    assert result[
        "llm_calls_used"
    ] == 5

    assert planner.provider_calls == 1
    assert search_node.provider_calls == 1
    assert source_fetcher.network_calls == 1

    assert len(
        evidence_extractor.calls
    ) == 1

    assert (
        critic.provider_llm_calls
        == 1
    )

    assert (
        synthesizer.provider_llm_calls
        == 2
    )

    assert len(
        result["sub_questions"]
    ) == 1

    assert len(
        result["search_results"]
    ) == 1

    assert len(
        result["sources"]
    ) == 1

    assert len(
        result["evidence"]
    ) == 1

    assert (
        "PRIVATE RAW PAGE DATA"
        not in repr(result)
    )

    assert isinstance(
        result["final_report"],
        ResearchReport,
    )

    report = result[
        "final_report"
    ]

    assert report is not None

    assert report.question == QUESTION

    assert report.content == (
        "The accumulated evidence "
        "supports the final answer."
    )

    assert report.content == (
        result["draft_content"]
    )

    assert report.citations == (
        result["citations"]
    )

    assert len(
        report.citations
    ) == 1

    assert report.citations[
        0
    ].evidence_ids == [
        result["evidence"][0].id
    ]


def test_production_graph_runs_critique_follow_up_without_replanning() -> None:
    follow_up_question = (
        "What additional evidence "
        "resolves the remaining gap?"
    )

    planner = GraphPlanner(
        planned=[
            _initial_sub_question()
        ]
    )

    search_node = GraphSearchNode()
    source_fetcher = (
        GraphSourceFetcher()
    )
    evidence_extractor = (
        GraphEvidenceExtractor()
    )

    critic = GraphCritic(
        outcomes=[
            CritiqueResult(
                sufficient=False,
                gaps=[
                    "An important evidence "
                    "gap remains."
                ],
                follow_up_questions=[
                    follow_up_question
                ],
                reasoning=(
                    "More research is "
                    "required."
                ),
            ),
            CritiqueResult(
                sufficient=True,
                gaps=[],
                follow_up_questions=[],
                reasoning=(
                    "The follow-up evidence "
                    "resolved the gap."
                ),
            ),
        ]
    )

    synthesizer = (
        GraphSynthesizer()
    )

    context = _context(
        planner=planner,
        search_node=search_node,
        source_fetcher=(
            source_fetcher
        ),
        evidence_extractor=(
            evidence_extractor
        ),
        critic=critic,
        synthesizer=synthesizer,
    )

    result = (
        build_research_graph()
        .invoke(
            _initial_state(),
            context=context,
        )
    )

    # Initial iteration + one critique-driven follow-up.
    assert result[
        "iteration_count"
    ] == 2

    # One query in each iteration.
    assert result[
        "search_queries_used"
    ] == 2

    assert result[
        "source_fetches_used"
    ] == 2

    # Planner       1
    # Evidence      2
    # Critique      2
    # Finalization  2
    # ----------------
    # Total         7
    assert result[
        "llm_calls_used"
    ] == 7

    # Planner must run only for the initial iteration.
    assert planner.provider_calls == 1

    assert len(
        planner.entered_calls
    ) == 1

    # Search executes once for the Planner output and once for the
    # Critic-generated follow-up.
    assert len(
        search_node.seen_batches
    ) == 2

    assert (
        search_node.provider_calls
        == 2
    )

    assert (
        source_fetcher.network_calls
        == 2
    )

    assert len(
        evidence_extractor.calls
    ) == 2

    assert (
        critic.provider_llm_calls
        == 2
    )

    assert len(
        critic.entered_calls
    ) == 2

    assert (
        synthesizer.provider_llm_calls
        == 2
    )

    assert len(
        synthesizer.entered_calls
    ) == 1

    assert len(
        result["sub_questions"]
    ) == 2

    initial_sub_question = (
        result["sub_questions"][0]
    )

    follow_up_sub_question = (
        result["sub_questions"][1]
    )

    assert (
        initial_sub_question
        .created_at_iteration
        == 0
    )

    assert (
        follow_up_sub_question.question
        == follow_up_question
    )

    assert (
        follow_up_sub_question
        .created_at_iteration
        == 1
    )

    assert (
        follow_up_sub_question.id
        != initial_sub_question.id
    )

    first_batch = (
        search_node.seen_batches[0]
    )

    second_batch = (
        search_node.seen_batches[1]
    )

    assert [
        item.id
        for item
        in first_batch.sub_questions
    ] == [
        initial_sub_question.id
    ]

    assert [
        item.id
        for item
        in second_batch.sub_questions
    ] == [
        follow_up_sub_question.id
    ]

    assert len(
        result["search_results"]
    ) == 2

    assert len(
        result["sources"]
    ) == 2

    assert len(
        result["evidence"]
    ) == 2

    assert {
        item.sub_question_id
        for item
        in result["evidence"]
    } == {
        initial_sub_question.id,
        follow_up_sub_question.id,
    }

    assert (
        "PRIVATE RAW PAGE DATA"
        not in repr(result)
    )

    final_critique = result[
        "critique"
    ]

    assert final_critique is not None
    assert final_critique.sufficient is True

    assert isinstance(
        result["final_report"],
        ResearchReport,
    )

    report = result[
        "final_report"
    ]

    assert report is not None

    assert report.question == QUESTION

    assert report.content == (
        result["draft_content"]
    )

    assert report.citations == (
        result["citations"]
    )

    assert len(
        report.citations
    ) == 1

    assert set(
        report.citations[
            0
        ].evidence_ids
    ) == {
        item.id
        for item
        in result["evidence"]
    }

class RejectedSynthesisLLM:
    """Provider responses exercise both real synthesis trust gates."""

    def __init__(self, *, reject_grounding: bool = False) -> None:
        self.calls: list[type] = []
        self.reject_grounding = reject_grounding

    def generate_structured(self, *, system_prompt, user_prompt, response_model):
        self.calls.append(response_model)
        if response_model is SynthesisResponse:
            return SynthesisResponse(
                content="REJECTED DRAFT: This proves the treatment always works.",
                claims=[SynthesizedClaim(
                    claim_text="This proves the treatment always works.",
                    evidence_support=[ClaimEvidenceSupport(
                        evidence_handle="UNKNOWN" if self.reject_grounding else "E1",
                        supporting_quote="Evidence sentence 1.",
                    )],
                )],
            )
        assert response_model is SynthesisVerificationResponse
        return SynthesisVerificationResponse(
            coverage_complete=True,
            claim_verdicts=[ClaimSupportVerdict(
                claim_handle="C1", supported=False,
                reason="The quote does not support the stronger claim.",
            )],
        )


@pytest.mark.parametrize("reject_grounding", [False, True])
def test_production_graph_rejection_returns_safe_report(reject_grounding: bool) -> None:
    llm = RejectedSynthesisLLM(reject_grounding=reject_grounding)
    citation_ids: list[str] = []

    def citation_id():
        citation_ids.append("unexpected-citation")
        return citation_ids[-1]

    context = _context(
        planner=GraphPlanner(planned=[_initial_sub_question()]),
        search_node=GraphSearchNode(),
        source_fetcher=GraphSourceFetcher(),
        evidence_extractor=GraphEvidenceExtractor(),
        critic=GraphCritic(outcomes=[CritiqueResult(
            sufficient=True, gaps=[], follow_up_questions=[],
        )]),
        synthesizer=Synthesizer(llm=llm, citation_id_factory=citation_id),
    )
    states = []
    for state in build_research_graph().stream(
        _initial_state(), context=context, stream_mode="values",
    ):
        states.append(state)

    expected_calls = [SynthesisResponse]
    if not reject_grounding:
        expected_calls.append(SynthesisVerificationResponse)
    assert llm.calls == expected_calls
    assert citation_ids == []
    assert states[-1]["llm_calls_used"] == 5
    assert len(states[-1]["evidence"]) == 1
    report = states[-1]["final_report"]
    assert isinstance(report, ResearchReport)
    assert report.question == QUESTION
    assert report.content == (
        "The available evidence could not be verified strongly "
        "enough to produce a grounded answer."
    )
    assert report.citations == []
    assert states[-1]["draft_content"] == report.content
    assert all(state["draft_content"] in (None, report.content) for state in states)
    assert all(state["citations"] == [] for state in states)
    assert "REJECTED DRAFT" not in repr(states)
    assert context.workspace.finalization_call is None


@pytest.mark.parametrize("resource, limit", [
    ("max_search_queries_per_run", 2),
    ("max_sources_per_run", 2),
    ("max_source_fetches_per_run", 2),
    ("max_llm_calls_per_run", 7),
])
def test_production_graph_exhausts_budget_after_follow_up(resource, limit) -> None:
    policy = _policy(max_research_iterations=3)
    policy = BudgetPolicy(limits=replace(policy.limits, **{resource: limit}))
    planner = GraphPlanner(planned=[_initial_sub_question()])
    search = GraphSearchNode()
    fetcher = GraphSourceFetcher()
    extractor = GraphEvidenceExtractor()
    critic = GraphCritic(outcomes=[
        CritiqueResult(sufficient=False, gaps=["First gap."],
                       follow_up_questions=["What resolves the first gap?"]),
        CritiqueResult(sufficient=False, gaps=["Second gap."],
                       follow_up_questions=["What resolves the second gap?"]),
    ])
    synthesizer = GraphSynthesizer()
    context = _context(
        planner=planner, search_node=search, source_fetcher=fetcher,
        evidence_extractor=extractor, critic=critic,
        synthesizer=synthesizer, budget_policy=policy,
    )
    result = build_research_graph().invoke(_initial_state(), context=context)
    assert result["iteration_count"] == 2
    assert planner.provider_calls == 1
    assert result["search_queries_used"] == search.provider_calls == 2
    assert result["source_fetches_used"] == fetcher.network_calls == 2
    assert len(extractor.calls) == len(result["evidence"]) == 2
    assert critic.provider_llm_calls == 2
    assert result["critique"].sufficient is False
    assert result["critique"].follow_up_questions == ["What resolves the second gap?"]
    assert result["llm_calls_used"] == 7
    assert synthesizer.provider_llm_calls == 2
    assert len(synthesizer.entered_calls) == 1
    assert synthesizer.entered_calls[0].fully_authorized
    assert isinstance(result["final_report"], ResearchReport)
    assert set(result["final_report"].citations[0].evidence_ids) == {
        item.id for item in result["evidence"]
    }
    assert "PRIVATE RAW PAGE DATA" not in repr(result)


def test_production_graph_denied_real_critic_preserves_finalization() -> None:
    class ForbiddenCriticLLM:
        def generate_structured(self, **kwargs):
            pytest.fail("Denied Critic must not call its provider")

    class RecordingCritic(Critic):
        def __init__(self):
            super().__init__(llm=ForbiddenCriticLLM(), max_follow_up_questions=2)
            self.calls = []

        def critique(self, call):
            self.calls.append(call)
            return super().critique(call)

    critic = RecordingCritic()
    synthesizer = GraphSynthesizer()
    search = GraphSearchNode()
    extractor = GraphEvidenceExtractor()
    context = _context(
        planner=GraphPlanner(planned=[_initial_sub_question()]),
        search_node=search, source_fetcher=GraphSourceFetcher(),
        evidence_extractor=extractor, critic=critic,
        synthesizer=synthesizer,
        budget_policy=_policy(max_llm_calls_per_run=4),
    )
    result = build_research_graph().invoke(_initial_state(), context=context)
    assert len(critic.calls) == 1
    assert critic.calls[0].requires_llm
    assert not critic.calls[0].authorized
    assert critic.calls[0].llm_calls_used == 0
    assert result["critique"].sufficient is False
    assert result["critique"].follow_up_questions == []
    assert "budget is unavailable" in result["critique"].gaps[0]
    assert result["iteration_count"] == 1
    assert result["search_queries_used"] == search.provider_calls == 1
    assert result["source_fetches_used"] == 1
    assert len(extractor.calls) == 1
    assert result["llm_calls_used"] == 4
    assert synthesizer.provider_llm_calls == 2
    assert synthesizer.entered_calls[0].fully_authorized
    assert context.workspace.critique_call is None
    assert isinstance(result["final_report"], ResearchReport)
    assert len(result["final_report"].citations) == 1


@pytest.mark.parametrize("failed_stage", [SynthesisResponse, SynthesisVerificationResponse])
@pytest.mark.parametrize("error_type", [LLMProviderError, RuntimeError])
def test_production_graph_finalization_operational_errors_propagate(failed_stage, error_type):
    failure = error_type("Operational failure")

    class FailingLLM(RejectedSynthesisLLM):
        def generate_structured(self, **kwargs):
            if kwargs["response_model"] is failed_stage:
                self.calls.append(failed_stage)
                raise failure
            return super().generate_structured(**kwargs)

    llm = FailingLLM()
    context = _context(
        planner=GraphPlanner(planned=[_initial_sub_question()]),
        search_node=GraphSearchNode(), source_fetcher=GraphSourceFetcher(),
        evidence_extractor=GraphEvidenceExtractor(),
        critic=GraphCritic(outcomes=[CritiqueResult(
            sufficient=True, gaps=[], follow_up_questions=[],
        )]),
        synthesizer=Synthesizer(llm=llm),
    )
    states = []
    with pytest.raises(error_type) as caught:
        for state in build_research_graph().stream(
            _initial_state(), context=context, stream_mode="values",
        ):
            states.append(state)
    assert caught.value is failure
    assert states[-1]["llm_calls_used"] == 5
    assert all(state["final_report"] is None for state in states)
    assert all(state["draft_content"] is None for state in states)
    assert all(state["citations"] == [] for state in states)
    assert "REJECTED DRAFT" not in repr(states)
    assert context.workspace.finalization_call is None
