"""Fixed answer-quality scenarios against the real graph with scripted providers."""
from dataclasses import replace
import json
from pathlib import Path
import socket

import pytest

from research_agent.graph.builder import build_research_graph
from research_agent.graph.nodes.evidence_collector import EvidenceCollector
from research_agent.graph.nodes.synthesis import Synthesizer, SynthesisResponse
from research_agent.graph.nodes.synthesis_verifier import SynthesisVerificationResponse
from research_agent.models.schemas import Evidence, SubQuestion
from test_production_graph import _isolated_context, _initial_state, _policy, GraphSourceFetcher, GraphPlanner


DATA = json.loads((Path(__file__).resolve().parents[1] / 'evals' / 'answer_quality.json').read_text(encoding='utf-8'))
CASES = DATA['cases']


class FixtureFetcher(GraphSourceFetcher):
    def __init__(self, case):
        super().__init__()
        self.case = case

    def fetch(self, batch):
        return [replace(item, page=replace(item.page, text=self.case['sources'][_source_index(item.source.url)]['text']))
                for item in super().fetch(batch)]


def _source_index(url):
    return int(url.split('source-')[1].split('.')[0]) - 1


class FixtureExtractor:
    def __init__(self, case):
        self.case = case

    def extract(self, call):
        index = _source_index(call.fetch_result.source.url)
        excerpt = self.case['sources'][index]['excerpt']
        if excerpt is None:
            return []
        assert excerpt in call.fetch_result.page.text
        return [Evidence(id=f'fixture-evidence-{index + 1}', source_id=call.fetch_result.source.id,
                         sub_question_id=call.sub_question.id, excerpt=excerpt)]


class ScriptedFinalizer:
    def __init__(self, case):
        self.case = case
        self.calls = []

    def generate_structured(self, *, system_prompt, user_prompt, response_model):
        self.calls.append(response_model)
        case = self.case
        if response_model is SynthesisResponse:
            return response_model.model_validate(dict(content=case['answer'], claims=case['claims']))
        assert response_model is SynthesisVerificationResponse
        return response_model.model_validate(dict(
            coverage_complete=case['coverage'],
            uncovered_factual_claims=[] if case['coverage'] else ['Late fees are always waived.'],
            claim_verdicts=case['verdicts']))


def test_dataset_is_unique_and_covers_acceptance_and_abstention():
    assert DATA['version'] == 2
    assert len({case['id'] for case in CASES}) == len(CASES)
    assert {case['expected'] for case in CASES} == {'verified', 'verification_rejected', 'no_evidence'}
    for case in CASES:
        assert case['sources']
        assert all(source['excerpt'] is None or source['excerpt'] in source['text'] for source in case['sources'])
        if case['expected'] == 'verified':
            assert len(case['expected_citations']) == len(case['claims'])
            assert all(1 <= index <= len(case['sources']) for indices in case['expected_citations'] for index in indices)
        else:
            assert case['expected_citations'] == []


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['id'])
def test_offline_answer_quality(case, monkeypatch, record_property):
    def forbidden(*args, **kwargs):
        pytest.fail('Offline evaluations must never connect to the network')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    llm = ScriptedFinalizer(case)
    citation_ids = []
    def citation_id():
        citation_ids.append(f'fixture-citation-{len(citation_ids) + 1}')
        return citation_ids[-1]
    planner = GraphPlanner(planned=[SubQuestion(id=f'fixture-question-{index}',
        question=f"{case['question']} Source {index}", rationale='Fixture source', created_at_iteration=0)
        for index in range(1, len(case['sources']) + 1)])
    context = replace(_isolated_context(planner=planner),
                      budget_policy=_policy(max_research_iterations=1),
                      source_fetcher=FixtureFetcher(case),
                      evidence_collector=EvidenceCollector(evidence_extractor=FixtureExtractor(case)),
                      synthesizer=Synthesizer(llm=llm, citation_id_factory=citation_id))
    state = _initial_state()
    state['original_question'] = case['question']
    result = build_research_graph().invoke(state, context=context)
    report = result['final_report']
    record_property('case_id', case['id'])
    record_property('expected_outcome', case['expected'])
    record_property('actual_outcome', result['finalization_outcome'])
    assert result['finalization_outcome'] == case['expected']
    assert report.question == case['question']
    assert len(llm.calls) == case['expected_calls']
    assert len(result['sources']) == len(case['sources'])
    sources = {source.id: source for source in result['sources']}
    evidence_by_id = {item.id: item for item in result['evidence']}
    for item in result['evidence']:
        source_index = _source_index(sources[item.source_id].url)
        assert item.id == f'fixture-evidence-{source_index + 1}'
        assert item.excerpt == case['sources'][source_index]['excerpt']
    if case['expected'] == 'verified':
        assert report.content == case['answer']
        assert len(report.citations) == len(case['claims']) == len(case['expected_citations'])
        assert len({citation.id for citation in report.citations}) == len(report.citations)
        for citation, claim, expected_sources in zip(report.citations, case['claims'], case['expected_citations'], strict=True):
            assert citation.claim_text == claim['claim_text']
            assert citation.evidence_ids == [f'fixture-evidence-{index}' for index in expected_sources]
            for evidence_id, support in zip(citation.evidence_ids, claim['evidence_support'], strict=True):
                assert support['supporting_quote'] in evidence_by_id[evidence_id].excerpt
    else:
        assert report.citations == []
        assert citation_ids == []
        assert report.content.strip()
        if case['answer']:
            assert case['answer'] not in report.content
            for claim in case['claims']:
                assert claim['claim_text'] not in report.content
        if case['expected'] == 'no_evidence':
            assert result['evidence'] == []
            assert 'insufficient' in report.content.casefold()
