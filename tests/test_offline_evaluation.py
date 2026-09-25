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
from research_agent.models.schemas import Evidence
from test_production_graph import _isolated_context, _initial_state, _policy, GraphSourceFetcher


DATA = json.loads((Path(__file__).resolve().parents[1] / 'evals' / 'answer_quality.json').read_text(encoding='utf-8'))
CASES = DATA['cases']


class FixtureFetcher(GraphSourceFetcher):
    def __init__(self, case):
        super().__init__()
        self.case = case

    def fetch(self, batch):
        return [replace(item, page=replace(item.page, text=self.case['source_text']))
                for item in super().fetch(batch)]


class FixtureExtractor:
    def __init__(self, case):
        self.case = case

    def extract(self, call):
        excerpt = self.case['excerpt']
        if excerpt is None:
            return []
        assert excerpt in call.fetch_result.page.text
        return [Evidence(id='fixture-evidence', source_id=call.fetch_result.source.id,
                         sub_question_id=call.sub_question.id, excerpt=excerpt)]


class ScriptedFinalizer:
    def __init__(self, case):
        self.case = case
        self.calls = []

    def generate_structured(self, *, system_prompt, user_prompt, response_model):
        self.calls.append(response_model)
        case = self.case
        if response_model is SynthesisResponse:
            return response_model.model_validate(dict(content=case['answer'], claims=[dict(
                claim_text=case['claim'], evidence_support=[dict(
                    evidence_handle=case['handle'], supporting_quote=case['quote'])])]))
        assert response_model is SynthesisVerificationResponse
        return response_model.model_validate(dict(
            coverage_complete=case['coverage'],
            uncovered_factual_claims=[] if case['coverage'] else ['Late fees are always waived.'],
            claim_verdicts=[dict(claim_handle='C1', supported=case['supported'])]))


def test_dataset_is_unique_and_covers_acceptance_and_abstention():
    assert DATA['version'] == 1
    assert len({case['id'] for case in CASES}) == len(CASES)
    assert {case['expected'] for case in CASES} == {'verified', 'verification_rejected', 'no_evidence'}


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['id'])
def test_offline_answer_quality(case, monkeypatch, record_property):
    def forbidden(*args, **kwargs):
        pytest.fail('Offline evaluations must never connect to the network')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    llm = ScriptedFinalizer(case)
    citation_ids = []
    def citation_id():
        citation_ids.append('fixture-citation')
        return citation_ids[-1]
    context = replace(_isolated_context(),
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
    if case['expected'] == 'verified':
        assert report.content == case['answer']
        assert len(report.citations) == 1
        citation = report.citations[0]
        assert citation.claim_text == case['claim']
        assert citation.evidence_ids == ['fixture-evidence']
        evidence = result['evidence'][0]
        assert evidence.source_id in {source.id for source in result['sources']}
        assert case['quote'] in evidence.excerpt
    else:
        assert report.citations == []
        assert citation_ids == []
        assert report.content.strip()
        if case['answer']:
            assert case['answer'] not in report.content
        if case['expected'] == 'no_evidence':
            assert result['evidence'] == []
            assert 'insufficient' in report.content.casefold()
