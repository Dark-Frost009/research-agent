from dataclasses import replace
import sqlite3

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import research_agent.history as history_module
import research_agent.ui_service as service
from research_agent.history import HistoryStore
from research_agent.comparison_export import comparison_markdown, _literal
from test_ui import APP, completed
from test_incomplete import partial
from test_history_filters import depth
from test_history_outcomes import summary


def test_export_has_both_runs_budget_and_sorted_source_differences():
    first = replace(completed('First question'), depth=depth('Quick'), summary=summary('verified'), issues=('A warning',))
    second = completed('Second question')
    second.sources[0].final_url = 'https://other.example/page'
    text = comparison_markdown('first-id', first, 'second-id', second)
    for expected in ['first-id', 'second-id', 'First question', 'Second question',
                     'Supported answer.', 'Quick', 'A warning', 'AI calls | 0 | 12',
                     'Outcome not recorded', 'Detailed budget usage and limits were not recorded',
                     '0 shared; 1 only in first; 1 only in second', 'different questions',
                     'https://example.com/final', 'https://other.example/page']:
        assert expected in text
    assert text == comparison_markdown('first-id', first, 'second-id', second)


def test_incomplete_export_never_invents_an_answer_or_budget():
    text = comparison_markdown('a', partial('Same'), 'b', partial('Same'))
    assert text.count('No completed answer is available') == 2
    assert text.count('Interrupted run.') == 2
    assert 'different questions' not in text
    assert '1 shared; 0 only in first; 0 only in second' in text
    assert '| Budget used |' not in text


@pytest.mark.parametrize('raw', ['<script>alert(1)</script> ![image](https://example.com/x)',
                                '```\n# forged heading\n``````\n![image](x)',
                                'Unicode: Straße 日本語\n~~~\ntext'])
def test_external_text_cannot_close_literal_fences(raw):
    block = _literal(raw)
    opening, *_, closing = block.splitlines()
    assert opening == closing + 'text'
    assert closing not in raw
    assert raw in block


def test_ui_export_matches_selection_and_does_not_modify_history(tmp_path, monkeypatch):
    store = HistoryStore(tmp_path / 'history.sqlite')
    a, b, c = [store.save(completed(question)) for question in ('First', 'Second', 'Third')]
    before = store.path.read_bytes()
    monkeypatch.setattr(history_module, 'get_history_store', lambda: store)
    monkeypatch.setattr(service, 'run_question', lambda *a, **k: pytest.fail('No provider calls'))
    downloads = []
    original = st.download_button
    def capture(label, data, *args, **kwargs):
        if kwargs.get('key') == 'download_comparison':
            downloads.append(data)
        return original(label, data, *args, **kwargs)
    monkeypatch.setattr(st, 'download_button', capture)
    app = AppTest.from_file(APP, default_timeout=10).run()
    app.toggle(key='compare_runs').set_value(True).run()
    app.selectbox(key='compare_first').set_value(a).run()
    assert downloads == []
    app.selectbox(key='compare_second').set_value(b).run()
    assert not app.exception
    assert downloads[-1] == comparison_markdown(a, store.load(a), b, store.load(b))
    app.selectbox(key='compare_second').set_value(c).run()
    assert downloads[-1] == comparison_markdown(a, store.load(a), c, store.load(c))
    assert store.path.read_bytes() == before
    with sqlite3.connect(store.path) as connection:
        connection.execute('UPDATE reports SET payload=? WHERE id=?', ('broken', c))
    app.run()
    assert not app.exception
    assert not any(item.label == 'Download comparison (.md)' for item in app.get('download_button'))
