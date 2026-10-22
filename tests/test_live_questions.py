"""Offline tests for the live-evaluation question set (evaluation Phase 2).

These check the data, never the models: that the shipped question file loads,
that every question actually checks something, and that the written-down
expectations catch the known failure in the two real CUBIC exports.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from research_agent.live_metrics import check_expectations
from research_agent.live_questions import (
    QuestionSet,
    load_questions,
)

_ROOT = Path(__file__).resolve().parents[1]
_QUESTIONS = _ROOT / "evals" / "live" / "questions.json"
_FIXTURES = Path(__file__).parent / "fixtures" / "live_runs"

_EXPECTED_IDS = [
    "cubic_vs_reno",
    "rag_basics",
    "rag_limitations",
    "ethernet_mtu",
    "python_313",
    "four_day_week",
    "quantum_tcp_standard",
    "linux_cubic_default",
]


def _shipped() -> QuestionSet:
    return load_questions(_QUESTIONS)


def _question(question_id: str):
    return next(
        q for q in _shipped().questions if q.id == question_id
    )


def _export_with_answer(content: str, outcome: str = "verified") -> dict:
    return {
        "report": {"content": content, "citations": []},
        "evidence": [],
        "sources": [],
        "summary": {"outcome": outcome},
    }


# ---------------------------------------------------------------------------
# The shipped file
# ---------------------------------------------------------------------------


def test_shipped_question_set_loads_with_the_agreed_questions() -> None:
    assert [q.id for q in _shipped().questions] == _EXPECTED_IDS


def test_questions_are_clean_single_lines() -> None:
    for q in _shipped().questions:
        assert q.question == q.question.strip()
        assert "\n" not in q.question
        assert q.question.endswith("?")


def test_exactly_one_question_expects_an_abstention() -> None:
    abstaining = [
        q.id
        for q in _shipped().questions
        if q.expectations.expect_abstain
    ]

    assert abstaining == ["quantum_tcp_standard"]


def test_every_answerable_question_checks_something() -> None:
    for q in _shipped().questions:
        if q.expectations.expect_abstain:
            continue

        assert (
            q.expectations.must_mention
            or q.expectations.pitfalls
        ), q.id


def test_every_question_explains_why_it_is_in_the_set() -> None:
    for q in _shipped().questions:
        assert q.notes.strip(), q.id


# ---------------------------------------------------------------------------
# The written-down expectations against real runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture",
    ["cubic_before_a2.json", "cubic_after_a2.json"],
)
def test_cubic_expectations_catch_the_known_binary_search_error(
    fixture: str,
) -> None:
    export = json.loads(
        (_FIXTURES / fixture).read_text(encoding="utf-8")
    )

    result = check_expectations(
        export,
        _question("cubic_vs_reno").expectations,
    )

    # The answer covers all three facts and still contains the wrong claim.
    assert result.facts_found == result.facts_total == 3
    assert result.pitfalls_flagged == [
        "binary search attributed to CUBIC"
    ]


def test_cubic_expectations_accept_a_correct_answer() -> None:
    export = _export_with_answer(
        "Reno grows its window linearly after a loss. "
        "BIC searched for the window with a binary search. "
        "CUBIC instead uses a cubic function of the time since the "
        "last congestion event."
    )

    result = check_expectations(
        export,
        _question("cubic_vs_reno").expectations,
    )

    assert result.missing_facts == []
    assert result.pitfalls_flagged == []


def test_cubic_expectations_do_not_count_the_word_cubic_alone() -> None:
    # "CUBIC" is in the question itself, so it must not satisfy a fact.
    export = _export_with_answer("CUBIC is a TCP algorithm.")

    result = check_expectations(
        export,
        _question("cubic_vs_reno").expectations,
    )

    assert result.facts_found == 0


@pytest.mark.parametrize(
    "answer",
    [
        "The standard MTU is 1500 bytes.",
        "Ethernet allows 1,500 bytes of payload.",
    ],
)
def test_ethernet_number_matches_common_formats(answer: str) -> None:
    result = check_expectations(
        _export_with_answer(answer),
        _question("ethernet_mtu").expectations,
    )

    assert result.missing_facts == []


@pytest.mark.parametrize(
    "answer",
    [
        "CUBIC became the default in Linux 2.6.19.",
        "It was kernel 2.6.19, in 2006.",
    ],
)
def test_kernel_version_is_found_as_a_whole_token(answer: str) -> None:
    result = check_expectations(
        _export_with_answer(answer),
        _question("linux_cubic_default").expectations,
    )

    assert result.missing_facts == []


def test_kernel_version_does_not_match_inside_a_longer_number() -> None:
    result = check_expectations(
        _export_with_answer("Linux 12.6.195 is not real."),
        _question("linux_cubic_default").expectations,
    )

    assert result.missing_facts == [0]


def test_a_confident_answer_fails_the_quantum_question() -> None:
    result = check_expectations(
        _export_with_answer(
            "The 2031 IEEE standard specified quantum packet framing."
        ),
        _question("quantum_tcp_standard").expectations,
    )

    assert result.abstain_ok is False


def test_a_stated_limitation_passes_the_quantum_question() -> None:
    result = check_expectations(
        _export_with_answer(
            "The available evidence does not establish that such a "
            "standard exists."
        ),
        _question("quantum_tcp_standard").expectations,
    )

    assert result.abstain_ok is True


# ---------------------------------------------------------------------------
# Loader validation
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _one(**overrides) -> dict:
    question = {
        "id": "ok_id",
        "category": "test",
        "question": "A question?",
        "expectations": {},
    }
    question.update(overrides)
    return question


def test_loader_accepts_a_minimal_valid_file(tmp_path) -> None:
    loaded = load_questions(
        _write(tmp_path, {"version": 1, "questions": [_one()]})
    )

    assert loaded.questions[0].id == "ok_id"


def test_loader_rejects_an_unknown_version(tmp_path) -> None:
    with pytest.raises(ValidationError):
        load_questions(
            _write(tmp_path, {"version": 2, "questions": [_one()]})
        )


def test_loader_rejects_an_empty_question_list(tmp_path) -> None:
    with pytest.raises(ValidationError):
        load_questions(
            _write(tmp_path, {"version": 1, "questions": []})
        )


def test_loader_rejects_duplicate_ids(tmp_path) -> None:
    with pytest.raises(ValidationError, match="unique"):
        load_questions(
            _write(
                tmp_path,
                {"version": 1, "questions": [_one(), _one()]},
            )
        )


@pytest.mark.parametrize(
    "bad_id",
    ["Has Space", "UPPER", "dash-id", "", "ünï"],
)
def test_loader_rejects_unsafe_ids(tmp_path, bad_id: str) -> None:
    with pytest.raises(ValidationError):
        load_questions(
            _write(
                tmp_path,
                {"version": 1, "questions": [_one(id=bad_id)]},
            )
        )


def test_loader_rejects_a_blank_question(tmp_path) -> None:
    with pytest.raises(ValidationError):
        load_questions(
            _write(
                tmp_path,
                {"version": 1, "questions": [_one(question="   ")]},
            )
        )


def test_loader_rejects_unknown_fields(tmp_path) -> None:
    with pytest.raises(ValidationError):
        load_questions(
            _write(
                tmp_path,
                {"version": 1, "questions": [_one(surprise=1)]},
            )
        )


def test_loader_rejects_an_invalid_pitfall_regex(tmp_path) -> None:
    bad = _one(
        expectations={
            "pitfalls": [{"name": "bad", "pattern": "(unclosed"}],
        }
    )

    with pytest.raises(ValidationError):
        load_questions(
            _write(tmp_path, {"version": 1, "questions": [bad]})
        )


def test_loader_reports_a_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_questions(tmp_path / "nope.json")
