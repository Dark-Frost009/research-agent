"""Deterministic metrics for one live research run.

This module belongs to the live-evaluation phase, not to the normal software
tests. It reads the JSON file the app offers as "Download evidence (.json)"
and turns it into numbers that can be compared between code versions:

- ``compute_run_metrics``: facts about the run itself (did every citation
  resolve, how many different sources were cited, how many claims lean on the
  same excerpt, how many page reads failed, how much budget was used).
- ``check_expectations``: per-question checks written down in advance (facts
  the answer must mention, known wrong statements to flag, and whether the
  question has no answer and the system should abstain).

Everything here is pure Python. It makes no LLM call, no network call, and
reads no settings or secrets, so it is safe to unit test and to run on any
saved export. It deliberately does not judge whether an answer is *true*.
That needs a human; these numbers only show where to look.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class _Frozen(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class Budget(_Frozen):
    used: int = Field(ge=0)
    limit: int = Field(ge=0)


class RunMetrics(_Frozen):
    """Objective facts about one run. Ratios are None when undefined."""

    outcome: str
    end_reason: str | None

    # Citation chain: citation -> evidence -> source.
    claims: int
    broken_links: int
    chain_valid: bool

    # Source diversity.
    sources_with_evidence: int
    sources_cited: int
    source_coverage: float | None

    # Claims that stand on exactly the same excerpts as an earlier claim.
    duplicate_claims: int
    claim_duplication_ratio: float | None

    # Evidence yield. "Distinct" ignores the same excerpt being extracted
    # twice from one source (for example once per sub-question).
    evidence_count: int
    distinct_evidence_count: int
    distinct_evidence_cited: int
    evidence_per_read_source: float | None

    # Fetching.
    sources_total: int
    sources_read: int
    sources_failed: int
    fetch_failure_rate: float | None

    issues: int
    budgets: dict[str, Budget]


class Pitfall(_Frozen):
    """A known wrong statement to flag, matched sentence by sentence.

    A sentence is flagged when ``pattern`` matches it and ``unless`` (if
    given) does not match that same sentence. Example: flag "binary search"
    in a CUBIC answer unless the same sentence names BIC, where it belongs.
    ``pattern`` ignores case; ``unless`` is case-sensitive, so ``\\bBIC\\b``
    does not match inside "CUBIC".
    """

    name: str = Field(min_length=1)
    pattern: str = Field(min_length=1)
    unless: str | None = None

    @field_validator("pattern", "unless")
    @classmethod
    def _must_compile(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                re.compile(value)
            except re.error as exc:
                # re.error is not a ValueError, so pydantic would let it
                # escape instead of reporting a validation error.
                raise ValueError(
                    f"invalid regular expression: {exc}"
                ) from exc
        return value


class Expectations(_Frozen):
    """What a question's answer should and should not contain."""

    # Each inner list is ONE fact; any alternative phrasing satisfies it.
    must_mention: list[list[str]] = Field(default_factory=list)
    pitfalls: list[Pitfall] = Field(default_factory=list)
    # True for a question with no real answer: the system should abstain or
    # clearly state a limitation instead of presenting a confident answer.
    expect_abstain: bool = False

    @field_validator("must_mention")
    @classmethod
    def _facts_need_phrases(
        cls,
        value: list[list[str]],
    ) -> list[list[str]]:
        for fact in value:
            if not fact or any(
                not isinstance(phrase, str) or not phrase.strip()
                for phrase in fact
            ):
                raise ValueError(
                    "each must_mention fact needs at least one "
                    "non-blank phrase."
                )
        return value


class ExpectationResult(_Frozen):
    facts_total: int
    facts_found: int
    # Zero-based indexes into Expectations.must_mention.
    missing_facts: list[int]
    pitfalls_flagged: list[str]
    # None when the question does not expect an abstention.
    abstain_ok: bool | None


# Phrases that count as "the answer says it could not establish this".
_LIMITATION = re.compile(
    r"insufficient"
    r"|(?:does|do) not (?:establish|contain|provide)"
    r"|not (?:been )?(?:established|supported|found)"
    r"|no (?:evidence|information|reliable)"
    r"|could not (?:be )?(?:verified|found)"
    r"|unable to",
    re.IGNORECASE,
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _require_mapping(
    export: Any,
    key: str,
) -> Any:
    if not isinstance(export, Mapping) or key not in export:
        raise ValueError(
            "not a research evidence export: "
            f"missing '{key}'."
        )
    return export[key]


def _parts(
    export: Any,
) -> tuple[Mapping, list, list, Mapping]:
    report = _require_mapping(export, "report")
    evidence = _require_mapping(export, "evidence")
    sources = _require_mapping(export, "sources")
    summary = _require_mapping(export, "summary")

    if (
        not isinstance(report, Mapping)
        or not isinstance(report.get("content"), str)
        or not isinstance(report.get("citations"), list)
    ):
        raise ValueError(
            "not a research evidence export: malformed 'report'."
        )

    if not isinstance(evidence, list) or not isinstance(sources, list):
        raise ValueError(
            "not a research evidence export: "
            "'evidence' and 'sources' must be lists."
        )

    if not isinstance(summary, Mapping):
        raise ValueError(
            "not a research evidence export: malformed 'summary'."
        )

    return report, evidence, sources, summary


def _excerpt_key(
    item: Mapping,
) -> tuple[str, str]:
    """Identity of an excerpt: its source plus its whitespace-normalized text."""

    return (
        item["source_id"],
        " ".join(item["excerpt"].split()),
    )


def _ratio(
    numerator: int,
    denominator: int,
) -> float | None:
    return (
        round(numerator / denominator, 4)
        if denominator
        else None
    )


def compute_run_metrics(
    export: Mapping[str, Any],
) -> RunMetrics:
    """Compute objective metrics from one "Download evidence (.json)" export."""

    report, evidence, sources, summary = _parts(export)

    evidence_by_id = {
        item["id"]: item
        for item in evidence
    }
    source_ids = {
        item["id"]
        for item in sources
    }

    broken_links = 0
    cited_sources: set[str] = set()
    cited_keys: set[tuple[str, str]] = set()
    seen_claim_keys: set[frozenset] = set()
    duplicate_claims = 0

    citations = report["citations"]

    for citation in citations:
        evidence_ids = citation.get("evidence_ids") or []

        # A citation with no evidence cannot be traced to anything.
        if not evidence_ids:
            broken_links += 1

        claim_keys: set[tuple[str, str]] = set()

        for evidence_id in evidence_ids:
            item = evidence_by_id.get(evidence_id)

            if item is None:
                broken_links += 1
                continue

            if item["source_id"] not in source_ids:
                broken_links += 1
                continue

            cited_sources.add(item["source_id"])
            claim_keys.add(_excerpt_key(item))

        cited_keys |= claim_keys
        frozen_keys = frozenset(claim_keys)

        if frozen_keys:
            if frozen_keys in seen_claim_keys:
                duplicate_claims += 1
            else:
                seen_claim_keys.add(frozen_keys)

    sources_with_evidence = {
        item["source_id"]
        for item in evidence
    }
    distinct_evidence = {
        _excerpt_key(item)
        for item in evidence
    }

    statuses = [
        item.get("fetch_status")
        for item in sources
    ]
    sources_read = statuses.count("success")
    sources_failed = statuses.count("failed")

    budgets: dict[str, Budget] = {}

    for name, value in summary.items():
        if (
            isinstance(value, Mapping)
            and isinstance(value.get("used"), int)
            and isinstance(value.get("limit"), int)
        ):
            budgets[name] = Budget(
                used=value["used"],
                limit=value["limit"],
            )

    issues = export.get("issues") or []

    return RunMetrics(
        outcome=str(summary.get("outcome", "unknown")),
        end_reason=summary.get("end_reason"),
        claims=len(citations),
        broken_links=broken_links,
        chain_valid=broken_links == 0,
        sources_with_evidence=len(sources_with_evidence),
        sources_cited=len(cited_sources),
        source_coverage=_ratio(
            len(cited_sources),
            len(sources_with_evidence),
        ),
        duplicate_claims=duplicate_claims,
        claim_duplication_ratio=_ratio(
            duplicate_claims,
            len(citations),
        ),
        evidence_count=len(evidence),
        distinct_evidence_count=len(distinct_evidence),
        distinct_evidence_cited=len(cited_keys),
        evidence_per_read_source=_ratio(
            len(evidence),
            sources_read,
        ),
        sources_total=len(sources),
        sources_read=sources_read,
        sources_failed=sources_failed,
        fetch_failure_rate=_ratio(
            sources_failed,
            sources_read + sources_failed,
        ),
        issues=len(issues),
        budgets=budgets,
    )


def check_expectations(
    export: Mapping[str, Any],
    expectations: Expectations,
) -> ExpectationResult:
    """Check one run's answer against a question's written-down expectations."""

    report, _, _, summary = _parts(export)

    content: str = report["content"]

    # Whole-word, case-insensitive matching. Plain substring matching would
    # find "BIC" inside "CUBIC" and wrongly count that fact as present.
    missing = [
        index
        for index, fact in enumerate(expectations.must_mention)
        if not any(
            re.search(
                rf"(?<!\w){re.escape(phrase.strip())}(?!\w)",
                content,
                re.IGNORECASE,
            )
            for phrase in fact
        )
    ]

    sentences = _SENTENCE_END.split(content)
    flagged: list[str] = []

    for pitfall in expectations.pitfalls:
        pattern = re.compile(pitfall.pattern, re.IGNORECASE)
        unless = (
            re.compile(pitfall.unless)
            if pitfall.unless
            else None
        )

        if any(
            pattern.search(sentence)
            and not (unless and unless.search(sentence))
            for sentence in sentences
        ):
            flagged.append(pitfall.name)

    abstain_ok: bool | None = None

    if expectations.expect_abstain:
        abstained = summary.get("outcome") != "verified"
        abstain_ok = abstained or bool(
            _LIMITATION.search(content)
        )

    return ExpectationResult(
        facts_total=len(expectations.must_mention),
        facts_found=len(expectations.must_mention) - len(missing),
        missing_facts=missing,
        pitfalls_flagged=flagged,
        abstain_ok=abstain_ok,
    )
