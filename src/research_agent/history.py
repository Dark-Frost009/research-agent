"""Local, versioned storage of completed reports and incomplete research."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
import sqlite3
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from research_agent.models.schemas import Evidence, ResearchReport, Source
from research_agent.ui_service import CompletedResearch
from research_agent.incomplete import IncompleteResearch
from research_agent.depth import ResearchDepth
from research_agent.run_summary import RunSummary

OUTCOME_FILTERS = {
    "Verified answer": "verified",
    "No evidence collected": "no_evidence",
    "Answer rejected by verification": "verification_rejected",
    "Not enough budget for answer checks": "finalization_budget",
    "No verified answer": "no_verified_answer",
    "Interrupted run": "incomplete",
    "Not recorded": "unknown",
}


class HistoryError(RuntimeError):
    """A local history operation failed; the current report remains usable."""


class _Snapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    report: ResearchReport
    evidence: list[Evidence]
    sources: list[Source]
    iterations: int = Field(ge=0)
    searches: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    issues: tuple[str, ...] = ()
    depth: ResearchDepth | None = None
    summary: RunSummary | None = None

    @model_validator(mode="after")
    def validate_links(self):
        if self.summary is not None and self.summary.outcome == "incomplete":
            raise ValueError("A completed report cannot claim an incomplete outcome.")
        for items in (self.sources, self.evidence, self.report.citations):
            if len({item.id for item in items}) != len(items):
                raise ValueError("Duplicate saved identifiers.")
        sources = {item.id for item in self.sources}
        evidence = {item.id for item in self.evidence}
        if any(item.source_id not in sources for item in self.evidence):
            raise ValueError("Saved evidence has no source.")
        if any(eid not in evidence for citation in self.report.citations
               for eid in citation.evidence_ids):
            raise ValueError("Saved citation has no evidence.")
        return self

    @classmethod
    def from_result(cls, result: CompletedResearch):
        return cls(report=result.report, evidence=result.evidence, sources=result.sources,
                   iterations=result.iterations, searches=result.searches,
                   warning_count=result.warning_count, issues=getattr(result, "issues", ()),
                   depth=getattr(result, "depth", None), summary=getattr(result, "summary", None))

    def to_result(self) -> CompletedResearch:
        return CompletedResearch(report=self.report, evidence=self.evidence,
                                 sources=self.sources, iterations=self.iterations,
                                 searches=self.searches, warning_count=self.warning_count,
                                 issues=self.issues, depth=self.depth, summary=self.summary)


@dataclass(frozen=True)
class SearchMatch:
    location: str
    excerpt: str


def _search_match(location: str, text: str, query: str) -> SearchMatch | None:
    offset = text.casefold().find(query)
    if offset < 0:
        return None
    # Case folding can expand characters (e.g. ß -> ss). Map the match
    # back to the original text before choosing its surrounding context.
    folded_offset = 0
    start = 0
    for start, character in enumerate(text):
        folded_offset += len(character.casefold())
        if folded_offset > offset:
            break
    left = max(0, start - min(40, max(0, 180 - len(query))))
    right = min(len(text), left + 180)
    excerpt = " ".join(text[left:right].split())
    return SearchMatch(location, ("…" if left else "") + excerpt + ("…" if right < len(text) else ""))


@dataclass(frozen=True)
class SavedReport:
    id: str
    question: str
    created_at: str
    citation_count: int
    status: str = "completed"
    match: SearchMatch | None = field(default=None, compare=False)

    @property
    def label(self) -> str:
        question = " ".join(self.question.split())
        if len(question) > 90:
            question = question[:87] + "..."
        prefix = "Incomplete · " if self.status == "incomplete" else ""
        return f"{prefix}{self.created_at[:16].replace('T', ' ')} UTC · {question} · {self.id[:6]}"


class HistoryStore:
    """Independent short transactions support multiple local browser sessions."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            connection.execute("""CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY, question TEXT NOT NULL,
                created_at TEXT NOT NULL, citation_count INTEGER NOT NULL,
                payload TEXT NOT NULL
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS incomplete_runs (
                id TEXT PRIMARY KEY, question TEXT NOT NULL,
                created_at TEXT NOT NULL, payload TEXT NOT NULL
            )""")
            connection.commit()
            return connection
        except Exception:
            connection.close()
            raise

    def save(self, result: CompletedResearch | IncompleteResearch) -> str:
        """Save once per completed run; duplicate questions remain distinct runs."""
        from datetime import timezone
        try:
            if isinstance(result, IncompleteResearch):
                # Revalidate even if a caller constructed/copied without validation.
                snapshot = IncompleteResearch.model_validate_json(result.model_dump_json())
                record_id = uuid4().hex
                with closing(self._connect()) as connection, connection:
                    connection.execute("INSERT INTO incomplete_runs VALUES (?, ?, ?, ?)", (
                        record_id, snapshot.question,
                        snapshot.created_at.astimezone(timezone.utc).isoformat(),
                        snapshot.model_dump_json(),
                    ))
                return record_id
            snapshot = _Snapshot.from_result(result)
            record_id = uuid4().hex
            created_at = snapshot.report.created_at.astimezone(timezone.utc).isoformat()
            with closing(self._connect()) as connection, connection:
                connection.execute("INSERT INTO reports VALUES (?, ?, ?, ?, ?)", (
                    record_id, snapshot.report.question, created_at,
                    len(snapshot.report.citations), snapshot.model_dump_json(),
                ))
            return record_id
        except (OSError, sqlite3.Error, ValidationError) as exc:
            raise HistoryError("Could not save the report locally.") from exc

    def list_reports(self, search: str = "", *, status: str | None = None,
                     depth: str | None = None, start_date: date | None = None,
                     end_date: date | None = None,
                     outcome: str | None = None) -> list[SavedReport]:
        if status not in (None, "completed", "incomplete"):
            raise ValueError("Unknown history status filter.")
        if depth not in (None, "Quick", "Standard", "Thorough", "Not recorded"):
            raise ValueError("Unknown history depth filter.")
        if outcome is not None and outcome not in OUTCOME_FILTERS.values():
            raise ValueError("Unknown history outcome filter.")
        if start_date and end_date and start_date > end_date:
            raise ValueError("Start date must be on or before end date.")
        if not self.path.exists():
            return []
        query = search.strip().casefold()
        try:
            with closing(self._connect()) as connection:
                # Do not load report bodies for the ordinary unfiltered listing.
                payload_column = ", payload" if query or depth or outcome else ""
                rows = connection.execute(
                    "SELECT id, question, created_at, citation_count, 'completed'" + payload_column + " FROM reports "
                    "UNION ALL SELECT id, question, created_at, 0, 'incomplete'" + payload_column + " FROM incomplete_runs "
                    "ORDER BY created_at DESC, id DESC"
                )
                matches = []
                for row in rows:
                    if status and row[4] != status:
                        continue
                    if start_date or end_date:
                        try:
                            created_at = datetime.fromisoformat(row[2])
                            if created_at.tzinfo is None:
                                continue
                            saved_date = created_at.astimezone(timezone.utc).date()
                        except (ValueError, TypeError):
                            continue
                        if ((start_date and saved_date < start_date)
                                or (end_date and saved_date > end_date)):
                            continue
                    model = _Snapshot if row[4] == "completed" else IncompleteResearch
                    snapshot = None
                    if depth or outcome:
                        try:
                            snapshot = model.model_validate_json(row[5])
                        except ValidationError:
                            continue
                        recorded_depth = snapshot.depth.name if snapshot.depth else "Not recorded"
                        if depth and recorded_depth != depth:
                            continue
                        recorded_outcome = ("incomplete" if isinstance(snapshot, IncompleteResearch)
                                            else snapshot.summary.outcome if snapshot.summary else "unknown")
                        if outcome and recorded_outcome != outcome:
                            continue
                    if not query:
                        matches.append(SavedReport(*row[:5]))
                        continue
                    question_match = _search_match("Question", row[1], query)
                    if question_match:
                        matches.append(SavedReport(*row[:5], match=question_match))
                        continue
                    try:
                        if snapshot is None:
                            snapshot = model.model_validate_json(row[5])
                    except ValidationError:
                        # A damaged entry remains discoverable by question and
                        # must not prevent searching other saved research.
                        continue
                    texts = []
                    if isinstance(snapshot, _Snapshot):
                        texts.append(("Report text", snapshot.report.content))
                    texts.extend(("Source title", source.title or "") for source in snapshot.sources)
                    texts.extend(("Evidence excerpt", item.excerpt) for item in snapshot.evidence)
                    for location, text in texts:
                        match = _search_match(location, text, query)
                        if match:
                            matches.append(SavedReport(*row[:5], match=match))
                            break
                return matches
        except (OSError, sqlite3.Error) as exc:
            raise HistoryError("Could not read local research history.") from exc

    def load(self, record_id: str) -> CompletedResearch | IncompleteResearch:
        if not self.path.exists():
            raise HistoryError("This saved report is no longer available.")
        try:
            with closing(self._connect()) as connection:
                row = connection.execute("SELECT payload FROM reports WHERE id = ?",
                                         (record_id,)).fetchone()
                if row is None:
                    partial = connection.execute("SELECT payload FROM incomplete_runs WHERE id = ?",
                                                 (record_id,)).fetchone()
                    if partial is not None:
                        return IncompleteResearch.model_validate_json(partial[0])
            if row is None:
                raise HistoryError("This saved report is no longer available.")
            return _Snapshot.model_validate_json(row[0]).to_result()
        except (OSError, sqlite3.Error, ValidationError) as exc:
            raise HistoryError("This saved report could not be opened. Other reports and current downloads are unaffected.") from exc


def get_history_store() -> HistoryStore:
    return HistoryStore(Path(__file__).resolve().parents[2] / ".local" / "research_history.sqlite")
