"""Local, versioned storage of completed reports and incomplete research."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
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
class SavedReport:
    id: str
    question: str
    created_at: str
    citation_count: int
    status: str = "completed"

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

    def list_reports(self, search: str = "") -> list[SavedReport]:
        if not self.path.exists():
            return []
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    "SELECT id, question, created_at, citation_count, 'completed' FROM reports "
                    "UNION ALL SELECT id, question, created_at, 0, 'incomplete' FROM incomplete_runs "
                    "ORDER BY created_at DESC, id DESC"
                ).fetchall()
            query = search.strip().casefold()
            return [SavedReport(*row) for row in rows if query in row[1].casefold()]
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
