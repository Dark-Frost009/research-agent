"""Versioned JSON backups: validate first, then merge in one transaction."""
from contextlib import closing
from datetime import datetime, timezone
import sqlite3
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator

from research_agent.history import HistoryStore, HistoryError, _Snapshot
from research_agent.incomplete import IncompleteResearch

MAX_BACKUP_BYTES = 20 * 1024 * 1024
MAX_BACKUP_RECORDS = 5000


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class CompletedRecord(_Record):
    kind: Literal["completed"]
    payload: _Snapshot


class IncompleteRecord(_Record):
    kind: Literal["incomplete"]
    payload: IncompleteResearch


class HistoryBackup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["research-agent-history"] = "research-agent-history"
    version: Literal[1] = 1
    created_at: AwareDatetime
    records: list[Annotated[CompletedRecord | IncompleteRecord, Field(discriminator="kind")]] = Field(max_length=MAX_BACKUP_RECORDS)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({record.id for record in self.records}) != len(self.records):
            raise ValueError("Duplicate record identifiers.")
        return self


def read_backup(raw: bytes) -> HistoryBackup:
    if len(raw) > MAX_BACKUP_BYTES:
        raise HistoryError("This backup exceeds the 20 MB limit.")
    try:
        return HistoryBackup.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        raise HistoryError("This is not a valid supported history backup. No records were restored.") from exc


def export_backup(store: HistoryStore) -> bytes:
    try:
        records = []
        if store.path.exists():
            with closing(store._connect()) as connection:
                # Both tables are read from the same SQLite snapshot.
                connection.execute("BEGIN")
                rows = connection.execute(
                    "SELECT id, 'completed', payload FROM reports UNION ALL "
                    "SELECT id, 'incomplete', payload FROM incomplete_runs ORDER BY id"
                ).fetchmany(MAX_BACKUP_RECORDS + 1)
                if len(rows) > MAX_BACKUP_RECORDS:
                    raise HistoryError("History exceeds the 5,000 entry backup limit. Individual report downloads remain available.")
                for record_id, kind, payload in rows:
                    model = _Snapshot if kind == "completed" else IncompleteResearch
                    records.append(dict(id=record_id, kind=kind, payload=model.model_validate_json(payload)))
        backup = HistoryBackup(created_at=datetime.now(timezone.utc), records=records)
        raw = backup.model_dump_json(indent=2).encode("utf-8")
        if len(raw) > MAX_BACKUP_BYTES:
            raise HistoryError("History exceeds the 20 MB backup limit. Individual report downloads remain available.")
        return raw
    except HistoryError:
        raise
    except (OSError, ValueError, sqlite3.Error) as exc:
        raise HistoryError("Could not prepare a complete backup. History may be unreadable or contain an invalid record; nothing was changed.") from exc


def restore_backup(store: HistoryStore, raw: bytes) -> tuple[int, int]:
    """Return (added, already present). Any conflict rolls back the whole merge."""
    backup = read_backup(raw)
    if not backup.records:
        return 0, 0
    added = skipped = 0
    try:
        with closing(store._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            for record in backup.records:
                existing = connection.execute(
                    "SELECT 'completed', payload FROM reports WHERE id=? UNION ALL "
                    "SELECT 'incomplete', payload FROM incomplete_runs WHERE id=?",
                    (record.id, record.id),
                ).fetchall()
                if existing:
                    model = _Snapshot if record.kind == "completed" else IncompleteResearch
                    if (len(existing) != 1 or existing[0][0] != record.kind
                            or model.model_validate_json(existing[0][1]) != record.payload):
                        raise HistoryError("A backup entry conflicts with existing history. Nothing was restored or overwritten.")
                    skipped += 1
                    continue
                if record.kind == "completed":
                    report = record.payload.report
                    connection.execute("INSERT INTO reports VALUES (?, ?, ?, ?, ?)", (
                        record.id, report.question, report.created_at.astimezone(timezone.utc).isoformat(),
                        len(report.citations), record.payload.model_dump_json()))
                else:
                    result = record.payload
                    connection.execute("INSERT INTO incomplete_runs VALUES (?, ?, ?, ?)", (
                        record.id, result.question, result.created_at.astimezone(timezone.utc).isoformat(),
                        result.model_dump_json()))
                added += 1
        return added, skipped
    except HistoryError:
        raise
    except (OSError, ValueError, sqlite3.Error) as exc:
        raise HistoryError("History could not be restored. No entries were added or overwritten.") from exc
