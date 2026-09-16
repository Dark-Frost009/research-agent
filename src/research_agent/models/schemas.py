"""Domain models for the Research Agent.
 
These are plain Pydantic data contracts. They intentionally contain no
business logic - no URL normalization, no ID generation, no fetching,
no citation validation, no graph orchestration. That logic lives in
dedicated modules elsewhere so it can be tested independently of these
schemas.
 
Traceability chain these models are designed to support:
 
    SubQuestion -> SearchResult -> Source -> Evidence -> Citation
"""
 
from datetime import datetime, timezone
from typing import Literal, Optional
 
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
 
 
def _utcnow() -> datetime:
    """Default factory for timezone-aware 'created now' timestamps."""
    return datetime.now(timezone.utc)
 
 
class _BaseSchema(BaseModel):
    """Shared configuration for every domain model in this module.
 
    - str_strip_whitespace: trims incidental whitespace from LLM/provider
      output before validation runs.
    - extra="forbid": rejects unexpected fields instead of silently
      dropping them, which catches typos and schema drift early.
    """
 
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
 
 
class SubQuestion(_BaseSchema):
    """A single piece of the research plan produced by the Planner.
 
    `created_at_iteration` records which research loop produced this
    sub-question: 0 for the initial plan, 1+ for critique-driven
    follow-ups.
    """
 
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    rationale: Optional[str] = None
    created_at_iteration: int = Field(default=0, ge=0)
 
 
class SearchResult(_BaseSchema):
    """Our normalized internal representation of one search-provider result.
 
    This is not a raw provider response. A provider adapter/wrapper is
    responsible for converting a raw Tavily/etc. response dict into this
    shape before it is ever used to construct a SearchResult. Keeping
    that conversion outside this schema is what lets the rest of the
    system stay provider-agnostic. SearchResult is intentionally not
    merged with Source.
    """
 
    sub_question_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    snippet: str = ""
    rank: Optional[int] = Field(default=None, ge=1)
    provider: str = Field(min_length=1)
    retrieved_at: AwareDatetime = Field(default_factory=_utcnow)
 
 
class Source(_BaseSchema):
    """Normalized metadata for a document/page referenced by the research.
 
    Deliberately holds no fetched page content - extracted content is
    converted directly into bounded Evidence objects elsewhere, and raw
    pages are never kept in shared graph state.
 
    fetch_status starts at "pending": a Source can exist right after
    search discovery, before the extract/fetch step has run.
    """
 
    id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    title: Optional[str] = None
    domain: str = Field(min_length=1)
    content_type: Optional[str] = None
    fetch_status: Literal["pending", "success", "failed", "skipped"] = "pending"
    fetched_at: Optional[AwareDatetime] = None
 
 
class Evidence(_BaseSchema):
    """A specific excerpt extracted from a Source that can support a claim.
 
    Named `excerpt` rather than `quote` because this is a neutral
    representation of extracted source text, not a guarantee of an
    exact verbatim quotation.
    """
 
    id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    sub_question_id: Optional[str] = None
    excerpt: str = Field(min_length=1)
    relevance_note: Optional[str] = None
 
 
class Citation(_BaseSchema):
    """A claim made in the report, linked back to the evidence supporting it.
 
    Carries no validity flag - validity is the output of a separate
    deterministic checker, not an intrinsic property of this object.
    """
 
    id: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
 
 
class CritiqueResult(_BaseSchema):
    """The critique step's verdict on whether the draft report is sufficient."""
 
    sufficient: bool
    gaps: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    reasoning: Optional[str] = None
 
 
class ResearchReport(_BaseSchema):
    """The final research report handed back to the user."""
 
    question: str = Field(min_length=1)
    content: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    created_at: AwareDatetime = Field(default_factory=_utcnow)
 
 
class RunMetrics(_BaseSchema):
    """Aggregate counters for one research run.
 
    Not wired into graph state in Phase 1 - most of these are meant to
    be derived at the end of a run from the lengths of accumulated
    state lists, plus whatever explicit counters (LLM calls, timing)
    get added once real providers exist.
    """
 
    run_id: str = Field(min_length=1)
    iterations_used: int = Field(ge=0)
    total_search_queries: int = Field(ge=0)
    total_sources: int = Field(ge=0)
    total_evidence_extracted: int = Field(ge=0)
    total_llm_calls: Optional[int] = Field(default=None, ge=0)
    duration_seconds: Optional[float] = Field(default=None, ge=0)
    errors: list[str] = Field(default_factory=list)