"""Safe, persistent summaries of committed budget usage and run outcomes."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from research_agent.graph.budget import BudgetLimits
from research_agent.graph.rejection import RejectionReason


EndReason = Literal[
    "sufficient",
    "no_followups",
    "no_new_followups",
    "round_limit",
    "search_limit",
    "source_limit",
    "fetch_limit",
    "ai_limit",
    "quota",
    "unavailable",
    "provider",
    "response",
    "unexpected",
    "unknown",
]

Outcome = Literal[
    "verified",
    "no_evidence",
    "verification_rejected",
    "finalization_budget",
    "no_verified_answer",
    "incomplete",
    "unknown",
]

_REASONS = {
    "sufficient": (
        "The critique judged the collected evidence sufficient to write an answer."
    ),
    "no_followups": (
        "The critique proposed no further research questions."
    ),
    "no_new_followups": (
        "All proposed follow-up questions were already in the research plan."
    ),
    "round_limit": (
        "The research-round budget prevented another round."
    ),
    "search_limit": (
        "The search budget prevented another round."
    ),
    "source_limit": (
        "The source budget prevented another round."
    ),
    "fetch_limit": (
        "The page-read budget prevented another round."
    ),
    "ai_limit": (
        "No AI budget remained for further research while preserving "
        "final verification."
    ),
    "quota": (
        "The AI service's rate or quota limit stopped the run."
    ),
    "unavailable": (
        "The AI service was temporarily unavailable."
    ),
    "provider": (
        "An AI service request failed."
    ),
    "response": (
        "An AI response did not pass the required checks."
    ),
    "unexpected": (
        "An unexpected error stopped the run."
    ),
    "unknown": (
        "The reason research ended was not recorded."
    ),
}

_OUTCOMES = {
    "verified": (
        "An answer with verified citations was produced."
    ),
    "no_evidence": (
        "No evidence was collected; no verified answer was produced."
    ),
    "verification_rejected": (
        "The proposed answer failed grounding or verification and was discarded."
    ),
    "finalization_budget": (
        "There was not enough AI budget for both final answer checks; "
        "no verified answer was produced."
    ),
    "no_verified_answer": (
        "No answer with verified citations was produced."
    ),
    "incomplete": (
        "The run is incomplete; no verified answer was released."
    ),
    "unknown": (
        "The final outcome was not recorded."
    ),
}


# Plain-language explanations of the rejection codes, for people reading a
# summary. The codes themselves are defined in research_agent.graph.rejection.
_REJECTIONS = {
    "quote_mismatch": (
        "A supporting quote did not match its cited evidence word for word."
    ),
    "synthesis_format": (
        "The drafted answer broke a citation rule, such as a claim missing "
        "from the answer text."
    ),
    "verifier_format": (
        "The verification step returned an unusable reply."
    ),
    "unsupported_claim": (
        "Verification judged at least one claim not supported by its "
        "cited evidence."
    ),
    "incomplete_coverage": (
        "Verification found factual statements without a supporting citation."
    ),
}


class BudgetCounter(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    used: int = Field(
        ge=0,
    )
    limit: int = Field(
        ge=0,
    )

    @model_validator(mode="after")
    def within_limit(self):
        if self.used > self.limit:
            raise ValueError(
                "Recorded usage exceeds the configured budget."
            )

        return self


class RunSummary(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    rounds: BudgetCounter
    searches: BudgetCounter
    sources: BudgetCounter
    page_reads: BudgetCounter
    ai_calls: BudgetCounter
    end_reason: EndReason
    outcome: Outcome
    # Why a proposed answer was rejected; only for "verification_rejected".
    # Optional so summaries saved before this field existed still load.
    rejection_reason: RejectionReason | None = None

    @model_validator(mode="after")
    def reason_only_for_rejections(self):
        if (
            self.rejection_reason is not None
            and self.outcome != "verification_rejected"
        ):
            raise ValueError(
                "Only a rejected answer can have a rejection reason."
            )

        return self

    @property
    def stop_message(self) -> str:
        return _REASONS[
            self.end_reason
        ]

    @property
    def outcome_message(self) -> str:
        if (
            self.outcome == "verified"
            and self.end_reason != "sufficient"
        ):
            return (
                "A partial answer with verified citations was produced. "
                "Additional research was still needed when the run ended."
            )

        message = _OUTCOMES[
            self.outcome
        ]

        if self.rejection_reason is not None:
            message += " " + _REJECTIONS[self.rejection_reason]

        return message

    def rows(self) -> list[dict]:
        return [
            {
                "Resource": label,
                "Budget used": counter.used,
                "Limit": counter.limit,
            }
            for label, counter in (
                (
                    "Research rounds",
                    self.rounds,
                ),
                (
                    "Searches",
                    self.searches,
                ),
                (
                    "Sources",
                    self.sources,
                ),
                (
                    "Page reads",
                    self.page_reads,
                ),
                (
                    "AI calls",
                    self.ai_calls,
                ),
            )
        ]

    def text_export(self) -> str:
        return "\n".join(
            [
                self.stop_message,
                self.outcome_message,
                *[
                    (
                        f"{row['Resource']}: "
                        f"{row['Budget used']} / "
                        f"{row['Limit']}"
                    )
                    for row in self.rows()
                ],
                (
                    "Budget used counts committed reservations, including "
                    "failed attempts and reserved final verification calls; "
                    "it is not provider billing or token usage."
                ),
            ]
        )


def summarize_run(
    state: dict,
    limits: BudgetLimits,
    *,
    error: EndReason | None = None,
) -> RunSummary:
    def counter(
        key,
        limit,
    ):
        return BudgetCounter(
            used=state.get(
                key,
                0,
            ),
            limit=limit,
        )

    return RunSummary(
        rounds=counter(
            "iteration_count",
            limits.max_research_iterations,
        ),
        searches=counter(
            "search_queries_used",
            limits.max_search_queries_per_run,
        ),
        sources=BudgetCounter(
            used=len(
                {
                    item.id
                    for item in state.get(
                        "sources",
                        [],
                    )
                }
            ),
            limit=limits.max_sources_per_run,
        ),
        page_reads=counter(
            "source_fetches_used",
            limits.max_source_fetches_per_run,
        ),
        ai_calls=counter(
            "llm_calls_used",
            limits.max_llm_calls_per_run,
        ),
        end_reason=(
            error
            or state.get(
                "research_stop_reason",
                "unknown",
            )
        ),
        outcome=(
            "incomplete"
            if error
            else state.get(
                "finalization_outcome",
                "unknown",
            )
        ),
        rejection_reason=(
            None
            if error
            else state.get(
                "finalization_rejection_reason",
            )
        ),
    )