"""The fixed question set for live evaluation.

Each question carries the expectations written down BEFORE any run, so a
result can never be quietly re-judged to fit. The file lives in
``evals/live/questions.json`` and is validated here, offline, so a typo in a
regular expression or a duplicated ID is caught by the normal tests and not
in the middle of a live batch that spends quota.

Matching rules are in ``live_metrics`` (whole-word, case-insensitive facts;
sentence-level pitfalls). This module only describes and loads the data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from research_agent.live_metrics import Expectations


class LiveQuestion(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    category: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expectations: Expectations
    # Why the question is in the set and how to read its result.
    notes: str = ""

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank.")
        return value


class QuestionSet(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    version: Literal[1]
    questions: list[LiveQuestion] = Field(min_length=1)

    @field_validator("questions")
    @classmethod
    def _ids_are_unique(
        cls,
        value: list[LiveQuestion],
    ) -> list[LiveQuestion]:
        ids = [question.id for question in value]

        if len(ids) != len(set(ids)):
            raise ValueError("question ids must be unique.")

        return value


def load_questions(
    path: Path,
) -> QuestionSet:
    """Read and validate a question file. Raises on any problem."""

    return QuestionSet.model_validate(
        json.loads(
            Path(path).read_text(encoding="utf-8")
        )
    )
