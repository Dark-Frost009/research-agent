"""Prompt contract for the research planning step.

The planner decomposes a user's research question into focused,
non-overlapping sub-questions. It does not perform research, create
citations, or answer the question itself.
"""


PLANNER_SYSTEM_PROMPT = """
You are the planning component of a research agent.

Your job is to decompose the user's research question into a small set
of focused research sub-questions.

Requirements:
- Preserve the user's original intent.
- Each sub-question must be independently researchable.
- Avoid duplicate or heavily overlapping sub-questions.
- Cover the important dimensions needed to answer the original question.
- Keep each sub-question concise and specific.
- Include a short rationale explaining why the sub-question matters.
- Do not answer the research question.
- Do not invent facts, sources, citations, URLs, or evidence.
- Do not generate IDs or internal system metadata.
""".strip()


def build_planner_user_prompt(
    original_question: str,
    *,
    max_sub_questions: int,
) -> str:
    """Build the user message sent to the planner LLM."""

    if not isinstance(original_question, str):
        raise TypeError(
            "original_question must be a string."
        )

    clean_question = original_question.strip()

    if not clean_question:
        raise ValueError(
            "original_question must not be blank."
        )

    if isinstance(max_sub_questions, bool) or not isinstance(
        max_sub_questions,
        int,
    ):
        raise TypeError(
            "max_sub_questions must be an integer."
        )

    if max_sub_questions < 1:
        raise ValueError(
            "max_sub_questions must be at least 1."
        )

    return (
        "Original research question:\n"
        f"{clean_question}\n\n"
        "Create a research plan containing between 1 and "
        f"{max_sub_questions} focused sub-questions."
    )