# Live evaluation

This folder is for evaluation that uses the **real** Gemini and Tavily services.
It is separate from the offline scenarios in `../answer_quality.json`, which use
fictional sources and scripted model replies and run inside `pytest`.

Live runs cost quota, give different results each time, and must never run in the
normal test suite.

## Files

- `questions.json` - the fixed question set. Each question carries its
  expectations, written down **before** any run: facts the answer should mention,
  known wrong statements to flag, and whether the question has no real answer.
- Metrics are computed by `src/research_agent/live_metrics.py` from the app's
  "Download evidence (.json)" export. Both the metrics and the question file are
  validated by offline tests (`tests/test_live_metrics.py`,
  `tests/test_live_questions.py`).

## How to read the checks

- Facts are matched as whole words, case-insensitively, and any listed
  alternative counts. Word forms are listed explicitly (for example "linear" and
  "linearly"), because "linear" does not match inside "linearly".
- A pitfall flags a sentence that matches `pattern` unless the same sentence
  also matches `unless`.
- These are signals about where to look. They do not prove an answer is correct
  or incorrect. Keyword facts in particular can be satisfied by a wrong answer.
- Results vary between runs because search results change. Compare means and
  ranges over several repeats, never a single run.

## Changing a question

Change expectations only before running a comparison. If a question is edited
after results exist, treat all earlier results for it as a different test.
