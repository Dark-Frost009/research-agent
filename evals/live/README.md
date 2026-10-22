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

## Running a batch

The runner (`src/research_agent/live_runner.py`) sends the questions through the
same function the app uses, scores each finished run, and saves one JSON file per
run under `.local/live-evals/<label>/` (git ignores `.local/`). Always preview
first; the preview makes no AI or search request:

```
python -m research_agent.live_runner --label baseline-d5612ba --dry-run
python -m research_agent.live_runner --label baseline-d5612ba --max-runs 8
```

- `--label` names the batch. Use one label per code version, for example the
  commit being measured. Each file also records the commit and whether the
  working tree had uncommitted edits (`+dirty`), so a mislabelled baseline shows.
- `--max-runs` (default 8) is the most runs one command may start. `--runs N`
  asks for N repeats per question; run 1 of every question happens before any
  run 2.
- Running the same command again resumes: runs that already have a result file
  are skipped. This is how a batch is spread over several days of free-tier
  quota.
- If the provider reports a quota or overload, the runner stops at once and does
  not save that run, because it would measure the quota and not the pipeline. A
  run that finished but says in its own issues that the provider throttled it is
  saved as `<question>__<n>.limited.json`, is not counted as done, and also ends
  the batch.
- Each file records the provider and model names, never keys. Exit code 0 means
  the batch finished, 2 means the provider ended it early, 1 means an error.

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
