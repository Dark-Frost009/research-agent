# Evaluation

The project separates three kinds of checking:

| | What it answers | Where | Uses real AI and search? |
|---|---|---|---|
| Unit and integration tests | Is the software correct? | `tests/` | No: all providers are scripted |
| Offline scenarios | Do the safeguards behave as designed? | `evals/`, `tests/test_offline_evaluation.py` | No |
| Live evaluation | How good are real answers? | `evals/live/` | Yes |

Passing tests never proves live answer quality, and live results are never used
as unit tests.

## Offline scenarios

```powershell
python -m pytest tests/test_offline_evaluation.py -v
```

These are fixed scenarios run through the production graph, with scripted
providers and the network blocked. They cover:

- grounded answers and citation provenance
- semantic rejection and unsupported claims
- swapped citations
- conflicting and insufficient evidence
- multiple sources and claims
- adversarial instructions inside sources

They are safeguard regression checks, not a benchmark. Passing the adversarial
prompt tests does not mean the system resists every prompt injection. See
[evals/README.md](../evals/README.md).

### Release-readiness report

```powershell
python -m research_agent.release_readiness
```

This runs the offline checks and writes Markdown and JSON reports to
`.local/readiness/`. It exits with 0 if the offline checks passed, 1 if they
need attention, and 2 if the check could not complete. Any failure, error or
skip prevents a pass. An offline pass does not certify live quality.

## Live evaluation

The tools are in `src/research_agent/live_runner.py`, `live_metrics.py` and
`live_compare.py`; usage is in [evals/live/README.md](../evals/live/README.md).

- **The question set.** 8 fixed questions in `evals/live/questions.json`. Each
  carries expectations written before any run: facts the answer should mention,
  known wrong statements ("pitfalls") to flag, and one fake question that should
  be refused.
- **The runner.** Sends the questions through the same function the app uses
  and saves one JSON file per run. It has a dry-run mode that makes no calls, a
  cap on runs per command, and a 60-second pause between runs. It resumes where
  it stopped, and stops cleanly on provider overload or quota, keeping those
  runs out of the results. Each file records the code version (marked `+dirty`
  for uncommitted edits) and the model names, never keys.
- **The comparison.** Re-scores two batches with the same current code. It shows
  mean [min–max] (n) per question, then averages across questions. Outcome
  rates are measured over all runs and answer quality over verified runs only,
  and the report warns about mixed code versions, models or depths.
- **No AI judge, no significance tests.** With a few runs per question, ranges
  and reading the answers are more honest than a p-value. Every answer in the
  pilot was read by hand and checked against primary sources.

## Findings (22–25 October 2026)

Gemini 3.5 Flash-Lite with Tavily, at Standard depth, unless noted.

### Pre-A2 baseline vs the A2 change

A2 de-duplicates evidence and interleaves it by source before synthesis. Each
batch had 8 questions × 2 runs:

| | Baseline | A2 |
|---|---|---|
| Verified rate (7 answerable questions) | 0.71 | 0.93 |
| Rejected by verification | 4 of 14 | 1 of 14 |
| Fake question refused | 2 of 2 | 2 of 2 |
| Facts found (verified runs) | 0.77 | 0.79 |
| Source coverage (verified runs) | 0.60 | 0.58 |
| Duplicate claims per answer (verified runs) | 1.25 | 0.67 |

- **Source spread.** A2's main goal, spreading citations across more sources,
  showed no measurable effect.
- **Duplicates.** Duplicate claims fell, as intended, though the samples are
  small.
- **Rejections.** They fell from 4 to 1, which is not statistically established
  (Fisher's exact test, p ≈ 0.33).
- **Answer quality.** Unchanged among verified answers.

### The same code run twice

Two batches of identical code gave almost identical averages (verified rate
0.93 vs 0.93, facts 0.82 vs 0.80). Individual questions, however, swung widely,
for example 3 vs 8 claims for one question. A per-question difference of that
size between two versions is therefore noise.

### Reading the answers

- **Scores mislead.** The keyword checks found 13 of 14 expected facts. Reading
  the answers found factual errors with valid citations in 2 of 6 answered
  questions.
- **CUBIC vs Reno.** Every verified answer across all batches (12 of 12) says
  CUBIC uses a "binary search increase". That describes BIC, its predecessor,
  and comes from one blog page, while Wikipedia and a textbook were in the same
  evidence.
- **The fake question.** The "2031 quantum TCP standard" was refused every
  time, with no invented claims.

### An experiment that did not work: flagging contradicted claims

The idea was to give the verifier every excerpt and let it mark a claim as
disputed when another excerpt contradicts it. The success criterion, set in
advance, was that the CUBIC claim is flagged in at least 3 of 4 runs.

**The result: 0 of 4 flagged**, with Flash-Lite and with the stronger Gemini 3.6
Flash. The contradiction in real evidence is implicit: it only shows if you
know that "BIC" means binary-search increase. So the change was **not merged**,
and the branch `eval-disputed-metrics` keeps it as a documented experiment.

### Rejection reasons

Rejected answers now record a reason code (see
[ARCHITECTURE.md](ARCHITECTURE.md#when-an-answer-is-rejected)). The recorded
rejections so far were `quote_mismatch` (twice) and `synthesis_format`. All
three came from the deterministic checks: the model quoted its evidence
inexactly.

### Provider reliability

- **Free-tier errors.** Gemini's free tier often returned 503 (overloaded),
  mostly in the evening in India, and occasionally 429 (per-minute limit).
- **No retries.** The Gemini SDK retries only when configured to, and the
  local app wasn't, so one momentary 503 failed a whole step.
- **The fix.** Retries for temporary errors were added; their effect at busy
  hours is not yet measured.

### Lessons

1. **"Verified" means the quote is real, not that the source is right.**
   Provenance and support are checked; source reliability is not.
2. **Keyword metrics over-score.** Reading answers stays part of the method.
3. **Rejections hide inside averages.** Measure outcomes and quality separately.
4. **Provider failures are not pipeline results.** Keep them out of the data.
5. **Measure the noise floor before trusting an improvement.**
6. **Set success criteria first, and accept a negative result.**

### Limits

- 2 runs per question per batch (4 for the CUBIC experiment).
- Mostly one model and one depth setting.
- 8 questions.
- Correctness was judged by the developer, not a blind rater.
