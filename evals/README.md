# Offline answer-quality scenarios

`answer_quality.json` contains eight fictional research cases with fixed questions,
source text, extracted evidence, proposed answers, supporting quotes, scripted
verifier verdicts, and expected final outcomes.

| Case | Expected behavior |
| --- | --- |
| Supported fact | Release the supported answer with a traceable citation |
| Qualified answer | Preserve the condition attached to a renewal policy |
| Invented evidence handle | Reject a reference to evidence that does not exist |
| Fabricated quote | Reject a quote absent from the cited evidence |
| Unsupported generalization | Discard a draft when the verifier rejects its claim |
| Undeclared assertion | Discard a draft when the verifier identifies incomplete coverage |
| Conflicting evidence | Discard an unjustified definitive answer when the verifier rejects it |
| Irrelevant source | Abstain without finalization calls when extraction returns no relevant evidence |

Run from the project folder with the virtual environment active:

```powershell
python -m pytest tests/test_offline_evaluation.py -v
```

For a machine-readable report, add `-o junit_family=legacy
--junitxml=.local/offline-evaluation.xml` to that command. Each scenario records
its ID, expected outcome, and actual outcome in the report. The evaluation tests
are also included in the normal full pytest suite.

## What this measures

The runner executes the compiled production graph, its budget orchestration,
real synthesis provenance checks, real verifier-response validation, and final
report assembly. It checks release versus abstention, citation-to-evidence-to-source
links, discarded drafts, and finalization call counts. Socket connections are
blocked during each scenario. No API key or provider quota is required.

Planning, search, fetching, extraction, critique, and finalization provider
responses are deterministic doubles. Semantic verdicts are scripted inputs, not
independent judgments made by a live model. A passing suite demonstrates that the
pipeline enforces these fixtures; it does **not** measure Gemini accuracy, live
retrieval relevance, extraction quality, or whether Gemini notices contradictions
or unsupported claims. Live answer-quality evaluation remains separate.

## Adding a case

Give each case a unique ID and keep the source text fictional. An extracted
excerpt must occur in its source text. Use `E1` for the supplied evidence unless
testing an invalid handle. Record the proposed answer, claim, quote, verifier
support and coverage verdicts, expected outcome, and expected finalization call
count. Empty evidence requires no finalization calls; deterministic provenance
rejections stop after synthesis; semantic acceptance or rejection uses both calls.

Expected outcomes are curated in the fixture, not generated from the result under
test. Do not change them merely to make a regression pass. The current fixture
format covers one source and one declared claim per scenario; expand the runner
explicitly before adding multi-source or multi-claim cases.
