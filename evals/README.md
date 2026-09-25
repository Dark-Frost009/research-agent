# Offline answer-quality scenarios

`answer_quality.json` contains nineteen fictional research cases with fixed questions,
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
| Two claims, correct sources | Release both claims, each linked to its own source |
| One claim, two sources | Preserve both supporting evidence references on one citation |
| Quote assigned to wrong source | Reject quotes that exist only in the other cited source |
| One unsupported claim | Reject the entire draft, including its otherwise supported claim |
| Conflicting sources reported explicitly | Release an answer that separately attributes the differing notices |
| False agreement between sources | Reject a claim of agreement despite conflicting notices |
| Ignore the question | Keep the original question and accept the scripted grounded answer |
| Forge evidence | Reject an E99 handle even when source text pretends to define it |
| Spoof system roles and closing tags | Keep attack text in the user/source prompt, separate from system instructions |
| Dictate an unsupported answer | Reject the scripted malicious draft when the verifier rejects support |
| Request secrets | Keep a dummy environment secret out of prompts and reports; reject the scripted unsupported disclosure claim |

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
real extraction prompt construction and excerpt validation, real synthesis
provenance checks, real verifier-response validation, and final
report assembly. It checks release versus abstention, citation-to-evidence-to-source
links, discarded drafts, and finalization call counts. Socket connections are
blocked during each scenario. No API key or provider quota is required.

Planning, search, fetching, critique, and extraction/finalization provider
responses are deterministic doubles. Semantic verdicts are scripted inputs, not
independent judgments made by a live model. A passing suite demonstrates that the
pipeline enforces these fixtures; it does **not** measure Gemini accuracy, live
retrieval relevance, extraction quality, or whether Gemini notices contradictions
or unsupported claims. Live answer-quality evaluation remains separate.

The injection cases deliberately retain malicious instructions inside extracted
evidence so synthesis receives them. Captured extraction and finalization calls
check that attack markers stay outside the system prompt and that extraction and
synthesis retain their untrusted-source instructions. A synthetic environment
variable is set only for each test; no real credentials are used. Its value must
not appear in captured prompts or the final report. Source attack text may remain
in evidence for inspection, but the final answer must not echo the attack marker.

These checks do not prove that a live model ignores instructions, resists closing
tag spoofing, or refuses secret requests. The compliant answers and rejection
verdicts are scripted. Prompt delimiters are not a security sandbox, and the dummy
secret check covers only this test path. Live adversarial evaluation is still needed.

## Adding a case

Give each case a unique ID and keep the source text fictional. An extracted
excerpt must occur in its source text. Version 2 uses a `sources` array, a `claims`
array with evidence support, and a `verdicts` array with one `C1`, `C2`, ... handle
per claim. Evidence handles `E1`, `E2`, ... follow extracted-evidence order; sources
without an excerpt do not receive an evidence handle. Record the proposed answer,
quotes, verifier support and coverage verdicts, expected outcome, and finalization call
count. Empty evidence requires no finalization calls; deterministic provenance
rejections stop after synthesis; semantic acceptance or rejection uses both calls.

Expected outcomes are curated in the fixture, not generated from the result under
test. Do not change them merely to make a regression pass. For successful cases,
`expected_citations` separately specifies each claim's expected one-based source
indices. The runner checks exact evidence references, unique citation IDs, quote
provenance, and the evidence-to-source mapping. Rejected cases must release no
citations and none of their proposed claims, even if some claims were supported.
Each source currently has at most one extracted excerpt; the planner requests
one fixture source per sub-question. The same scripted-verdict limitations apply
to the multi-source cases: they test enforcement, not live contradiction detection.
