# Research Agent

A production-oriented research agent built with Python and LangGraph.

The agent takes a research question, breaks it into focused sub-questions, searches the web, safely fetches source pages, extracts grounded evidence, critiques whether that evidence is sufficient, optionally performs another research iteration, and produces a final answer with citations tied back to verified evidence.

The project is designed around a simple principle:

> An LLM should not be trusted merely because it produced a plausible answer.

Instead, research, evidence extraction, synthesis, citation creation, and semantic verification are separated into explicit stages with bounded budgets and fail-closed validation.

---

## What the Agent Does

Given a question such as:

```text
What is retrieval-augmented generation and what problem does it solve?
```

the production workflow is approximately:

```text
User Question
     │
     ▼
Reserve Research Iteration
     │
     ▼
Planner
     │
     ▼
Sub-Questions
     │
     ▼
Search
     │
     ▼
Source Admission
     │
     ▼
Safe Web Fetching
     │
     ▼
Evidence Extraction
     │
     ▼
Critic
     │
     ├──────── Evidence insufficient ────────┐
     │                                       │
     │                                       ▼
     │                              Follow-Up Questions
     │                                       │
     │                              Reserve Next Iteration
     │                                       │
     │                                  Search Again
     │                                       │
     └───────────────────────────────────────┘
     │
     ▼
Final Synthesis
     │
     ▼
Semantic Verification
     │
     ▼
Trusted Citations
     │
     ▼
ResearchReport
```

The research loop is bounded by whole-run budgets so the graph cannot continue searching, fetching, or calling the LLM indefinitely.

---

## Key Features

- LangGraph-based production research workflow
- Google Gemini as the LLM provider
- Tavily web search
- Safe webpage fetching with SSRF protections
- NAT64-aware IP validation
- Explicit redirect validation
- Structured Pydantic outputs for LLM stages
- Deterministic evidence and citation handles
- Exact quote grounding
- Evidence-to-claim provenance validation
- Semantic support verification before trusted citations are created
- Evidence sufficiency critic
- Bounded multi-iteration research loop
- Whole-run search, fetch, source, iteration, and LLM budgets
- Protected finalization LLM reserve
- Deterministic LangGraph reducers
- CLI entry point
- Extensive unit, orchestration, graph-slice, integration, and production-graph tests
- Real end-to-end provider smoke testing

---

# Architecture

## 1. Planner

The Planner converts the original research question into focused `SubQuestion` objects.

The Planner:

- receives only the original research question
- produces structured output through Gemini
- is used only for the initial research iteration
- does not execute searches itself
- consumes one optional-research LLM budget unit when authorized

Follow-up iterations do not call the Planner again.

The Critic already produces concrete follow-up questions, so those questions are converted deterministically into `SubQuestion` objects without spending another Planner LLM call.

---

## 2. Search

Each authorized sub-question is sent to the configured search provider.

Production currently uses:

```text
Tavily
```

Search execution is separated into two stages:

```text
reserve_search
      ↓
execute_search
```

The reservation stage determines exactly how many queries are allowed before any provider side effect occurs.

This prevents uncontrolled search fan-out.

---

## 3. Source Admission

Search results are converted into durable `Source` objects.

Source admission is bounded by:

```text
MAX_SOURCES_PER_RUN
```

Sources are deduplicated using stable IDs.

The original search-result URL remains the stable source identity even when a webpage redirects during fetching.

The fetched destination is stored separately as:

```text
final_url
```

---

## 4. Safe Web Fetching

Authorized sources are fetched through `WebPageFetcher`.

The fetch layer includes protections for:

- unsafe URL schemes
- private IP ranges
- loopback addresses
- link-local addresses
- unsafe redirects
- IPv4-mapped addresses
- NAT64 embedded IPv4 addresses
- excessive redirect chains
- oversized responses
- unsupported content types

A redirect does not automatically become trusted.

Every redirect destination must pass the URL safety validator before another network request is made.

Raw webpage text is transient and is not stored in durable graph state.

---

## 5. Evidence Extraction

Fetched pages are passed to the `EvidenceExtractor`.

Evidence is always associated with:

```text
Source
   +
SubQuestion
```

The extraction stage requires evidence excerpts to be grounded directly in fetched page text.

The LLM cannot create arbitrary supporting quotes.

Returned evidence must pass deterministic validation before becoming durable graph state.

---

## 6. Critic

After evidence collection, the Critic asks:

```text
Is the accumulated evidence sufficient to answer the original question?
```

The Critic receives grounded evidence only.

It does not receive raw webpage contents.

The Critic can return:

```text
sufficient = True
```

or:

```text
sufficient = False
gaps = [...]
follow_up_questions = [...]
```

If more research is justified and whole-run budgets allow it, the graph begins another research iteration.

Otherwise, the graph proceeds to finalization using the evidence already collected.

---

# Research Loop

The production loop is:

```text
Evidence
   ↓
Critic
   │
   ├── sufficient
   │       ↓
   │   Finalization
   │
   └── insufficient
           ↓
     Check remaining budgets
           │
           ├── unavailable
           │       ↓
           │   Finalization
           │
           └── available
                   ↓
            Reserve iteration
                   ↓
          Follow-up SubQuestions
                   ↓
                 Search
                   ↓
                 Fetch
                   ↓
               Evidence
                   ↓
                 Critic
```

The router checks whether at least one additional research cycle is actually viable before entering another iteration.

---

# Grounded Synthesis

Final synthesis does not receive arbitrary source documents.

It receives validated `Evidence`.

Before synthesis, evidence receives deterministic handles:

```text
E1
E2
E3
...
```

The LLM must return claims together with:

```text
claim_text
evidence_handle
supporting_quote
```

The system then validates that:

1. every referenced evidence handle exists
2. every supporting quote is an exact substring of the referenced evidence excerpt
3. duplicate evidence handles are rejected
4. duplicate synthesized claims are rejected
5. the returned claim text exists in the generated answer
6. citation IDs are not created until validation succeeds

This prevents citations from being accepted solely because the LLM says they are correct.

---

# Semantic Verification

Exact quote matching proves that a quote came from the source.

It does not by itself prove that the quote actually supports the claim.

The project therefore includes a separate semantic verification stage.

After synthesis, another structured LLM call evaluates claim-to-evidence support.

This stage verifies:

```text
Claim
   ↓
Referenced Evidence
   ↓
Does the evidence actually support the claim?
```

Trusted `Citation` objects are created only after both deterministic provenance checks and semantic verification succeed.

If deterministic grounding validation or semantic verification rejects the proposed answer, production finalization discards that answer and returns a `ResearchReport` with this fixed content and zero citations:

```text
The available evidence could not be verified strongly enough to produce a grounded answer.
```

Only `SynthesisValidationError` and `SynthesisVerificationError` trigger this fallback. Rejected draft content and verifier diagnostics are not copied into graph state or the final report. No additional provider calls are made, and the already-recorded two-call finalization reservation is retained, including when grounding rejection prevents the verifier call.

Provider failures, generic structured-response errors, and programming errors still propagate as operational failures; the CLI reports failure and exits with a nonzero status. The fallback does not weaken either validation gate.


---

# Protected Finalization

Finalization requires two LLM calls:

```text
1. Synthesis
2. Semantic verification
```

The budget system reserves those calls as an atomic finalization pair.

Optional research work is not allowed to consume this protected capacity.

Default:

```env
FINALIZATION_LLM_RESERVE=2
```

If only part of the required finalization capacity is available, the finalization pair is not partially executed.

---

# Whole-Run Budget System

The agent uses centralized whole-run budgets rather than allowing individual nodes to make unlimited provider calls.

The main limits are:

```env
MAX_RESEARCH_ITERATIONS=2

MAX_SEARCH_QUERIES_PER_RUN=8
MAX_SEARCH_QUERIES_PER_ITERATION=5
MAX_SEARCH_RESULTS_PER_QUERY=5

MAX_SOURCES_PER_RUN=12
MAX_SOURCE_FETCHES_PER_RUN=12

MAX_LLM_CALLS_PER_RUN=64
FINALIZATION_LLM_RESERVE=2
```

Durable state tracks actual usage:

```text
iteration_count
search_queries_used
source_fetches_used
llm_calls_used
```

Remaining capacity is calculated rather than stored.

This avoids duplicated mutable budget state.

Provider attempts consume budget once execution has been authorized, even when the provider later fails.

---

# Reservation Before Side Effects

External operations follow a reservation/execution pattern.

Example:

```text
reserve_source_fetch
        ↓
execute_source_fetch
```

The reservation determines exactly what work is authorized.

Only that authorized batch is allowed to reach the external provider.

This pattern is used for operations such as:

```text
Planner LLM calls
Search calls
Source fetches
Evidence extraction LLM calls
Critic LLM calls
Finalization LLM calls
```

This prevents worker-level fan-out from bypassing whole-run limits.

---

# State and Reducers

`ResearchState` contains durable research data such as:

```text
original_question
sub_questions
search_results
sources
evidence
critique
draft_content
citations
final_report

iteration_count
search_queries_used
source_fetches_used
llm_calls_used

errors
```

Additive usage counters use reducers with delta semantics.

Source merging is deterministic and independent of branch execution order.

Transient authorization objects and fetched page contents are stored outside durable research data in the graph workspace.

---

# Production Graph

The compiled production graph lives in:

```text
src/research_agent/graph/builder.py
```

The main route is:

```text
START
  ↓
reserve_initial_iteration
  ↓
reserve_planner
  ↓
execute_planner
  ↓
reserve_search
  ↓
execute_search
  ↓
admit_sources
  ↓
prepare_evidence_collection_plan
  ↓
reserve_source_fetch
  ↓
execute_source_fetch
  ↓
reserve_evidence_extraction
  ↓
execute_evidence_extraction
  ↓
reserve_critique
  ↓
execute_critique
  ↓
route_after_critique
  ├──────── continue_research
  │               ↓
  │      reserve_follow_up_iteration
  │               ↓
  │      adapt_critique_follow_ups
  │               ↓
  │            search...
  │
  └──────── finalize
                  ↓
          reserve_finalization
                  ↓
          execute_finalization
                  ↓
          assemble_final_report
                  ↓
                 END
```

---

# Production Dependency Wiring

The production composition root lives in:

```text
src/research_agent/bootstrap.py
```

It builds:

```text
Settings
   │
   ├── GeminiLLMClient
   │
   ├── TavilySearchClient
   │
   ├── WebPageFetcher
   │
   └── BudgetPolicy
            │
            ▼
         Planner
         SearchNode
         SourceNode
         SourceFetcher
         EvidenceExtractor
         EvidenceCollector
         Critic
         Synthesizer
            │
            ▼
    ResearchGraphContext
            │
            ▼
    Compiled LangGraph
```

Production provider logic is intentionally kept out of graph nodes.

---

# Project Structure

```text
research-agent/
│
├── src/
│   └── research_agent/
│       ├── bootstrap.py
│       ├── config.py
│       ├── main.py
│       │
│       ├── graph/
│       │   ├── budget.py
│       │   ├── builder.py
│       │   ├── context.py
│       │   ├── state.py
│       │   │
│       │   └── nodes/
│       │       ├── critic.py
│       │       ├── critic_orchestration.py
│       │       ├── evidence.py
│       │       ├── evidence_collector.py
│       │       ├── evidence_orchestration.py
│       │       ├── finalization_orchestration.py
│       │       ├── follow_up_orchestration.py
│       │       ├── iteration.py
│       │       ├── planner.py
│       │       ├── planner_orchestration.py
│       │       ├── report_assembly.py
│       │       ├── research_routing.py
│       │       ├── search.py
│       │       ├── search_orchestration.py
│       │       ├── sources.py
│       │       ├── source_fetcher.py
│       │       ├── source_fetch_orchestration.py
│       │       ├── source_orchestration.py
│       │       ├── synthesis.py
│       │       └── synthesis_verifier.py
│       │
│       ├── llm/
│       │   ├── client.py
│       │   └── gemini.py
│       │
│       ├── models/
│       │   └── schemas.py
│       │
│       ├── prompts/
│       │   ├── critic.py
│       │   ├── evidence.py
│       │   ├── planner.py
│       │   ├── synthesis.py
│       │   └── synthesis_verification.py
│       │
│       └── tools/
│           ├── url_safety.py
│           ├── web_extract.py
│           └── web_search.py
│
├── tests/
├── .env.example
├── pyproject.toml
└── README.md
```

---

# Requirements

Current project requirements include:

```text
Python >= 3.12
Pydantic
pydantic-settings
LangGraph
Tavily Python SDK
Requests
BeautifulSoup
Google GenAI SDK
Pytest
```

---

# Installation

The commands below are written for Windows PowerShell.

Clone the repository and enter the project directory.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project in editable mode:

```powershell
pip install -e .
```

For development dependencies:

```powershell
pip install -e ".[dev]"
```

---

# Environment Configuration

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Then configure your providers.

Example:

```env
APP_ENV=development
LOG_LEVEL=INFO

LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.6-flash
LLM_API_KEY=YOUR_GEMINI_API_KEY

SEARCH_PROVIDER=tavily
TAVILY_API_KEY=YOUR_TAVILY_API_KEY

MAX_RESEARCH_ITERATIONS=2
MAX_SUB_QUESTIONS=5

MAX_SEARCH_QUERIES_PER_RUN=8
MAX_SEARCH_QUERIES_PER_ITERATION=5
MAX_SEARCH_RESULTS_PER_QUERY=5

MAX_SOURCES_PER_RUN=12
MAX_SOURCE_FETCHES_PER_RUN=12

MAX_LLM_CALLS_PER_RUN=64
FINALIZATION_LLM_RESERVE=2

REQUEST_TIMEOUT_SECONDS=15
MAX_RESPONSE_BYTES=2097152
MAX_TEXT_CHARS=100000
MAX_REDIRECTS=5
```

The `.env` file is ignored by Git.

Never commit API keys.

---

# Running the Agent

The project exposes a console command:

```powershell
research-agent "What is retrieval-augmented generation and what problem does it solve?"
```

The equivalent module invocation is:

```powershell
python -m research_agent.main "What is retrieval-augmented generation and what problem does it solve?"
```

Help:

```powershell
research-agent --help
```

A successful run prints a final research report and citation information.

Example shape:

```text
RESEARCH REPORT
============================================================

<grounded final answer>

CITATIONS
------------------------------------------------------------
[citation-id] <supported claim>
  Evidence: <evidence-id>
```

---

# Example Live Result

During development, the production graph was run against real Gemini and Tavily services for the question:

```text
What is retrieval-augmented generation and what problem does it solve?
```

The live graph successfully completed:

```text
Planner
→ Tavily Search
→ Source Admission
→ Web Fetching
→ Evidence Extraction
→ Critique
→ Synthesis
→ Semantic Verification
→ ResearchReport
```

One verified live run produced:

```text
6 trusted citations
```

and a final grounded answer explaining RAG, external retrieval, training-data cutoffs, hallucination risk, domain-specific information, and the difference between retrieval augmentation and retraining.

A follow-up live smoke-test attempt on 2026-09-19, after adding the safe rejection fallback, stopped at the first Gemini Planner call with `LLMProviderError`. It did not reach Tavily search, evidence collection, or finalization. The six-citation result above is the earlier successful run; the latest attempt does not reverify the live success path. The underlying provider-error cause was not captured in this attempt.

Because search engines, webpages, and LLM outputs are external and dynamic, identical questions are not expected to produce identical sources or outputs every time.

The agent is designed to fail closed when usable evidence is insufficient rather than fabricate a confident answer.

For example, a bounded smoke-test run correctly returned:

```text
The available evidence is insufficient to answer the research question.
```

when adequate grounded evidence was not available in that run.

---

# Testing

Run the full suite:

```powershell
pytest -q
```

Current verified development checkpoint:

```text
1847 passed
```

The suite includes:

- configuration tests
- schema tests
- budget-policy tests
- URL-safety tests
- web-search tests
- web-extraction tests
- Gemini adapter tests
- Planner tests
- Search tests
- Source tests
- Source-fetch tests
- Evidence extraction tests
- Evidence collection tests
- Critic tests
- synthesis tests
- semantic-verification tests
- orchestration tests
- LangGraph slice tests
- production graph topology tests
- production graph execution tests (13 cases), including real synthesis/verifier rejection, rejection-content isolation, budget exhaustion after follow-up research, denied real Critic authorization, and finalization operational-error propagation
- bootstrap/composition-root tests
- CLI tests

Provider-facing tests use mocked or deterministic dependencies unless explicitly performing a manual live smoke test.

---

# Security Model

The project treats LLM output and external web content as untrusted.

Important security boundaries include:

```text
Search result URL
      ↓
URL validation
      ↓
Network request
      ↓
Redirect destination
      ↓
URL validation again
      ↓
Fetched page
```

External webpage content is also treated as untrusted prompt data.

Prompts explicitly instruct LLM stages not to follow instructions embedded inside retrieved webpages.

The project does not treat source text as trusted system instructions.

---

# Citation Trust Model

The citation pipeline intentionally separates several different questions:

```text
Did the quote come from the evidence?
                ↓
Does that evidence exist?
                ↓
Does the quote support the claim?
                ↓
Only then create a trusted Citation
```

This means:

```text
LLM-generated citation ≠ trusted citation
```

A citation becomes trusted only after passing the project's deterministic and semantic verification gates.

---

# Why the Critic Runs Before Final Synthesis

The Critic evaluates evidence before finalization.

The graph intentionally does not:

```text
Synthesize
→ Critique
→ Research More
→ Synthesize Again
```

because final synthesis and semantic verification consume a protected two-call finalization budget.

Instead:

```text
Research
→ Critique
→ Research More if needed
→ Final Synthesis once
→ Semantic Verification once
```

This protects finalization capacity and avoids spending final-answer LLM calls before research is complete.

---

# Why Follow-Up Research Does Not Call Planner Again

The Critic already returns explicit follow-up research questions.

Calling Planner again would:

- spend another LLM call
- duplicate work
- introduce unnecessary nondeterminism
- consume optional-research budget

Instead, follow-up questions are deterministically converted into new `SubQuestion` objects.

---

# Current Limitations

The production graph is currently compiled without a persistent LangGraph checkpointer.

The reservation/execution architecture ensures normal reducer ordering during a running graph execution, but it does not by itself provide crash-durable accounting across process termination.

True crash-resume guarantees would require additional work such as:

```text
persistent checkpointer
+
durable operation reservations
+
idempotency strategy
```

The project therefore does not currently claim crash-safe resumability.

Other current limitations include:

```text
Gemini is the only production LLM provider wired in bootstrap.py

Tavily is the only production search provider wired in bootstrap.py

Local Streamlit interface and CLI; no authenticated public hosting

Search and webpage availability can vary between runs

Some webpages may be inaccessible or unsupported

External provider availability and rate limits can affect live runs
```

---

# Design Principles

Both the CLI and Streamlit use the shared graph execution configuration in
`graph/execution.py`. Its step limit scales with the configured maximum research
iterations and leaves room for finalization; provider budgets remain enforced
separately by the graph. Offline integration tests exercise both entry points
through two and eight research rounds.

The Gemini adapter translates known SDK API errors and HTTP transport failures
into safe provider messages. SDK errors with status 429 indicate quota/rate limits;
502, 503, and 504 indicate temporary unavailability. Tests construct actual SDK
exception classes without making requests. Invalid SDK response JSON is classified
as a response error. Unexpected programming errors retain their original type
at the adapter boundary instead of being relabeled as provider outages; the web
interface still presents safe messages. No adapter retries are added.

The project emphasizes:

```text
Bounded execution
Deterministic validation
Explicit orchestration
Provider isolation
Evidence provenance
Fail-closed behavior
Security before convenience
Testability
Clear separation between transient and durable state
```

The goal is not merely to demonstrate that an LLM can search the web.

The goal is to demonstrate how an LLM-powered research system can be engineered so that search, evidence, citations, budgets, provider calls, and failure behavior are explicit and testable.

---

# Development Status

Current development checkpoint:

```text
Production LangGraph        Implemented
Planner                     Implemented
Tavily Search               Implemented
Safe Web Fetching           Implemented
Evidence Extraction         Implemented
Critic / Research Loop      Implemented
Whole-Run Budgets           Implemented
Grounded Synthesis          Implemented
Semantic Verification       Implemented
Safe Rejection Report       Implemented
Citation Validation         Implemented
Production Composition      Implemented
CLI                         Implemented
Live Provider Run           Verified
Full Test Suite             1847 passing
Persistent Checkpointing    Not yet implemented
```

---

# License

No license has been selected yet.

## Workspace isolation

Each production graph invocation requires a fresh `ResearchGraphContext` and
`TransientWorkspace`. Call `build_research_context(settings)` for each request,
or `build_research_application(settings)` to build the graph and context together.
The CLI already creates a fresh application for every `run_research()` call.

The compiled graph can be reused with fresh contexts. Its initial node atomically
claims the workspace before clearing temporary data, reserving budget, or calling
providers. Sequential or concurrent reuse of the same workspace raises an error.
The claim remains consumed after completion, failure, or `clear_all()`; start a new
request with fresh dependencies rather than resetting a used workspace. Follow-up
iterations within one invocation remain supported. This guard applies to the
production graph entry point, not direct calls to individual orchestration nodes.
It does not add retries, checkpoint resumability, or guarantees about the thread
safety of provider clients manually shared between otherwise separate contexts.


## Local web interface

Install the project dependencies, then run from the project folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
```

Open http://localhost:8502. Enter a question and select **Start research**.
The page shows research progress, the completed report, source links and
supporting excerpts, plus TXT and JSON downloads. Completed reports are saved locally and can be reopened from the sidebar
after a restart. Unsubmitted question text remains session-only.

The interface loads the project's `.env` file and uses the existing Gemini and
Tavily configuration and research budgets. Every submitted request gets fresh
dependencies and a fresh workspace. Viewing/downloading results does not rerun
research. A failed request leaves the previous completed report available,
clearly labeled with its own question. Provider error details and credentials
are not displayed. Unverified fallback reports are labeled without claiming
they contain a verified answer. The interface is local and has no authentication;
it is not configured for public hosting.


## Offline demo

Turn on **Offline demo** near the top of the Streamlit page to explore a bundled,
fictional library-policy example. It includes a sample answer, prepared citations,
source excerpts, an illustrative run summary, and TXT/JSON downloads clearly marked
`DEMO`. It does not perform search, fetch pages, call AI providers, read provider
settings, or access research history. No API keys or internet connection are needed
once the app and its dependencies are installed.

The example is hand-authored, not a real or freshly verified research result.
Its displayed example budget is simulated; actual provider calls are zero. Demo
data uses a separate format and is never assigned to the real report or incomplete
run session fields. Turn the toggle off to return to existing results and unsaved
work. Normal session expiry and browser-refresh limitations still apply.

## Research depth

The Streamlit **Research depth** control offers three presets. **Standard** is
selected initially. These are maximums, not targets or guarantees of answer
quality, runtime, token consumption, or provider quota availability:

| Mode | Rounds | Searches | Sources / page reads | AI calls |
| --- | ---: | ---: | ---: | ---: |
| Quick | 1 | 2 | 4 / 4 | 12 |
| Standard | 2 | 5 | 8 / 8 | 32 |
| Thorough | 3 | 8 | 12 / 12 | 64 |

Every value is capped by the existing environment configuration. For example,
the default two-round cap limits Thorough to two rounds. The interface shows the
effective limits before submission. Modes also cap planned sub-questions at
2/3/5, searches per round at 2/3/5, and results per search at 3/4/5 respectively.
Changing modes does not edit `.env`, switch models, or start a request.

All modes use the same grounding and semantic verification checks. The configured
finalization reserve is preserved and must be at least two calls; a mode cannot
start if its AI-call ceiling leaves no capacity beyond that reserve. The AI-call
limit counts application-level calls; SDK retries and token usage are not estimated.

Completed reports, incomplete entries, and their exports retain the selected mode
and effective limits. Changing the selector does not relabel an earlier result.
Older history entries remain readable without invented mode information. CLI
runs continue to use their configured budgets directly.

## Offline answer-quality scenarios

Run `python -m pytest tests/test_offline_evaluation.py -v` for nineteen fixed research
scenarios covering grounded answers, citation provenance, semantic rejection,
conflicting evidence, and insufficient evidence, including multiple sources and
claims, swapped citations, rejection when any claim lacks support, and adversarial
source instructions. They exercise the production
graph with scripted providers and blocked network connections, using no quota.
These are safeguard regression checks, not a benchmark of live Gemini quality:
semantic verdicts and extracted evidence are supplied by the fixtures. Injection
cases inspect real extraction/synthesis prompt boundaries and use a dummy secret;
passing them does not establish live model resistance to prompt injection.
See [the evaluation guide](evals/README.md) for case descriptions and report output.

## Saved research history

Turn on **Compare saved runs**, then select two distinct entries to view their
answers, recorded outcomes, research depth, budget usage, and sources side by
side. Comparison uses all saved history regardless of sidebar filters and keeps
your current result and unsaved work intact. Incomplete runs show no completed
answer; missing older metadata is labeled as not recorded. Source overlap uses
exact available HTTP(S) URLs, not matching page contents. Counts are saved budget
reservations rather than token usage or billing. This read-only view uses no
provider calls and does not assign an answer-quality score.
Choose **Download comparison (.md)** after selecting both runs to keep a portable
Markdown snapshot with their saved entry IDs, questions, answers, outcomes, depth,
budget usage, warnings, source lists, and shared/distinct source URLs. Research
text is fenced as literal text so embedded HTML and image markup remain inert.
The download uses the same loaded snapshots as the displayed comparison and does
not change history or use provider quota.

Select a saved report or incomplete run, then choose **Research this question
again** in the sidebar to fill in its original question and recorded depth.
Review or edit them above and click **Start research** when ready. Preparing a
retry makes no provider calls and preserves the original history entry; starting
research creates a fresh run rather than resuming collected evidence. The saved
depth mode uses current configured limits. If no depth was recorded, your current
selection is kept. Unsaved results must be saved or explicitly discarded before
preparing another run.

Expand **Filter saved research** to combine search with run status, saved research
depth, and optional From/Through dates. Dates are inclusive and use UTC, matching
the history labels. **Not recorded** finds older runs without depth metadata;
it does not infer a mode from their budget. Invalid date ranges show a warning.
**Clear history filters** resets these filters while keeping your search text.
Filtering does not open a report, change unsaved work, or use provider quota.
Unreadable depth metadata is excluded when a depth filter is selected.

The **Research outcome** filter distinguishes verified answers, runs with no
evidence, answers rejected by verification, insufficient budget for final answer
checks, other runs without a verified answer, and interrupted runs. These use the
saved outcome, not guesses based on report wording or citation counts. Older
completed reports with missing or unknown outcomes appear under **Not recorded**;
incomplete runs remain **Interrupted run**, even without a summary. Unreadable
records are excluded when an outcome filter is selected. Outcome filters combine
with all other filters and search, and reset with **Clear history filters**.

Use **Search saved research** in the sidebar to find text in saved questions,
report bodies, source titles, or evidence excerpts, including incomplete runs.
Search matches a literal substring, ignores letter case, and trims outer spaces;
results keep their newest-first order. Empty searches show all entries. Searches
run locally without provider calls and do not change saved or unsaved research.
Damaged entries remain searchable by question, but their invalid contents are
excluded so they cannot prevent other results from appearing.
Each search result includes a short match preview; selecting it shows the excerpt
below the list before you open the report. Previews use original text and label
the matching field: question, report text, source title, or evidence excerpt.
When multiple fields match, the first in that order is shown. Long excerpts are
trimmed around the match with ellipses. Previews do not change saved content.

### Backup and restore

In the sidebar, expand **History backup & restore**, choose **Prepare history
backup**, then download the JSON file. The snapshot includes all saved completed
and incomplete runs, evidence, citations, depth settings, and recorded summaries.
It excludes unsaved session results, provider configuration, and demo data. Keep
the file private because it contains your research content.

Choose a backup file to preview its entry counts, then select **Restore missing
history**. All records are validated before writing. Restore merges in one
transaction: identical IDs/content are skipped, and conflicting IDs cancel the
entire restore without overwriting existing history. Repeating a restore is safe.
The active result and unsaved work are not replaced. Invalid files, unsupported
versions, broken evidence references, and backups over 20 MB or 5,000 entries
are rejected. Restore checks data structure; it does not reverify research claims.

Backups use a consistent SQLite read snapshot while the app is running. Preparing
a new download is explicit; an earlier prepared download does not automatically
include later research. If any stored record is invalid, export fails rather than
silently omitting it. Neither backup nor restore makes provider calls.

### Run summaries

New web-interface runs retain a **Run summary** in history and exports, including
incomplete runs. The table compares committed research-round, search, source,
page-read, and AI-call budgets against the actual limits for that run. These are
reservation counters, not provider billing totals: failed attempts consume budget,
and finalization reserves both calls before either executes. A failure during the
first finalization call can therefore still account for two reserved AI calls.

The graph records why research stopped before finalization spends its reserve:
sufficient evidence, no further/new questions, or the first budget blocking another
round. The summary separately states whether finalization produced verified
citations, found no evidence, lacked finalization budget, or discarded a rejected
answer. An incomplete run records its safe failure category instead. No rejected
drafts, raw provider errors, or secrets are added to summaries.

Older saved results still open normally and show that detailed summaries were not
recorded. Missing AI/page-read totals and stop reasons are never reconstructed
from current settings or guessed from old report text.

Completed web-interface runs are automatically saved to
`.local/research_history.sqlite` inside this project. The **Saved research**
sidebar lets you search by question and open any saved report, including its
citations, source URLs, supporting evidence, usage counts and diagnostic notices.
Opening or downloading a saved report makes no provider calls. Runs returning
an insufficient-evidence fallback are saved with their original warnings;
graph runs interrupted by an exception are saved separately as **Incomplete**
entries with the question, collected sources and evidence, counters, last recorded
stage, and a fixed explanation of why the run stopped. Configuration failures
before graph execution do not create an entry.

History is local to this project and shared by browser sessions on this computer.
It is excluded from Git. It contains report content and extracted evidence, not
API keys, runtime contexts, or full fetched webpages. Back up the SQLite file
while the app is stopped if you want an independent copy of your history.
Save failures leave the report available in the current session with downloads
and a retry button. Reports created before this feature are not recovered from
closed sessions; an existing open result can be saved with **Save report to history**.

### Incomplete research

An incomplete entry has no report, draft answer, or citations. Opening it shows
collected source excerpts and offers a JSON download, with an explicit warning
that the run did not produce a verified answer. Rejected synthesis text and raw
exception messages are never included in its snapshot. Existing completed
reports remain available in history; they are not displayed as the interrupted
run's answer. Verification rejections already handled by the graph continue to
produce their existing, clearly labeled fallback reports.

Incomplete entries capture the last graph state delivered to the interface when
an ordinary execution exception is caught. They are **not checkpoints**: stopping
the process, closing the app during execution, cancellation, or a power failure
does not guarantee a saved partial run. Opening an entry makes no provider calls
and does not resume execution. Submitting the question again starts a fresh run.
If local saving fails, the current incomplete entry remains downloadable and can
be saved using **Save incomplete run**. Download or save it before navigating away.

### Protecting unsaved work

If a report or incomplete run has not been saved, starting new research and
opening another history entry are blocked before any provider call or replacement.
The current result stays available for download and retrying its save. This also
protects results left in sessions from before automatic history saving existed.
After saving, click the desired action again; blocked requests are not queued.

If saving remains unavailable, download your result, open **Discard unsaved
research…**, and select **Discard unsaved session copies** to continue. This
explicit action removes only unsaved session results, not saved history or
downloaded files. Session-only results still cannot survive a browser refresh,
session expiry, or process shutdown, so save or download them before leaving.
