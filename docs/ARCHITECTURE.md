# Architecture

This document explains how the Research Agent works and why it is built this
way. The [README](../README.md) gives the short version.

## Contents

- [The research workflow](#the-research-workflow)
- [Stages](#stages)
- [The research loop](#the-research-loop)
- [Grounded synthesis](#grounded-synthesis)
- [Semantic verification](#semantic-verification)
- [When an answer is rejected](#when-an-answer-is-rejected)
- [Budgets](#budgets)
- [State and reducers](#state-and-reducers)
- [The production graph](#the-production-graph)
- [Dependency wiring](#dependency-wiring)
- [Workspace isolation](#workspace-isolation)
- [Provider errors and retries](#provider-errors-and-retries)
- [Security model](#security-model)
- [Design decisions](#design-decisions)
- [Limitations](#limitations)

## The research workflow

```text
User question
     │
     ▼
Reserve research iteration
     │
     ▼
Planner ──► sub-questions
     │
     ▼
Search ──► search results
     │
     ▼
Source admission ──► sources
     │
     ▼
Safe web fetching ──► page text (transient)
     │
     ▼
Evidence extraction ──► grounded evidence
     │
     ▼
Critic
     ├── evidence insufficient and budget allows
     │        │
     │        ▼
     │   follow-up questions ──► search again (next iteration)
     │
     └── sufficient, or no budget left
              │
              ▼
         Synthesis
              │
              ▼
   Deterministic grounding checks
              │
              ▼
   Semantic verification (separate AI pass)
              │
              ▼
   Trusted citations ──► ResearchReport
```

Every external call (AI model, search, page fetch) is bounded by whole-run
budgets, so the graph cannot keep searching, fetching or calling the model
indefinitely.

## Stages

### 1. Planner

Turns the question into focused `SubQuestion` objects using structured output.
It runs only for the first iteration and never searches itself. Follow-up
iterations reuse the Critic's questions instead (see
[design decisions](#design-decisions)).

### 2. Search

Each authorized sub-question is sent to Tavily. Search is split into
`reserve_search` (decide exactly how many queries are allowed) and
`execute_search` (run only those), so search cannot fan out beyond the budget.

### 3. Source admission

Search results become durable `Source` objects, bounded by
`MAX_SOURCES_PER_RUN` and deduplicated by stable IDs derived from the URL. The
original search-result URL stays the identity even when a page redirects; the
destination is stored separately as `final_url`.

### 4. Safe web fetching

`WebPageFetcher` fetches authorized sources with protection against:

- unsafe URL schemes
- private, loopback and link-local addresses
- IPv4-mapped and NAT64-embedded addresses
- unsafe redirects: every redirect target is validated again before the next request
- excessive redirect chains
- oversized responses and unsupported content types

Raw page text is transient: it is never stored in durable graph state.

### 5. Evidence extraction

Each fetched page is linked to its `Source` and `SubQuestion`. The model
proposes excerpts, and Python keeps only those that appear word for word in
the fetched page text. The model cannot invent supporting quotes. Each
candidate is validated on its own: mismatches are rejected, the exact ones are
kept, and the provenance rule is never weakened.

### 6. Critic

Asks whether the accumulated evidence is enough to answer the question. It sees
grounded evidence only, never raw pages, and returns either `sufficient = True`
or a list of gaps and follow-up questions.

## The research loop

```text
Evidence ──► Critic
               ├── sufficient ──► finalization
               └── insufficient
                     ├── no viable budget ──► finalization
                     └── budget available
                           ──► reserve iteration
                           ──► follow-up sub-questions
                           ──► search ──► fetch ──► evidence ──► Critic
```

The router checks that at least one more research cycle is actually viable
before starting it, so an exhausted budget finalizes cleanly instead of
entering a broken partial iteration.

## Grounded synthesis

Synthesis receives validated evidence only, never source documents. Before the
call, Python assigns each excerpt a handle (`E1`, `E2`, …), so the model never
sees or produces internal IDs. The model returns an answer plus, for every
claim, the handles it relies on and a supporting quote for each.

Python then checks, deterministically:

1. every evidence handle exists
2. every supporting quote is an exact substring of the excerpt it cites
3. no claim repeats an evidence handle, and no claim is duplicated
4. every claim appears word for word in the answer text
5. citation IDs are created only after all of this passes

A citation is never accepted just because the model says it is correct.

## Semantic verification

An exact quote proves where text came from. It does not prove that the quote
supports the claim. "The study enrolled 500 participants" is a real quote, but
it does not support "the treatment reduced mortality by 40%".

So a separate AI pass checks two things:

- **Support:** is each claim actually justified by its quotes?
- **Coverage:** does the answer contain factual statements that no claim covers?

It is a separate adversarial pass, not an independent one: it uses the same
provider. Trusted `Citation` objects exist only after both the deterministic
checks and this pass succeed.

## When an answer is rejected

If grounding validation or semantic verification fails, the whole proposed
answer is discarded: the policy is atomic and fail-closed. The report then
contains a fixed message ("The available evidence could not be verified
strongly enough…") and no citations. Rejected text and verifier diagnostics are
never copied into state, reports or history.

What is kept is one fixed **reason code**, chosen by Python at the check that
failed. It never contains model output. The codes are defined in
`src/research_agent/graph/rejection.py`:

| Code | Meaning |
|---|---|
| `quote_mismatch` | a supporting quote was not found word for word in its evidence |
| `synthesis_format` | another structural rule was broken, such as a claim missing from the answer text |
| `verifier_format` | the verifier's own reply was malformed or inconsistent |
| `unsupported_claim` | the verifier judged a claim not supported by its quotes |
| `incomplete_coverage` | the verifier found factual statements without a citation |

Provider failures and programming errors stay operational failures. They are
not relabeled as verification rejections.

## Budgets

### Whole-run limits

Default limits (configurable in `.env`):

```text
MAX_RESEARCH_ITERATIONS=2
MAX_SUB_QUESTIONS=5
MAX_SEARCH_QUERIES_PER_RUN=8
MAX_SEARCH_QUERIES_PER_ITERATION=5
MAX_SEARCH_RESULTS_PER_QUERY=5
MAX_SOURCES_PER_RUN=12
MAX_SOURCE_FETCHES_PER_RUN=12
MAX_LLM_CALLS_PER_RUN=64
FINALIZATION_LLM_RESERVE=2
```

State records committed usage (`iteration_count`, `search_queries_used`,
`source_fetches_used`, `llm_calls_used`). Remaining capacity is calculated, not
stored. An authorized attempt consumes budget even if the provider then fails.

### Reservation before side effects

Every external operation is split into a reserve step and an execute step:

```text
reserve_source_fetch ──► execute_source_fetch
```

The reserve step decides exactly what is authorized, and only that batch reaches
the provider. This applies to Planner, Search, page fetches, evidence
extraction, Critic and finalization, so no worker can fan out past the
whole-run limits.

### Protected finalization

Synthesis and semantic verification are reserved together as an atomic pair
(`FINALIZATION_LLM_RESERVE=2`). Optional research can never consume this
reserve, and the pair is never partially authorized.

## State and reducers

`ResearchState` is a `TypedDict` holding durable data only: the question,
sub-questions, search results, sources, evidence, critique, draft, citations,
final report, usage counters and errors. Clients, credentials, page text and
authorization objects live outside it, in a transient workspace.

LangGraph combines updates from nodes with **reducers**:

- list fields such as evidence are appended;
- usage counters add deltas, and invalid deltas are rejected;
- sources are merged deterministically by ID, using a total order (fetch status,
  then metadata completeness, then a field tiebreak). The result is therefore
  independent of the order in which parallel branches finish.

## The production graph

The compiled graph lives in `src/research_agent/graph/builder.py`:

```text
START
  → reserve_initial_iteration
  → reserve_planner → execute_planner
  → reserve_search → execute_search
  → admit_sources
  → prepare_evidence_collection_plan
  → reserve_source_fetch → execute_source_fetch
  → reserve_evidence_extraction → execute_evidence_extraction
  → reserve_critique → execute_critique
  → route_after_critique
       ├── continue_research
       │     → reserve_follow_up_iteration
       │     → adapt_critique_follow_ups
       │     → reserve_search …
       └── finalize
             → reserve_finalization → execute_finalization
             → assemble_final_report
             → END
```

Routing decisions are conditional edges, not logic hidden inside nodes.

## Dependency wiring

`src/research_agent/bootstrap.py` is the composition root. It turns
`Settings` into the Gemini client, the Tavily client, `WebPageFetcher` and the
budget policy, then assembles a `ResearchGraphContext`. Provider code is kept
out of graph nodes, so tests can substitute scripted providers.

```text
Settings
   ├── GeminiLLMClient
   ├── TavilySearchClient
   ├── WebPageFetcher
   └── BudgetPolicy
          ▼
   Research services ──► ResearchGraphContext ──► compiled LangGraph
```

## Workspace isolation

Every run needs a fresh `ResearchGraphContext` and `TransientWorkspace`. Use
`build_research_context(settings)` or `build_research_application(settings)`;
the CLI and the app create fresh dependencies for every request. The first node
atomically claims the workspace, so reusing one, sequentially or concurrently,
raises an error. The claim stays consumed after success, failure or
`clear_all()`.

## Provider errors and retries

The Gemini adapter translates SDK API errors and HTTP transport failures into
safe, generic messages (`LLMUnavailableError`, `LLMRateLimitError`, …).
Unexpected programming errors keep their original type instead of being
mislabeled as outages.

- **Local app.** Temporary failures (HTTP 408, 500, 502, 503, 504) are retried
  by the Gemini SDK: up to 4 attempts with about 2, 4 and 8 seconds between
  them. Live evaluation showed that without this, one momentary 503 failed a
  whole step. A retry re-sends the same logical call, so the run budget is
  unaffected.
- **No retry on 429.** Rate and quota limits are never retried: a spent daily
  quota will not recover, and callers stop cleanly on it.
- **Public app.** One attempt with a 30-second timeout, because each visitor
  run has a strict time limit (see [PUBLIC_SERVICE.md](PUBLIC_SERVICE.md)).

## Security model

LLM output and web content are both treated as untrusted.

- **Network boundary.** Search URL → validation → request → redirect target →
  validation again → page.
- **Prompt injection.** Retrieved text is passed to every AI stage as data,
  wrapped and labeled untrusted, with instructions never to follow commands
  inside it. This reduces prompt-injection risk; it does not eliminate it.
- **Deterministic checks.** Quotes, handles and citation links are checked by
  Python, not by asking a model.
- **Display.** Untrusted titles, claims and excerpts are rendered as plain text,
  never Markdown, so a page cannot inject links or tracking images.
- **Secrets.** Provider errors and credentials are never displayed. `.env` and
  Streamlit secrets are excluded from Git.

## Design decisions

### Why the Critic runs before synthesis

The graph does not synthesize, critique, then research more and synthesize
again. Synthesis and verification spend the protected reserve, so research
completes first, and then the answer is written and verified once.

### Why follow-ups don't call the Planner again

The Critic already returns concrete follow-up questions. Converting them into
`SubQuestion` objects deterministically avoids an extra AI call, duplicated
work and extra nondeterminism.

### Why SearchResult, Source and Evidence are separate

- A **search result** is what the provider returned.
- A **source** is a real page with its own fetch status.
- **Evidence** is a specific excerpt from a source.

Keeping them distinct is what makes the trail question → sub-question → search
result → source → evidence → citation traceable.

## Limitations

- **No crash durability.** No persistent LangGraph checkpointer is used. Budget
  accounting is correct during a run, but not crash-durable; crash-resume would
  need a checkpointer, durable reservations and idempotency.
- **One provider each.** Gemini is the only LLM provider and Tavily the only
  search provider wired in.
- **Local app is private.** It has no authentication, and its SQLite history
  is shared by sessions of the same app instance.
- **Source reliability is not checked.** A claim is verified against its own
  quote, not against whether the source is right (see
  [EVALUATION.md](EVALUATION.md)).
- **Live results vary.** Search results, pages, provider availability and
  model output change between runs.
