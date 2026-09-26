Research Agent
A production-oriented research agent built with Python, LangGraph, Gemini, Tavily, and Streamlit.
The agent takes a research question, breaks it into focused sub-questions, searches the web, safely fetches source pages, extracts grounded evidence, critiques whether that evidence is sufficient, optionally performs another research iteration, and produces a final answer whose trusted citations must pass both deterministic grounding checks and semantic verification.
The project is built around one principle:
An LLM should not be trusted merely because it produced a plausible answer.

Research, evidence extraction, synthesis, citation creation, and semantic verification are therefore separated into explicit stages with bounded budgets and fail-closed validation.
What the Agent Does
Given a question such as:
What is retrieval-augmented generation, and what is one limitation it does not fully solve?
the production workflow is approximately:
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
Deterministic Grounding Validation
     │
     ▼
Semantic Verification
     │
     ▼
Trusted Citations
     │
     ▼
ResearchReport
The research loop is bounded by whole-run budgets so the graph cannot continue searching, fetching, or calling the LLM indefinitely.
Key Features
- LangGraph-based production research workflow
- Google Gemini as the production LLM provider
- Tavily web search
- Safe webpage fetching with SSRF protections
- NAT64-aware IP validation
- Explicit redirect validation
- Structured Pydantic outputs for LLM stages
- Deterministic evidence and citation handles
- Exact quote grounding
- Candidate-level evidence grounding
- Evidence-to-claim provenance validation
- Semantic support verification before trusted citations are created
- Evidence sufficiency critic
- Bounded multi-iteration research loop
- Whole-run search, fetch, source, iteration, and LLM budgets
- Protected finalization LLM reserve
- Deterministic LangGraph reducers
- Fresh workspace isolation for production graph invocations
- CLI entry point
- Streamlit web interface
- Local SQLite research history
- Search and filters for saved research
- Saved-run comparison
- History backup and restore
- Incomplete-run preservation
- Research-depth presets
- Run summaries and budget reporting
- Offline demo
- Offline evaluation dashboard
- Local release-readiness reports
- Public portfolio demo-only mode
- Extensive unit, orchestration, graph-slice, integration, UI, persistence, and production-graph tests
- Manual live-provider smoke testing
Architecture
1. Planner
The Planner converts the original research question into focused SubQuestion objects.
The Planner:
- receives the original research question
- produces structured output through Gemini
- is used only for the initial research iteration
- does not execute searches itself
- consumes one optional-research LLM budget unit when authorized
Follow-up iterations do not call the Planner again.
The Critic already produces concrete follow-up questions, so those questions are converted deterministically into SubQuestion objects without spending another Planner LLM call.
2. Search
Each authorized sub-question is sent to the configured search provider.
Production currently uses:
Tavily
Search execution is separated into reservation and execution:
reserve_search
      ↓
execute_search
The reservation stage determines exactly how many queries are allowed before any provider side effect occurs.
This prevents uncontrolled search fan-out.
3. Source Admission
Search results are converted into durable Source objects.
Source admission is bounded by:
MAX_SOURCES_PER_RUN
Sources are deduplicated using stable IDs.
The original search-result URL remains the stable source identity even when a webpage redirects during fetching.
The fetched destination is stored separately as:
final_url
4. Safe Web Fetching
Authorized sources are fetched through WebPageFetcher.
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
5. Evidence Extraction
Fetched pages are passed to the EvidenceExtractor.
Evidence is always associated with:
Source
   +
SubQuestion
The extraction stage requires evidence excerpts to be grounded directly in fetched page text.
The LLM cannot create arbitrary supporting quotes.
Returned evidence must pass deterministic validation before becoming durable graph state. Candidate-level validation keeps exact grounded candidates and rejects mismatches rather than weakening the provenance rules.
6. Critic
After evidence collection, the Critic asks:
Is the accumulated evidence sufficient to answer the original question?
The Critic receives grounded evidence only.
It does not receive raw webpage contents.
It can return:
sufficient = True
or:
sufficient = False
gaps = [...]
follow_up_questions = [...]
If more research is justified and whole-run budgets allow it, the graph begins another research iteration.
Otherwise, the graph proceeds to finalization using the evidence already collected.
Research Loop
The production loop is:
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
The router checks whether at least one additional research cycle is actually viable before entering another iteration.
Grounded Synthesis
Final synthesis does not receive arbitrary source documents.
It receives validated Evidence.
Before synthesis, evidence receives deterministic handles:
E1
E2
E3
...
The synthesis model must return claims together with evidence references and supporting quotes.
Conceptually:
claim_text
evidence_handle
supporting_quote
The system validates that:
1. every referenced evidence handle exists
2. every supporting quote is an exact substring of the referenced evidence excerpt
3. duplicate evidence handles are rejected where they violate the synthesis contract
4. duplicate synthesized claims are rejected
5. returned claim text is represented in the generated answer
6. factual claims must be grounded through the synthesis contract
7. citation IDs are not created until grounding validation succeeds
This prevents citations from being accepted solely because the LLM says they are correct.
Semantic Verification
Exact quote matching proves that a quote came from the referenced evidence.
It does not by itself prove that the quote actually supports the claim.
The project therefore includes a separate semantic verification stage.
After deterministic grounding validation, another structured LLM call evaluates claim-to-evidence support:
Claim
   ↓
Referenced Evidence
   ↓
Does the evidence actually support the claim?
Trusted Citation objects are created only after both deterministic provenance checks and semantic verification succeed.
If deterministic grounding validation or semantic verification rejects the proposed answer, production finalization discards that answer and returns a ResearchReport with a fixed safe fallback and zero trusted citations.
Rejected draft content and verifier diagnostics are not copied into graph state or the final report.
Provider failures, generic structured-response errors, and programming errors remain operational failures rather than being relabeled as successful verification rejection.
The validation gates are fail-closed; they are not weakened to force an answer.
Protected Finalization
Finalization requires two protected LLM calls:
1. Synthesis
2. Semantic verification
The budget system reserves those calls as an atomic finalization pair.
Optional research work is not allowed to consume this protected capacity.
Default:
FINALIZATION_LLM_RESERVE=2
If the required finalization capacity is unavailable, the pair is not partially authorized.
Whole-Run Budget System
The agent uses centralized whole-run budgets rather than allowing individual nodes to make unlimited provider calls.
Default limits include:
MAX_RESEARCH_ITERATIONS=2
MAX_SUB_QUESTIONS=5

MAX_SEARCH_QUERIES_PER_RUN=8
MAX_SEARCH_QUERIES_PER_ITERATION=5
MAX_SEARCH_RESULTS_PER_QUERY=5

MAX_SOURCES_PER_RUN=12
MAX_SOURCE_FETCHES_PER_RUN=12

MAX_LLM_CALLS_PER_RUN=64
FINALIZATION_LLM_RESERVE=2
Durable state tracks actual committed usage:
iteration_count
search_queries_used
source_fetches_used
llm_calls_used
Remaining capacity is calculated rather than stored.
Provider attempts consume budget once execution has been authorized, even when the provider later fails.
Reservation Before Side Effects
External operations use a reservation/execution pattern.
Example:
reserve_source_fetch
        ↓
execute_source_fetch
The reservation determines exactly what work is authorized.
Only that authorized batch is allowed to reach the external provider.
The pattern is used for operations such as:
Planner LLM calls
Search calls
Source fetches
Evidence extraction LLM calls
Critic LLM calls
Finalization LLM calls
This prevents worker-level fan-out from bypassing whole-run limits.
State and Reducers
ResearchState contains durable research data such as:
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
Additive usage counters use reducers with delta semantics.
Source merging is deterministic and independent of branch execution order.
Transient authorization objects and fetched page contents are stored outside durable research data in the graph workspace.
Production Graph
The compiled production graph lives in:
src/research_agent/graph/builder.py
The main route is approximately:
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
  │              ↓
  │      reserve_follow_up_iteration
  │              ↓
  │      adapt_critique_follow_ups
  │              ↓
  │           search...
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
Production Dependency Wiring
The production composition root lives in:
src/research_agent/bootstrap.py
It builds the configured providers, graph services, and budget policy before assembling a fresh production research application.
Production provider logic is intentionally kept out of graph nodes.
Conceptually:
Settings
   │
   ├── GeminiLLMClient
   ├── TavilySearchClient
   ├── WebPageFetcher
   └── BudgetPolicy
            │
            ▼
        Research services
            │
            ▼
   ResearchGraphContext
            │
            ▼
      Compiled LangGraph
Workspace Isolation
Each production graph invocation requires a fresh ResearchGraphContext and TransientWorkspace.
Use build_research_context(settings) for a fresh context or build_research_application(settings) to build the graph and context together.
The CLI creates a fresh application for every research run.
The compiled graph can be reused only with fresh contexts. The initial node atomically claims the workspace before clearing temporary data, reserving budget, or calling providers.
Sequential or concurrent reuse of the same workspace raises an error.
The claim remains consumed after completion, failure, or clear_all(). A new request should use fresh dependencies rather than resetting a used workspace.
This guard applies to the production graph entry point, not direct calls to individual orchestration nodes.
It does not add retries, persistent checkpoint resumability, or guarantees about provider clients that a caller manually shares between otherwise separate contexts.
Project Structure
The project uses a src/ layout. Important files and modules include:
research-agent/
│
├── app.py
│
├── src/
│   └── research_agent/
│       ├── bootstrap.py
│       ├── config.py
│       ├── main.py
│       ├── ui_service.py
│       ├── history.py
│       ├── backup_ui.py
│       ├── comparison_ui.py
│       ├── demo.py
│       ├── depth.py
│       ├── evaluation_dashboard.py
│       ├── incomplete.py
│       ├── release_readiness.py
│       ├── run_summary.py
│       │
│       ├── graph/
│       │   ├── budget.py
│       │   ├── builder.py
│       │   ├── context.py
│       │   ├── execution.py
│       │   ├── state.py
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
├── evals/
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
The tree above highlights the main architecture rather than every test or support file.
Requirements
Current runtime requirements include:
Python >= 3.12
Pydantic 2.x
pydantic-settings 2.x
LangGraph
Tavily Python SDK
Requests
BeautifulSoup
Google GenAI SDK
HTTPX
Streamlit
Development dependencies include:
Pytest
The authoritative dependency constraints are in pyproject.toml.
Installation
The commands below are written for Windows PowerShell.
Clone the repository and enter the project directory.
Create a virtual environment:
python -m venv .venv
Activate it:
.venv\Scripts\Activate.ps1
Install the project in editable mode:
pip install -e .
For development dependencies:
pip install -e ".[dev]"
Environment Configuration
Copy the example environment file:
Copy-Item .env.example .env
Then configure the environment.
Example:
# --------------------------------------------------
# Application
# --------------------------------------------------

APP_ENV=development
LOG_LEVEL=INFO

# Keep false for the normal local/live application.
# Set true for a public portfolio deployment that must not expose
# live provider calls or shared local research history.
PUBLIC_DEMO_ONLY=false


# --------------------------------------------------
# LLM
# --------------------------------------------------

LLM_PROVIDER=gemini

# Legacy/default model.
# Used as the fallback when a role-specific model is blank.
LLM_MODEL=YOUR_GEMINI_MODEL

# Optional model for planning, evidence extraction, and critique.
LLM_WORKER_MODEL=

# Optional model for synthesis and semantic verification.
LLM_FINAL_MODEL=

LLM_API_KEY=YOUR_GEMINI_API_KEY


# --------------------------------------------------
# Web Search
# --------------------------------------------------

SEARCH_PROVIDER=tavily
TAVILY_API_KEY=YOUR_TAVILY_API_KEY


# --------------------------------------------------
# Research Limits
# --------------------------------------------------

MAX_RESEARCH_ITERATIONS=2
MAX_SUB_QUESTIONS=5

MAX_SEARCH_QUERIES_PER_RUN=8
MAX_SEARCH_QUERIES_PER_ITERATION=5
MAX_SEARCH_RESULTS_PER_QUERY=5

MAX_SOURCES_PER_RUN=12
MAX_SOURCE_FETCHES_PER_RUN=12

MAX_LLM_CALLS_PER_RUN=64
FINALIZATION_LLM_RESERVE=2


# --------------------------------------------------
# Network / Extraction
# --------------------------------------------------

REQUEST_TIMEOUT_SECONDS=15
MAX_RESPONSE_BYTES=2097152
MAX_TEXT_CHARS=100000
MAX_REDIRECTS=5
LLM_WORKER_MODEL and LLM_FINAL_MODEL are optional. When either is blank, that role falls back to LLM_MODEL.
The .env file is ignored by Git.
.streamlit/secrets.toml is also ignored so a local Streamlit secrets file cannot be committed accidentally if one is created later.
Never commit API keys.
Running the Agent
CLI
The project exposes a console command:
research-agent "What is retrieval-augmented generation and what problem does it solve?"
Equivalent module invocation:
python -m research_agent.main "What is retrieval-augmented generation and what problem does it solve?"
Help:
research-agent --help
A successful run prints a final research report and citation information.
Example shape:
RESEARCH REPORT
============================================================

<grounded final answer>

CITATIONS
------------------------------------------------------------
[citation-id] <supported claim>
  Evidence: <evidence-id>
Local Web Interface
Install the development dependencies, then run from the project folder:
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
Open:
http://localhost:8502
With PUBLIC_DEMO_ONLY=false, the full local/live application is available.
The Streamlit interface includes:
- question input
- research-depth selection
- live Gemini/Tavily research
- progress status
- verified/partial/no-verified-answer status badges
- answer-first report layout
- progressively disclosed research notes
- citations and supporting evidence
- all researched sources
- run details
- TXT and JSON downloads
- automatically saved completed research
- incomplete-run snapshots when an ordinary execution exception is captured
- saved-history search and filters
- saved-run comparison
- backup and restore
- "Research this question again"
- unsaved-work protection
- offline demo
- offline evaluation dashboard
Every submitted live request gets fresh dependencies and a fresh workspace.
Viewing, comparing, reopening, or downloading saved results does not rerun research.
Provider error details and credentials are not displayed.
The full live interface is designed for local/private use. It has no user authentication, and its SQLite history is shared by browser sessions that use the same application instance.
Public Portfolio Deployment
The project includes an explicit safe public mode:
PUBLIC_DEMO_ONLY=true
When enabled, the Streamlit entry point switches to a bundled portfolio demonstration before the shared history store, live provider path, developer tools, or normal research controls are initialized.
Public demo-only mode:
Bundled fictional sample
        │
        ├── sample report
        ├── sample citations
        ├── sample evidence
        ├── illustrative run summary
        └── demo downloads

No Gemini calls
No Tavily calls
No webpage fetching
No live research form
No shared research-history UI
No backup/restore UI
No saved-run comparison
No developer-tools toggles
The mode displays an explicit notice explaining that the public deployment is a portfolio demo and that live Gemini/Tavily research and saved research history are disabled.
This mode exists because the normal local application uses one project-local SQLite history store and does not implement public-user authentication, per-user history ownership, public rate limiting, or public provider-quota controls.
Setting PUBLIC_DEMO_ONLY=true does not delete or simplify the local application. It changes only the hosted execution path. Set it back to false to use the full local/live interface.
A future unrestricted public live-research deployment would require additional controls such as authenticated users, per-user data isolation, and provider-abuse protection.
Offline Demo
In the normal local application, turn on Offline demo under Developer tools to explore a bundled fictional library-policy example.
It includes:
- sample answer
- prepared citations
- source excerpts
- illustrative run summary
- TXT and JSON downloads marked as demo content
The offline demo does not:
- search the web
- fetch source pages
- call AI providers
- require API keys
- access research history
- assign its data to real report/incomplete-run session fields
The example is hand-authored and is not a real or freshly verified research result.
Its displayed budget is simulated; actual provider calls are zero.
In normal local mode, turn the Offline demo toggle off to return to the existing session.
In PUBLIC_DEMO_ONLY=true, the same bundled demo is rendered as the public portfolio experience and no toggle is exposed.
Research Depth
The Streamlit Research depth control offers three presets.
Standard is selected initially.
These are maximums, not targets or guarantees of answer quality, runtime, token consumption, or provider quota availability:
Mode	Rounds	Searches	Sources / page reads	AI calls
Quick	1	2	4 / 4	12
Standard	2	5	8 / 8	32
Thorough	3	8	12 / 12	64


Every preset is capped by the existing environment configuration.
For example, a global two-round environment cap still limits Thorough to two rounds.
The interface shows effective limits before submission.
Modes also cap planned sub-questions, searches per round, and results per search.
Changing modes does not:
- edit .env
- switch models
- start research
All modes use the same grounding and semantic verification checks.
The configured finalization reserve is preserved.
Completed reports, incomplete entries, and exports retain the selected mode and effective limits.
Older history entries remain readable without inventing mode information.
CLI runs continue to use their configured budgets directly.
Saved Research History
Completed web-interface runs are automatically saved to:
.local/research_history.sqlite
The Saved research sidebar can:
- list completed and incomplete runs
- search saved questions
- search report text
- search source titles
- search evidence excerpts
- filter by run status
- filter by research depth
- filter by outcome
- filter by date range
- open saved reports
- prepare a saved question for a fresh research run
Opening or downloading a saved report makes no provider calls.
History is local to the project and excluded from Git.
It contains saved report content and extracted evidence, not API keys, runtime contexts, or full fetched webpages.
The history store is intended for local/private use. It is shared by browser sessions using the same application instance and is therefore deliberately bypassed by public demo-only mode.
Compact Saved-Report View
When a completed or incomplete saved report is open, the new-research workflow is collapsed into:
Start new research
This keeps the report near the top of the page while preserving all research controls.
Choosing Research this question again prepares the saved question and recorded depth and expands the new-research area so the prepared values can be reviewed before starting a fresh run.
Preparing a retry makes no provider calls.
Starting the retry creates a fresh run rather than resuming previously collected evidence.
Saved-Run Comparison
Turn on Compare saved runs to select two distinct saved entries and compare:
- answers
- recorded outcomes
- research depth
- budget usage
- source lists
- shared/distinct source URLs
Incomplete runs are represented as incomplete rather than being given a fabricated answer.
Older missing metadata is labeled as not recorded.
Comparison is read-only and makes no provider calls.
A Markdown comparison snapshot can also be downloaded.
Backup and Restore
In the sidebar, expand History backup & restore.
A prepared JSON backup can contain:
- completed runs
- incomplete runs
- evidence
- citations
- depth settings
- recorded summaries
It excludes:
- unsaved session results
- provider configuration
- API keys
- demo data
Keep backups private because they contain research content.
Restore validates the entire backup before writing.
Restore merges missing entries in one transaction. Identical existing entries are skipped, while conflicting IDs cancel the restore rather than overwriting existing content.
Backups over the configured validation limits are rejected.
Neither backup nor restore makes provider calls.
Public demo-only mode does not initialize or expose backup/restore controls.
Run Summaries
New web-interface runs retain a run summary in history and exports, including incomplete runs when a safe partial snapshot is available.
The summary reports committed budget usage for resources such as:
Research rounds
Searches
Sources
Page reads
AI calls
These counts are reservation counters, not provider billing totals.
Failed authorized attempts consume budget.
Finalization reserves both protected calls before execution, so a failure during the first finalization call can still account for the pair of reserved calls.
The summary also records why research stopped and whether finalization produced verified citations, found no evidence, lacked finalization capacity, or discarded a rejected answer.
A run can contain trusted citations while still being presented as a partial answer if the research loop ended for a reason other than evidence sufficiency.
Older results without detailed summaries remain readable and are labeled accordingly rather than being reconstructed from current settings.
Incomplete Research
An incomplete entry has no verified completed answer.
Opening it shows:
- the original question
- the last recorded stage
- safe stop information
- collected sources
- collected evidence
- recorded run details
- JSON download
Rejected synthesis text and raw exception messages are not included in the incomplete snapshot.
Incomplete entries are not checkpoints.
Opening one does not resume execution.
Submitting the question again starts a fresh run.
Stopping the process, closing the app during execution, cancellation, or power failure does not guarantee that a partial run is saved because the production graph does not currently use a persistent LangGraph checkpointer.
Protecting Unsaved Work
If a report or incomplete run has not been saved, actions that would replace the active result are blocked.
The current result stays available for download and save retry.
If saving remains unavailable, the user can download the result and explicitly discard unsaved session copies before navigating away.
The discard action removes only unsaved session results. It does not delete saved history or downloaded files.
Session-only results cannot survive browser-session expiry, refresh in all cases, or process shutdown, so important work should be saved or downloaded before leaving.
Offline Evaluation Dashboard
The local Streamlit application includes an Offline evaluation dashboard under Developer tools.
The dashboard runs bundled evaluation scenarios in a separate process and displays:
- scenario results
- grouped failure categories
- saved evaluation history
- baseline comparisons
- downloadable JSON summaries
The evaluation dashboard does not make live provider calls.
It uses deterministic/scripted dependencies and blocked network access for the bundled offline cases.
Evaluation history is separate from research history.
Public demo-only mode does not expose the evaluation dashboard.
Offline Answer-Quality Scenarios
Run:
python -m pytest tests/test_offline_evaluation.py -v
for the bundled fixed research scenarios.
The scenarios cover areas such as:
- grounded answers
- citation provenance
- semantic rejection
- conflicting evidence
- insufficient evidence
- multiple sources and claims
- swapped citations
- unsupported claims
- adversarial source instructions
They exercise the production graph with scripted providers and blocked network connections.
These are safeguard regression checks, not a benchmark of live Gemini quality.
Semantic verdicts and extracted evidence used by fixtures are deterministic test inputs.
Passing adversarial prompt-boundary tests does not establish universal resistance to prompt injection.
See:
evals/README.md
for the evaluation guide.
Local Release-Readiness Report
Run:
python -m research_agent.release_readiness
to run the offline release-readiness process and write Markdown and JSON reports under:
.local/readiness/
Use:
python -m research_agent.release_readiness --output-dir PATH
to choose another output directory.
The readiness process records a source fingerprint and compares scenario results with the latest saved evaluation when an appropriate baseline is available.
Missing or unreadable baselines are explicitly unassessed.
Tests that fail, error, skip, or produce incomplete results prevent an offline pass.
Source changes during the run also prevent a pass.
Exit codes:
0  Offline checks passed
1  Offline checks need attention
2  Readiness check could not complete
An offline pass does not certify live Gemini quality or unrestricted public deployment readiness.
The public portfolio demo-only path is a separate deployment-safety control.
Example Live Result
During development, the production graph has been run against real Gemini and Tavily services.
Live smoke testing has exercised the full path:
Planner
→ Tavily Search
→ Source Admission
→ Safe Web Fetching
→ Evidence Extraction
→ Critique
→ Synthesis
→ Deterministic Grounding Validation
→ Semantic Verification
→ ResearchReport
A later live RAG run after the grounding and synthesis hardening produced separately grounded claims with trusted citations through the current verification path.
Live output is intentionally not expected to be identical across runs because search results, webpages, provider availability, and LLM outputs are external and dynamic.
The agent is designed to fail closed when the available evidence cannot be verified strongly enough rather than fabricate a confident grounded answer.
Testing
Run the full suite:
pytest -q
Current verified development checkpoint:
2124 passed
The suite includes coverage for areas such as:
- configuration
- schemas
- budget policy
- URL safety
- web search
- web extraction
- Gemini adapter behavior
- Planner
- Search
- Source admission
- Source fetching
- Evidence extraction
- Evidence collection
- Critic
- Synthesis
- deterministic grounding validation
- semantic verification
- orchestration
- LangGraph slices
- graph topology
- production graph execution
- bootstrap/composition root
- CLI
- Streamlit UI
- research depth
- local history
- saved-run comparison
- backup and restore
- incomplete research
- run summaries
- offline demo isolation
- public demo-only isolation
- offline evaluation
- release-readiness behavior
Provider-facing automated tests use mocked, scripted, or deterministic dependencies unless a manual live smoke test is being performed explicitly.
Security Model
The project treats both LLM output and external web content as untrusted.
Important network boundaries include:
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
External webpage content is treated as untrusted prompt data.
Prompts instruct LLM stages not to follow instructions embedded inside retrieved webpages.
The project does not treat source text as trusted system instructions.
Other relevant controls include:
- URL-scheme restrictions
- private/loopback/link-local address rejection
- NAT64-aware validation
- redirect revalidation
- response-size limits
- content-type checks
- bounded provider calls
- structured-output validation
- exact provenance validation
- semantic citation verification
- generic user-safe provider error messages
- local secret files excluded from Git
- public demo-only mode that bypasses live providers and shared research history
The project does not claim that these controls make arbitrary public live-provider exposure safe. That is why the public portfolio deployment uses demo-only mode.
Citation Trust Model
The citation pipeline intentionally separates different questions:
Did the quote come from the evidence?
                ↓
Does that evidence exist?
                ↓
Does the evidence support the claim?
                ↓
Only then create a trusted Citation
Therefore:
LLM-generated citation ≠ trusted citation
A citation becomes trusted only after passing the project's deterministic and semantic verification gates.
Why the Critic Runs Before Final Synthesis
The Critic evaluates evidence before finalization.
The graph intentionally does not use:
Synthesize
→ Critique
→ Research More
→ Synthesize Again
because synthesis and semantic verification consume protected finalization capacity.
Instead:
Research
→ Critique
→ Research More if needed
→ Final Synthesis once
→ Semantic Verification once
This protects finalization capacity and avoids spending final-answer calls before research is complete.
Why Follow-Up Research Does Not Call Planner Again
The Critic already returns explicit follow-up research questions.
Calling Planner again would:
- spend another LLM call
- duplicate work
- introduce unnecessary nondeterminism
- consume optional-research budget
Instead, follow-up questions are deterministically converted into new SubQuestion objects.
Current Limitations
The production graph is currently compiled without a persistent LangGraph checkpointer.
The reservation/execution architecture ensures normal reducer ordering during a running graph execution, but it does not by itself provide crash-durable accounting across process termination.
True crash-resume guarantees would require additional work such as:
persistent checkpointer
+
durable operation reservations
+
idempotency strategy
The project therefore does not claim crash-safe resumability.
Other current limitations include:
Gemini is the only production LLM provider wired in bootstrap.py

Tavily is the only production search provider wired in bootstrap.py

The full live Streamlit mode is local/private and has no user authentication

Local research history is shared by browser sessions on the same app instance

No public per-user history ownership is implemented

No unrestricted public provider-quota/rate-limit layer is implemented

Public deployment is therefore intentionally demo-only

Search and webpage availability can vary between runs

Some webpages may be inaccessible or unsupported

External provider availability and rate limits can affect live runs
Design Principles
Both the CLI and Streamlit use shared graph execution configuration from:
graph/execution.py
The graph step limit scales with the configured maximum research iterations and leaves room for finalization.
Provider budgets remain enforced separately by the graph.
The Gemini adapter translates known SDK API errors and HTTP transport failures into safe provider-facing messages.
Unexpected programming errors retain their original type at the adapter boundary instead of being mislabeled as provider outages.
No automatic adapter retry layer is added.
The project emphasizes:
Bounded execution
Deterministic validation
Explicit orchestration
Provider isolation
Evidence provenance
Fail-closed behavior
Security before convenience
Testability
Clear separation between transient and durable state
Progressive disclosure in the UI
Local/private data boundaries
The goal is not merely to demonstrate that an LLM can search the web.
The goal is to demonstrate how an LLM-powered research system can be engineered so that search, evidence, citations, budgets, provider calls, persistence, failure behavior, and deployment boundaries are explicit and testable.
Development Status
Current development checkpoint:
Production LangGraph          Implemented
Planner                       Implemented
Tavily Search                 Implemented
Safe Web Fetching             Implemented
Evidence Extraction           Implemented
Critic / Research Loop        Implemented
Whole-Run Budgets             Implemented
Grounded Synthesis            Implemented
Semantic Verification         Implemented
Safe Rejection Report         Implemented
Citation Validation           Implemented
Production Composition        Implemented
CLI                           Implemented
Streamlit UI                  Implemented
Research Depth Presets        Implemented
Local Research History        Implemented
Saved-Run Comparison          Implemented
Backup / Restore              Implemented
Incomplete-Run Capture        Implemented
Run Summaries                 Implemented
Offline Demo                  Implemented
Offline Evaluation Dashboard  Implemented
Public Portfolio Demo Mode    Implemented
Live Provider Run             Verified
Full Test Suite               2124 passing
Persistent Checkpointing      Not yet implemented
Authenticated Public Live UI  Not implemented
License
No license has been selected yet.