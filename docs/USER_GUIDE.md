# User guide: the local web app

How to use the local Streamlit application. Setup is in the
[README](../README.md#quick-start).

The local app is for private use on your own machine. It has no
authentication, and its saved history is shared by every browser session of the
same app instance. Do not expose it as a multi-user service. The hosted
version is a separate entry point, described in
[PUBLIC_SERVICE.md](PUBLIC_SERVICE.md).

## Starting the app

The local app runs only with an explicit opt-in. Without it, `app.py` shows the
fictional public demo instead.

```powershell
$env:RESEARCH_AGENT_LOCAL = "1"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
```

Then open <http://localhost:8502>.

## Running research

Type a question, choose a research depth and start. The page shows progress by
stage ("Searching the web", "Reading source pages", …), then the answer first,
followed by its citations, supporting evidence and all sources researched. Each
claim's sources open in a new tab, and the supporting excerpts are one click
away.

Every request gets fresh dependencies and a fresh workspace. Error details and
credentials are never shown.

### Research depth

The presets are maximums, not targets. Every preset is also capped by your
`.env` limits.

| Mode | Rounds | Searches | Sources / page reads | AI calls |
|---|---|---|---|---|
| Quick | 1 | 2 | 4 / 4 | 12 |
| Standard (default) | 2 | 5 | 8 / 8 | 32 |
| Thorough | 3 | 8 | 12 / 12 | 64 |

All modes use the same grounding and verification checks and keep the
finalization reserve. Changing the mode does not edit `.env`, switch models or
start research.

### Outcomes

The report and its run summary show how the run ended:

- **Verified:** an answer with trusted citations.
- **Partial:** verified citations, but research stopped before the Critic was
  satisfied, for example because the budget ran out.
- **No evidence:** nothing usable was found, so no answer is given.
- **Rejected:** the proposed answer failed grounding or verification and was
  discarded. The run summary gives the reason in plain language, for example
  "A supporting quote did not match its cited evidence word for word."
- **Incomplete:** the run was interrupted, for example by a provider quota. The
  sources and evidence collected so far are kept, but there is no answer.

## Downloads

- **Report (.txt):** answer, claims, sources, limitations and run summary.
- **Evidence (.json):** the report, every evidence excerpt and every source,
  with IDs that link citation → evidence → source. The live-evaluation tools
  read this file.

## Saved research history

Completed and incomplete web runs are saved automatically to
`.local/research_history.sqlite`, which Git ignores. The sidebar can search
questions, report text, source titles and evidence, filter by status, depth,
outcome and date, and reopen any entry. Opening or downloading a saved report
makes no provider calls. "Research this question again" prepares a fresh run;
it does not resume the old one.

History stores reports and extracted evidence, never API keys, runtime objects
or full web pages.

### Comparing saved runs

Turn on **Compare saved runs** to compare two entries: answers, outcomes, depth,
budget use and sources, including shared and distinct URLs. It is read-only, and
a Markdown snapshot can be downloaded.

### Backup and restore

Under **History backup & restore**, download a JSON backup of completed and
incomplete runs. Restore validates the whole file first. It then adds missing
entries in one transaction, skips identical ones, and cancels on conflicting
IDs rather than overwriting anything. Backups contain research content, so keep
them private.

### Unsaved work

If a result has not been saved, actions that would replace it are blocked, and
the result stays downloadable. A result that exists only in the session does not
survive closing the browser or stopping the app.

## Run summaries

Each run records budget use (rounds, searches, sources, page reads, AI calls),
why research stopped, and how finalization ended. These are reservation
counters, not provider billing: a failed attempt still counts.

## Incomplete runs

An incomplete entry shows the question, the last stage reached, the stop reason,
the collected sources and evidence, and a JSON download. It is not a checkpoint:
opening it does not resume execution. A crash or forced stop may not save a
partial run, because no persistent checkpointer is used.

## Offline demo

Under **Developer tools**, turn on **Offline demo** to explore a hand-written,
fictional example. It makes no AI calls, no searches and no page fetches, needs
no keys, and never touches your history.

## Offline evaluation dashboard

Also under **Developer tools**. It runs the bundled offline scenarios in a
separate process and shows results, failure groups, history and baseline
comparisons. It makes no live provider calls. See
[EVALUATION.md](EVALUATION.md).

## Command line

```powershell
research-agent "What is retrieval-augmented generation and what problem does it solve?"
```

This prints the report and its citations. CLI runs use the `.env` budgets
directly, not the depth presets.
