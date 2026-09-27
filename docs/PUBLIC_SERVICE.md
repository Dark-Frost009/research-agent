# Public service: bring your own keys

Status: Google sign-in staging deployed and verified; live public research remains disabled pending launch checks.

`public_app.py` is separate from the existing local `app.py` and offline demo.
Visitors supply both Gemini and Tavily keys. The application does not fall back
to the operator's environment keys or `.env`. Keys are processed on the server,
so this is not a browser-only or zero-knowledge service. No keys, questions or
results are written to the local history database by this entry point.

Reports stay in the Streamlit session, with explicit downloads and a clear
session action. Do not configure session tracing, HTTP body/header logging,
or exception reporting that records keys or research content. Password widgets
mask the display; they do not encrypt server memory. Session lifetime is not a
guarantee of immediate memory erasure on disconnect.

## Operator configuration

- `PUBLIC_ENABLED` defaults to `false` (fail closed).
- `PUBLIC_GEMINI_MODEL` must name the model selected and tested by the operator.
- Sign-in is mandatory. Configure Streamlit OIDC before enabling public traffic.
  Identity must include `iss`, `sub` and future `exp`.
- No provider keys belong in the hosted application's environment or secrets.
- Keep Streamlit XSRF and CORS protections enabled. HTTPS is required externally.

The offline tests inject authenticated identities and stub provider activity. Start with `python -m streamlit run public_app.py`.
Do not expose the local `app.py` as a live multi-user service.

Authentication setup: https://docs.streamlit.io/develop/concepts/connections/authentication
Use the Streamlit authentication extra or its documented Authlib dependency
when building the deployment environment; never commit `secrets.toml`.

## Current limits

Runs are limited to 1 round, 2 searches, 4 sources, 4 page reads and 12 application
AI-call reservations. Public research uses a disposable subprocess, killed and
reaped by the parent at 180 seconds. A worker watchdog also exits at 180 seconds
if its parent disappears. In-flight remote requests may still complete or incur
charges after local termination; a timeout does not refund provider usage.
No partial evidence is recovered on a hard timeout. Completed and interrupted
results cross an anonymous pipe as JSON; keys are never put in process arguments,
temporary files or logs. Worker stdout/stderr logging is suppressed.

Public Gemini requests have a 30-second SDK timeout and one attempt (no automatic
SDK retries). Public Tavily searches have a 15-second timeout, basic search depth,
and automatic parameter selection disabled; the current SDK uses a requests
session with no automatic retries. Page reads retain their 15-second timeout.
Transport timeouts are not a substitute for the enforced overall deadline.
These settings do not change the local research app's provider defaults.

Admission uses SQLite transactions, shared across sessions and processes that
use the same database file. Limits count starts, including failed and timed-out
runs, in rolling windows:

- Per verified issuer/subject: 3 starts/hour, 10/day, one active run.
- Whole host: 60 starts/hour, 200/day, two active runs.
- A crashed run occupies its slot for at most 240 seconds from admission.
- A missing identity, unreadable/corrupt database or lock timeout blocks research.
- Clearing session keys or signing out does not reset counters.

`PUBLIC_USAGE_DB` optionally sets the SQLite file path. Its default is
`.local/public-usage.sqlite3` under the application root, ignored by Git.
The store contains only a SHA-256 hash of issuer/subject, a random run ID and
timestamps. This hash is a pseudonymous identifier, not anonymous data.
Rows older than 24 hours are deleted at the next admission check after their
active lease expires; storage is not a forensic secure-erasure guarantee.
Backups of this database require their own retention policy.

Counters survive process restarts on the same disk. Streamlit Community Cloud
disk replacement/redeployment can reset them. Multiple hosts with separate files
do not share these limits. This is a single-host protection, not durable distributed
abuse prevention or an edge request limiter. Keep live public research disabled
until that deployment limitation has an accepted solution. No paid storage has
been provisioned. Public output renders untrusted research as plain text.

## Required before launch

1. Target: https://research-agent-frosty.streamlit.app/ with mandatory sign-in.
   Google callback and logout were verified on this host. See
   [GOOGLE_SIGN_IN.md](GOOGLE_SIGN_IN.md) for configuration; OAuth is still in Testing mode.
2. Resolve durable shared quota storage across host replacements and add edge
   connection/request limits. Verify subprocess termination and load on the Linux
   host; local offline tests do not establish hosted behavior.
3. Build a reproducible deployment (locked dependencies, container, health check,
   HTTPS, secrets, non-root execution, restricted outbound networking).
4. Review SDK logging and credential handling end to end. Add privacy/retention
   notice and operator contact. Decide whether persistent user history is needed.
5. Verify fresh installation, authenticated isolation, expiry/logout, failed
   providers, concurrent use, load behavior, restarts and rollback on the host.
6. Run a consented live smoke test with visitor-owned test credentials. Never
   describe the mocked/offline tests as proof of live provider reliability.

The desktop presentation assets and existing local history remain unchanged.


## Sign-in-only staging

On the existing Cloud `app.py` entry point, `PUBLIC_AUTH_PREVIEW=true` routes
to the public sign-in interface before local history or developer tools load.
Keep `PUBLIC_ENABLED=false`: signed-in users see a confirmation and logout,
without API-key inputs, research execution or report access. `PUBLIC_DEMO_ONLY`
can remain true for rollback: set `PUBLIC_AUTH_PREVIEW=false` to return to the demo.
`requirements.txt` installs the public authentication extra on Cloud. Dependency
versions still need a reviewed reproducible lock before the live-service launch.

In Streamlit Cloud Secrets, use quoted root-level flags (`PUBLIC_AUTH_PREVIEW = "true"`,
`PUBLIC_ENABLED = "false"`). Streamlit promotes strings to environment variables,
not TOML booleans. The nested `auth.expose_tokens` remains a TOML boolean.
