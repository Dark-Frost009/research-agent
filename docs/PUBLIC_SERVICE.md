# Public service: bring your own keys

Status: local implementation foundation, **not approved for public deployment**.

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
AI-call reservations. Two runs can execute simultaneously per Python process.
The in-process semaphore is only a capacity guard: it is not a persistent or
distributed abuse limiter. Provider retries and token charges are not represented
by reservation counts. Public output renders untrusted research as plain text.

## Required before launch

1. Target: https://research-agent-frosty.streamlit.app/ with mandatory sign-in.
   Configure Google sign-in using [GOOGLE_SIGN_IN.md](GOOGLE_SIGN_IN.md) and
   verify its callback on this exact host.
2. Add durable per-user/global request quotas and edge connection/request limits;
   implement bounded execution time and provider timeouts/retry policy.
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
