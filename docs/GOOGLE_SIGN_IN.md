# Google sign-in setup

Target: https://research-agent-frosty.streamlit.app/
Public entry point: `public_app.py` (not the existing local `app.py`).
This document prepares configuration; it does not enable or deploy live service.

## Account setup

1. Open https://console.cloud.google.com/auth/overview and select or create
   a project dedicated to Research Agent. Complete any Google account/terms
   steps yourself.
2. Configure the required Branding details using your actual support and
   contact information. Do not invent a domain you own.
3. Choose an External audience for personal Google accounts. Keep the app in
   Testing and add your own Google account as a test user. Do not publish yet.
4. In Clients, create a Web application client named Research Agent Web.
5. Add this exact authorized redirect URI:
   `https://research-agent-frosty.streamlit.app/oauth2callback`
   JavaScript origins are not required for Streamlit's server-side login flow.
6. For a separate local test, optionally add:
   `http://localhost:8505/oauth2callback`
   Use that exact host and port when launching the local public entry point.
7. Keep the Client ID and Client secret in your password manager. Never paste
   the secret into chat, commit it, or put real credentials in the example file.

## Configure the application when ready for staging

Install the authentication dependencies with `python -m pip install ".[public]"`.
The Cloud dependency installation must also include this extra before switching
the entry point. A reproducible Cloud dependency lock is still pending.

Use `google-auth.secrets.example.toml` as the structure for the Cloud app's Secrets
settings. Replace placeholders there, not in the tracked example file. Generate
a cookie secret with a cryptographically secure generator, e.g. Python's
`secrets.token_urlsafe(64)`, and save it directly in the private configuration.
Keep the same cookie secret across deployments; replacing it invalidates cookies.
Do not add Gemini or Tavily keys: visitors supply them in their own sessions.

For local testing, use the gitignored `.streamlit/secrets.toml` and change the
redirect URI to `http://localhost:8505/oauth2callback`. Preserve existing secrets
when editing this file. Start `python -m streamlit run public_app.py --server.port
8505 --server.address 127.0.0.1` and open `http://localhost:8505`.

Keep public research disabled until abuse limits, provider timeouts, deployment
checks and an authenticated staging test are complete. During a controlled test,
enable the public entry point only with the OAuth audience still restricted to
your test accounts. Check sign-in, sign-out, expired sessions, two-account
isolation, wrong keys and key clearing. Do not display `st.user` or tokens.

Google configuration and Cloud secrets alone do not update the currently hosted
app. Publishing code/switching entry points and live testing are separate steps.

Official guide: https://docs.streamlit.io/develop/tutorials/authentication/google
