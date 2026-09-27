"""Public BYOK entry point. Does not import the local app or history UI."""
from pathlib import Path
import sys
import time
import math

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

import streamlit as st
from research_agent.incomplete import ResearchInterrupted
from research_agent.public_service import (
    PublicConfig, PublicInputError, ServiceBusyError, AdmissionError,
    ResearchTimeoutError, run_visitor_question,
)


def clear_private_session():
    # This entry point owns the whole session; no history or user data survives.
    for key in list(st.session_state):
        del st.session_state[key]


st.set_page_config(page_title='Research Agent', page_icon='🔎')
st.title('Research Agent')
st.write('Research a question using your own Gemini and Tavily API keys.')
try:
    config = PublicConfig()
except Exception:
    st.error('The public service configuration is invalid. Contact the operator.')
    st.stop()

if not config.enabled and not config.auth_preview:
    st.info('The public research service is not enabled yet.')
    st.stop()

if st.user.get('is_logged_in') is not True:
    clear_private_session()
    st.info('Sign in before starting research.')
    if st.button('Sign in with Google'):
        try:
            st.login()
        except Exception:
            st.error('Sign-in is unavailable. Contact the operator.')
    st.stop()
# Do not use an email address or browser-supplied field as tenant identity.
issuer, subject = st.user.get('iss'), st.user.get('sub')
expiry = st.user.get('exp')
if (not isinstance(issuer, str) or not issuer or
        not isinstance(subject, str) or not subject or
        isinstance(expiry, bool) or not isinstance(expiry, (int, float)) or
        not math.isfinite(expiry) or not time.time() < expiry):
    clear_private_session()
    st.error('Your sign-in has expired or is incomplete. Sign out and sign in again.')
    if st.button('Sign out'):
        st.logout()
    st.stop()
identity = (issuer, subject)
if st.session_state.get('_identity') != identity:
    clear_private_session()
    st.session_state['_identity'] = identity
if st.button('Sign out'):
    clear_private_session()
    st.logout()
    st.stop()

if not config.enabled:
    # Authentication can be tested without exposing any credential inputs or
    # starting research. PUBLIC_ENABLED remains the research kill switch.
    st.success('Google sign-in is working. You are signed in.')
    st.info('Live research is disabled during setup. No API keys are needed.')
    st.stop()
if not config.gemini_model.strip():
    st.error('The research model has not been configured. Contact the operator.')
    st.stop()

st.info('Your keys are sent to this server and used to contact Gemini and Tavily. '
        'Provider charges and quotas apply to your accounts. Questions and source '
        'content are sent to those providers. Do not enter confidential information.')
st.caption('This version keeps results in your current session only. It does not '
           'save keys or research to a history database. Download results before leaving.')
st.caption('Usage protection stores a hashed account identifier and run timestamps on this server, '
           'with records older than 24 hours removed at the next admission check. '
           'It stores no keys, questions or reports.')
st.button('Clear keys and results', on_click=clear_private_session)

with st.form('research'):
    st.text_input('Gemini API key', type='password', key='gemini_key', max_chars=512)
    st.text_input('Tavily API key', type='password', key='tavily_key', max_chars=512)
    st.text_area('Research question', key='question', max_chars=2000)
    st.caption('Per run: up to 1 research round, 2 searches, 4 sources and 12 AI calls. '
               'Up to 3 starts per rolling hour and 10 per rolling day per account, '
               'with one active run per account and a three-minute cutoff. Failed runs count. '
               'Shared service limits also apply. These limits are not a price guarantee.')
    st.checkbox('I authorize this run using my keys and understand provider charges may apply.', key='consent')
    submitted = st.form_submit_button('Start research')

if submitted:
    if not st.session_state.consent:
        st.warning('Confirm authorization before starting research.')
    else:
        st.session_state.pop('result', None)
        st.session_state.pop('partial', None)
        with st.status('Researching', expanded=True) as status:
            try:
                result = run_visitor_question(
                    st.session_state.question, lambda message: status.update(label=message),
                    gemini_key=st.session_state.gemini_key,
                    tavily_key=st.session_state.tavily_key, model=config.gemini_model,
                    identity=identity,
                )
            except (PublicInputError, ServiceBusyError, AdmissionError) as exc:
                status.update(label='Research did not start', state='error')
                st.warning(str(exc))
            except ResearchTimeoutError as exc:
                status.update(label='Research timed out', state='error')
                st.warning(str(exc))
            except ResearchInterrupted as exc:
                status.update(label='Research incomplete', state='error')
                st.session_state.partial = exc.partial
            except Exception:
                status.update(label='Research could not finish', state='error')
                st.error('The request could not finish. Check your keys, provider quota, '
                         'and provider availability. No new report was released.')
            else:
                st.session_state.result = result
                status.update(label='Research finished', state='complete')

partial = st.session_state.get('partial')
if partial is not None:
    st.warning(partial.stop_message)
    st.download_button('Download incomplete evidence', partial.json_export(),
                       file_name='incomplete-evidence.json', mime='application/json')

result = st.session_state.get('result')
if result is not None:
    st.subheader('Research report')
    st.text(result.report.question)
    # Plain text avoids executing untrusted HTML or loading remote Markdown images.
    st.text(result.report.content)
    if not result.report.citations:
        st.warning('No answer with verified citations was produced.')
    for issue in result.issues:
        st.warning(issue)
    with st.expander('Sources and supporting passages', expanded=True):
        evidence = {item.id: item for item in result.evidence}
        sources = {item.id: item for item in result.sources}
        for index, citation in enumerate(result.report.citations, 1):
            st.text(f'[{index}] {citation.claim_text}')
            for evidence_id in citation.evidence_ids:
                item = evidence[evidence_id]
                source = sources[item.source_id]
                st.text(source.title or source.domain)
                st.text(source.final_url or source.url)
                st.text(item.excerpt)
    if result.summary:
        st.text(result.summary.text_export())
    st.download_button('Download report', result.text_export(),
                       file_name='research-report.txt', mime='text/plain')
    st.download_button('Download evidence', result.json_export(),
                       file_name='research-evidence.json', mime='application/json')
