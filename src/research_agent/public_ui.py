"""Streamlit pieces shared by the hosted entry point.

- render_outcome: the report/partial-evidence display used by both hosted modes.
  Untrusted research text is always rendered as plain text.
- render_live_research: the password-gated section shown above the offline demo.

Unlock state lives only in this browser session's server-side state and is tied
to the current password hash: after the operator changes the password, every
open session is locked again on its next interaction.
"""
import hashlib

import streamlit as st

import research_agent.public_service as public
from research_agent.incomplete import ResearchInterrupted
from research_agent.public_access import LoginLockedError, LoginUnavailableError, check_password
from research_agent.public_execution import ResearchTimeoutError
from research_agent.public_limits import AdmissionError


def render_outcome(partial, result, *, key: str = '') -> None:
    if partial is not None:
        st.warning(partial.stop_message)
        st.download_button('Download incomplete evidence', partial.json_export(),
                           file_name='incomplete-evidence.json', mime='application/json',
                           key=f'{key}partial_download')
    if result is None:
        return
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
                       file_name='research-report.txt', mime='text/plain',
                       key=f'{key}report_download')
    st.download_button('Download evidence', result.json_export(),
                       file_name='research-evidence.json', mime='application/json',
                       key=f'{key}evidence_download')


def _fingerprint(stored_hash: str) -> str:
    return hashlib.sha256(stored_hash.encode('utf-8')).hexdigest()


def clear_live_session() -> None:
    for name in [name for name in st.session_state if str(name).startswith('live_')]:
        del st.session_state[name]


def _render_unlock(stored_hash: str, usage_db) -> None:
    st.write('Run the agent on your own question with live Gemini and Tavily. '
             'Ask the author for the password.')
    with st.form('live_access', clear_on_submit=True):
        st.text_input('Password', type='password', key='live_password_input', max_chars=256)
        submitted = st.form_submit_button('Unlock live research')
    if not submitted:
        return
    password = st.session_state.get('live_password_input') or ''
    try:
        accepted = check_password(usage_db, password, stored_hash)
    except LoginLockedError as exc:
        st.warning(str(exc))
        return
    except LoginUnavailableError as exc:
        st.error(str(exc))
        return
    if not accepted:
        st.warning('Incorrect password.')
        return
    st.session_state['live_unlocked_for'] = _fingerprint(stored_hash)
    st.rerun()


def _render_research(live) -> None:
    st.success('Live research is unlocked in this browser tab.')
    st.caption('Questions and source pages are sent to Gemini and Tavily. '
               'Do not enter confidential information. Nothing is saved on this server.')
    st.caption(f'Per run: 1 research round, 2 searches, 4 sources, 12 AI calls and a three-minute '
               f'cutoff. Shared by everyone with the password: {public.LIVE_HOURLY} runs per hour, '
               f'{public.LIVE_DAILY} per day, one at a time. Failed runs count.')
    with st.form('live_research'):
        st.text_area('Research question', key='live_question', max_chars=2000)
        start = st.form_submit_button('Start live research')
    st.button('Lock live research', on_click=clear_live_session)
    if start:
        st.session_state.pop('live_result', None)
        st.session_state.pop('live_partial', None)
        with st.status('Researching', expanded=True) as status:
            try:
                result = public.run_live_question(
                    st.session_state.get('live_question', ''),
                    lambda message: status.update(label=message), config=live)
            except (public.PublicInputError, public.ServiceBusyError, AdmissionError) as exc:
                status.update(label='Research did not start', state='error')
                st.warning(str(exc))
            except ResearchTimeoutError as exc:
                status.update(label='Research timed out', state='error')
                st.warning(str(exc))
            except ResearchInterrupted as exc:
                status.update(label='Research incomplete', state='error')
                st.session_state['live_partial'] = exc.partial
            except Exception:
                status.update(label='Research could not finish', state='error')
                st.error('The run could not finish. The free-tier AI service may be busy or out '
                         'of quota; try again in a few minutes. No new report was released.')
            else:
                st.session_state['live_result'] = result
                status.update(label='Research finished', state='complete')
    render_outcome(st.session_state.get('live_partial'), st.session_state.get('live_result'),
                   key='live_')


def render_live_research(live, usage_db) -> None:
    """Password gate, then the live question form. `live` must be a ready LiveAccessConfig."""
    stored_hash = live.password_hash.get_secret_value()
    unlocked_for = st.session_state.get('live_unlocked_for')
    unlocked = unlocked_for == _fingerprint(stored_hash)
    if unlocked_for is not None and not unlocked:
        clear_live_session()  # the password changed since this tab was unlocked
    with st.expander('🔐 Live research (password required)', expanded=unlocked):
        if unlocked:
            _render_research(live)
        else:
            _render_unlock(stored_hash, usage_db)
