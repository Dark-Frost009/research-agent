"""Local Streamlit entry point. Start with: python -m streamlit run app.py."""

from datetime import timezone
from html import escape
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

# Cloud authentication staging runs before importing local history or widgets.
# PublicConfig reads environment only; it never loads the local .env file.
from research_agent.public_service import PublicConfig
try:
    _public_config = PublicConfig()
except Exception:
    st.error("The public service configuration is invalid. Contact the operator.")
    st.stop()
if _public_config.enabled or _public_config.auth_preview:
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parent / "public_app.py"), run_name="__main__")
    st.stop()


from research_agent.backup_ui import render_backup_controls
from research_agent.config import Settings
from research_agent.depth import DEPTH_NAMES
from research_agent.history import (
    OUTCOME_FILTERS,
    HistoryError,
    get_history_store,
)
from research_agent.incomplete import (
    IncompleteResearch,
    ResearchInterrupted,
)
from research_agent.ui_service import (
    friendly_error,
    preview_depth,
    run_question,
    safe_source_url,
)


PRESERVED_WIDGET_KEYS = (
    "question",
    "research_depth",
    "history_search",
    "history_selection",
    "history_status",
    "history_depth",
    "history_start",
    "history_end",
    "history_outcome",
    "compare_runs",
    "compare_first",
    "compare_second",
)

DEPTH_DESCRIPTIONS = {
    "Quick": "Fast overview",
    "Standard": "Balanced research",
    "Thorough": "Deeper investigation",
}


st.set_page_config(
    page_title="Research Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_ui_styles() -> None:
    """Apply presentation-only styling without changing app behavior."""

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1120px;
            padding-top: 3.25rem;
            padding-bottom: 4rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.4rem;
        }

        .ra-eyebrow {
            margin-bottom: 0.65rem;
            color: #9aa4b2;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }

        .ra-title {
            max-width: 760px;
            margin: 0;
            font-size: clamp(2.25rem, 5vw, 4.1rem);
            font-weight: 760;
            line-height: 1.02;
            letter-spacing: -0.045em;
        }

        .ra-subtitle {
            max-width: 760px;
            margin-top: 1rem;
            margin-bottom: 2rem;
            color: #a9b0bb;
            font-size: 1.02rem;
            line-height: 1.65;
        }

        .ra-section-label {
            margin-top: 0.2rem;
            margin-bottom: 0.65rem;
            color: #9aa4b2;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.10em;
            text-transform: uppercase;
        }

        .ra-question {
            margin: 0.35rem 0 1.25rem;
            color: #d7dce3;
            font-size: 1.03rem;
            line-height: 1.55;
        }

        .ra-answer {
            margin: 0.5rem 0 1.6rem;
            padding: 1.25rem 1.35rem;
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.025);
            font-size: 1.04rem;
            line-height: 1.75;
        }

        .ra-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.38rem;
            margin-bottom: 0.8rem;
            padding: 0.34rem 0.68rem;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .ra-badge-verified {
            background: rgba(40, 167, 69, 0.12);
        }

        .ra-badge-partial {
            background: rgba(255, 193, 7, 0.12);
        }

        .ra-badge-incomplete {
            background: rgba(220, 53, 69, 0.12);
        }

        .ra-muted {
            color: #929aa6;
            font-size: 0.88rem;
            line-height: 1.55;
        }

        div[data-testid="stExpander"] {
            border-radius: 12px;
        }

        div[data-testid="stForm"] {
            border-radius: 14px;
            padding: 1rem 1rem 0.3rem;
        }

        .stButton > button,
        .stDownloadButton > button,
        .stLinkButton > a {
            border-radius: 10px;
        }

        [data-testid="stMetric"] {
            padding: 0.25rem 0;
        }

        hr {
            margin-top: 2rem;
            margin-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def unsaved_entries():
    """Include legacy sessions and hidden results, not just failed-save flags."""

    return [
        (value_key, saved_key)
        for value_key, saved_key in (
            ("completed_research", "saved_report_id"),
            ("incomplete_research", "saved_incomplete_id"),
        )
        if st.session_state.get(value_key) is not None
        and not st.session_state.get(saved_key)
    ]


def preserve_widget_state() -> None:
    """Keep real widget values when offline views hide normal widgets."""

    for key in PRESERVED_WIDGET_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def prepare_research_again(record_id):
    if unsaved_entries():
        st.session_state["research_again_notice"] = (
            "warning",
            "Save or explicitly discard your unsaved research before "
            "preparing another run. Your question and depth have been kept.",
        )
        return

    try:
        saved_result = get_history_store().load(record_id)
    except HistoryError as exc:
        st.session_state["research_again_notice"] = (
            "warning",
            str(exc),
        )
        return

    st.session_state["question"] = (
        saved_result.question
        if isinstance(saved_result, IncompleteResearch)
        else saved_result.report.question
    )

    saved_depth = getattr(saved_result, "depth", None)
    if saved_depth is not None:
        st.session_state["research_depth"] = saved_depth.name

    depth_note = (
        "The saved depth is selected using your current configured limits."
        if saved_depth
        else (
            "No depth was recorded, so your current depth selection "
            "has been kept."
        )
    )

    st.session_state["research_again_notice"] = (
        "info",
        "Question ready above. "
        + depth_note
        + " Review it and click Start research to begin a fresh run using "
        "provider quota. No research has started; the original entry is kept.",
    )


def render_run_summary(result, *, heading: bool = False) -> None:
    """Render exact run details without making them the primary UI."""

    if heading:
        st.subheader("Run summary")

    summary = getattr(result, "summary", None)

    if summary is None:
        st.caption(
            "A detailed run summary was not recorded for this older result."
        )
        a, b, c = st.columns(3)
        a.metric("Research rounds", result.iterations)
        b.metric("Searches", result.searches)
        c.metric("Sources found", len(result.sources))
        return

    st.text(summary.stop_message)
    st.text(summary.outcome_message)
    st.table(summary.rows())
    st.caption(
        "Budget used counts committed reservations, including failed attempts "
        "and reserved final verification calls. These are not provider "
        "billing or token totals."
    )


def render_research_notes(result) -> None:
    """Keep warnings available while avoiding redundant success-screen noise."""

    issues = tuple(getattr(result, "issues", ()))
    warning_count = int(getattr(result, "warning_count", 0) or 0)
    summary = getattr(result, "summary", None)

    show_generic_note = bool(warning_count) and (
        not issues
        or summary is None
        or getattr(summary, "end_reason", None) != "sufficient"
    )

    if not show_generic_note and not issues:
        return

    label_count = max(warning_count, len(issues))
    label = (
        f"Research notes ({label_count})"
        if label_count
        else "Research notes"
    )

    with st.expander(label, expanded=False):
        if show_generic_note:
            st.info(
                "Some research steps could not complete. "
                "The report uses the evidence that was available."
            )

        for issue in issues:
            st.warning(issue)


def render_status_badge(result) -> None:
    """Render a simple user-facing trust/completeness badge."""

    citations = bool(result.report.citations)
    summary = getattr(result, "summary", None)
    end_reason = getattr(summary, "end_reason", None)

    if not citations:
        css_class = "ra-badge-incomplete"
        label = "No verified answer"
        icon = "●"
    elif end_reason == "sufficient":
        css_class = "ra-badge-verified"
        label = "Verified"
        icon = "✓"
    else:
        css_class = "ra-badge-partial"
        label = "Verified partial answer"
        icon = "◐"

    st.markdown(
        (
            f'<div class="ra-badge {css_class}">'
            f"{icon} {escape(label)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_answer_text(content: str) -> None:
    """Render provider text as plain Streamlit text.

    Keeping the answer as a real ``st.text`` element preserves the existing
    UI/test contract while the surrounding layout provides the visual
    hierarchy.
    """

    st.text(content)


def render_developer_tools(*, compact: bool = False) -> None:
    """Render offline-mode controls without entering the mode mid-run."""

    if compact:
        tool_container = st.popover("Developer tools")
    else:
        tool_container = st.expander(
            "Developer tools",
            expanded=False,
        )

    with tool_container:
        st.toggle(
            "Offline demo",
            key="offline_demo",
            help="Explore a fictional sample without providers or API keys.",
        )
        st.toggle(
            "Offline evaluation dashboard",
            key="evaluation_dashboard",
        )

def render_public_demo_only_if_requested() -> None:
    """Enter a safe public portfolio demo before local/live features load."""

    try:
        public_demo_only = Settings().public_demo_only
    except Exception as exc:
        st.error(friendly_error(exc))
        st.stop()

    if not public_demo_only:
        return

    # Public mode must stop before history, providers, or live-research
    # controls are initialized.
    preserve_widget_state()

    st.info(
        "Public portfolio demo · Live Gemini/Tavily research and saved "
        "research history are disabled on this deployment."
    )

    from research_agent.demo import render_demo

    render_demo(public_mode=True)
    st.stop()


def render_active_offline_view_if_requested() -> None:
    """Enter offline views before history/providers or normal widgets run."""

    offline_demo = bool(
        st.session_state.get("offline_demo")
    )
    evaluation_dashboard = bool(
        st.session_state.get("evaluation_dashboard")
    )

    if not offline_demo and not evaluation_dashboard:
        return

    # Keep the active controls rendered so users/tests can leave an offline
    # view. Normal research/history widgets are intentionally not rendered.
    with st.expander("Developer tools", expanded=True):
        offline_demo = st.toggle(
            "Offline demo",
            key="offline_demo",
            help="Explore a fictional sample without providers or API keys.",
        )
        evaluation_dashboard = st.toggle(
            "Offline evaluation dashboard",
            key="evaluation_dashboard",
        )

    # This must happen before the hidden normal widgets are instantiated in
    # the current run. It preserves their values across Streamlit cleanup.
    preserve_widget_state()

    if offline_demo:
        from research_agent.demo import render_demo

        render_demo()
        st.stop()

    if evaluation_dashboard:
        from research_agent.evaluation_dashboard import (
            render_evaluation_dashboard,
        )

        render_evaluation_dashboard()
        st.stop()


def render_sidebar(history) -> bool:
    """Render saved research and history tools."""

    with st.sidebar:
        st.markdown("## Research Agent")
        st.caption(
            "Saved locally. Opening a saved report does not use Gemini."
        )
        st.divider()

        st.markdown("### Saved research")

        search = st.text_input(
            "Search saved research",
            key="history_search",
            placeholder="Search history…",
            help=(
                "Search questions, report text, source titles, and evidence "
                "excerpts. Matches ignore letter case and use the exact text "
                "you enter."
            ),
        )

        def clear_history_filters():
            st.session_state["history_status"] = "All"
            st.session_state["history_depth"] = "All"
            st.session_state["history_outcome"] = "All"
            st.session_state["history_start"] = None
            st.session_state["history_end"] = None

        with st.expander("Filter saved research"):
            status_filter = st.radio(
                "Run status",
                ("All", "Completed", "Incomplete"),
                horizontal=True,
                key="history_status",
            )
            depth_filter = st.radio(
                "Saved research depth",
                ("All", *DEPTH_NAMES, "Not recorded"),
                key="history_depth",
            )
            outcome_filter = st.radio(
                "Research outcome",
                ("All", *OUTCOME_FILTERS),
                key="history_outcome",
            )
            st.caption(
                "Outcomes use saved run details. Older completed reports "
                "without an outcome appear under Not recorded; incomplete "
                "runs appear under Interrupted run."
            )
            st.caption(
                "Dates include both endpoints and use UTC. Older runs without "
                "a saved depth appear under Not recorded."
            )
            start_date = st.date_input(
                "From date (UTC)",
                value=None,
                key="history_start",
            )
            end_date = st.date_input(
                "Through date (UTC)",
                value=None,
                key="history_end",
            )
            st.button(
                "Clear history filters",
                key="clear_history_filters",
                on_click=clear_history_filters,
                use_container_width=True,
            )

        filters_active = (
            status_filter != "All"
            or depth_filter != "All"
            or outcome_filter != "All"
            or start_date
            or end_date
        )
        invalid_dates = (
            start_date
            and end_date
            and start_date > end_date
        )

        if invalid_dates:
            st.warning("From date must be on or before Through date.")

        try:
            saved = (
                []
                if invalid_dates
                else history.list_reports(
                    search,
                    status=(
                        None
                        if status_filter == "All"
                        else status_filter.lower()
                    ),
                    depth=(
                        None
                        if depth_filter == "All"
                        else depth_filter
                    ),
                    start_date=start_date,
                    end_date=end_date,
                    outcome=OUTCOME_FILTERS.get(outcome_filter),
                )
            )
        except HistoryError:
            st.warning(
                "Local history could not be read. "
                "You can still research and download reports."
            )
            saved = []

        if saved:
            labels = {
                item.id: (
                    item.label
                    + (
                        f" · {item.match.location}: {item.match.excerpt}"
                        if item.match
                        else ""
                    )
                )
                for item in saved
            }
            selected = st.selectbox(
                "Saved reports",
                options=list(labels),
                format_func=labels.__getitem__,
                key="history_selection",
            )
            selected_entry = next(
                item
                for item in saved
                if item.id == selected
            )

            if selected_entry.match:
                st.caption(
                    "Match in "
                    + selected_entry.match.location.lower()
                )
                st.text(selected_entry.match.excerpt)

            open_requested = st.button(
                "Open saved research",
                key="open_history",
                use_container_width=True,
            )

            if open_requested and unsaved_entries():
                st.warning(
                    "Your current research is unsaved. Save it below, or "
                    "download it and explicitly discard the session copy "
                    "before opening another entry."
                )
            elif open_requested:
                try:
                    opened = history.load(selected)
                except HistoryError as exc:
                    st.error(str(exc))
                else:
                    if isinstance(opened, IncompleteResearch):
                        st.session_state["incomplete_research"] = opened
                        st.session_state["saved_incomplete_id"] = selected
                    else:
                        st.session_state.pop(
                            "incomplete_research",
                            None,
                        )
                        st.session_state["completed_research"] = opened
                        st.session_state["saved_report_id"] = selected
                        st.session_state["history_save_failed"] = False

            st.button(
                "Research this question again",
                key="research_again",
                on_click=prepare_research_again,
                args=(selected,),
                help=(
                    "Fill in the selected saved question and recorded depth. "
                    "Review them above, then click Start research to run it "
                    "again."
                ),
                use_container_width=True,
            )
        else:
            st.caption(
                "No matching saved reports."
                if search.strip() or filters_active
                else "Your completed reports will appear here."
            )

        st.divider()
        st.markdown("### History tools")
        compare_runs = st.toggle(
            "Compare saved runs",
            key="compare_runs",
        )
        render_backup_controls(history)

    return compare_runs


def render_research_controls(*, compact: bool = False):
    """Render the primary question-first research workflow."""

    depth_name = st.radio(
        "Research depth",
        DEPTH_NAMES,
        index=1,
        horizontal=True,
        key="research_depth",
    )

    st.caption(
        DEPTH_DESCRIPTIONS.get(
            depth_name,
            "Research with the selected depth.",
        )
    )

    depth_ready = True
    depth_limits = None

    try:
        depth_limits = preview_depth(depth_name)
    except Exception as exc:
        depth_ready = False
        st.error(friendly_error(exc))

    if depth_limits is not None:
        if compact:
            depth_container = st.popover("Depth details")
        else:
            depth_container = st.expander(
                "Depth details",
                expanded=False,
            )

        with depth_container:
            st.caption(depth_limits.summary)
            st.caption(
                "Your configured caps may reduce these limits. Every mode "
                "checks citations and reserves final answer verification. "
                "AI calls are application-level limits, not token or provider "
                "quota guarantees."
            )

    with st.form("research", clear_on_submit=False):
        question = st.text_area(
            "What would you like to research?",
            key="question",
            height=145,
            placeholder=(
                "For example: What are the main limitations of "
                "retrieval-augmented generation?"
            ),
        )
        submitted = st.form_submit_button(
            "Start research",
            type="primary",
            disabled=not depth_ready,
            use_container_width=True,
        )

    if compact:
        about_container = st.popover("About this workspace")
    else:
        about_container = st.expander(
            "About this workspace",
            expanded=False,
        )

    with about_container:
        st.caption(
            "Uses your configured Gemini and Tavily services. Reports and "
            "incomplete runs are saved on this computer. Open them from "
            "Saved research in the sidebar."
        )

    return question, submitted, depth_name


def run_submitted_research(
    *,
    question: str,
    submitted: bool,
    depth_name: str,
    history,
) -> None:
    """Execute research only after the existing submission safeguards pass."""

    if not submitted:
        return

    if not question.strip():
        st.warning("Enter a research question to begin.")
        return

    if unsaved_entries():
        st.warning(
            "New research has not started. Save your unsaved research below, "
            "or download it and explicitly discard the session copy first."
        )
        return

    st.session_state.pop("incomplete_research", None)
    st.session_state.pop("saved_incomplete_id", None)

    with st.status(
        "Starting your research",
        expanded=True,
    ) as status:
        progress = st.empty()

        def update(message):
            status.update(label=message)
            progress.write(message)

        try:
            result = run_question(
                question,
                update,
                depth=depth_name,
            )
        except ResearchInterrupted as exc:
            status.update(
                label="Research stopped — incomplete run",
                state="error",
                expanded=True,
            )
            st.session_state["incomplete_research"] = exc.partial

            try:
                st.session_state["saved_incomplete_id"] = history.save(
                    exc.partial
                )
            except HistoryError:
                st.session_state["saved_incomplete_id"] = None
        except Exception as exc:
            status.update(
                label="Research could not finish",
                state="error",
                expanded=True,
            )
            st.error(friendly_error(exc))
        else:
            st.session_state["completed_research"] = result
            st.session_state["saved_report_id"] = None

            try:
                st.session_state["saved_report_id"] = history.save(result)
            except HistoryError:
                st.session_state["history_save_failed"] = True
            else:
                st.session_state["history_save_failed"] = False

            status.update(
                label="Research finished",
                state="complete",
                expanded=False,
            )


def render_unsaved_protection(history) -> None:
    """Render the existing safeguards for unsaved session results."""

    pending = unsaved_entries()
    if not pending:
        return

    st.info(
        "Unsaved research is protected in this browser session. Save or "
        "download it before closing or refreshing the page."
    )

    with st.popover("Discard unsaved research…"):
        st.warning(
            "Discarding removes the unsaved results from this session. "
            "Any downloaded copies and saved history entries are kept. "
            "This cannot be undone."
        )

        if st.button(
            "Discard unsaved session copies",
            key="confirm_discard_unsaved",
        ):
            for value_key, saved_key in pending:
                st.session_state.pop(value_key, None)
                st.session_state.pop(saved_key, None)

            st.session_state["history_save_failed"] = False
            st.rerun()

    if (
        st.session_state.get("incomplete_research") is not None
        and ("completed_research", "saved_report_id") in pending
    ):
        hidden = st.session_state["completed_research"]

        with st.expander(
            "Earlier unsaved report",
            expanded=True,
        ):
            st.text(hidden.report.question)

            left, right = st.columns(2)
            left.download_button(
                "Download earlier report (.txt)",
                hidden.text_export(),
                file_name="earlier-research-report.txt",
                mime="text/plain",
                use_container_width=True,
            )
            right.download_button(
                "Download earlier evidence (.json)",
                hidden.json_export(),
                file_name="earlier-research-evidence.json",
                mime="application/json",
                use_container_width=True,
            )

            if st.button(
                "Save earlier report",
                key="save_hidden_report",
            ):
                try:
                    st.session_state["saved_report_id"] = history.save(
                        hidden
                    )
                except HistoryError:
                    st.error(
                        "Saving failed. The earlier report remains "
                        "available to download."
                    )
                else:
                    st.session_state["history_save_failed"] = False
                    st.rerun()


def render_incomplete_research(partial, history) -> None:
    """Render an incomplete run with details progressively disclosed."""

    st.divider()
    st.subheader("Incomplete research")
    st.markdown(
        '<div class="ra-badge ra-badge-incomplete">● Incomplete</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="ra-question">{escape(partial.question)}</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        partial.stop_message
        + " No verified answer was produced by this run."
    )
    st.caption("Last recorded stage: " + partial.last_stage)
    st.caption(
        "Collected evidence is available below. This entry cannot resume a "
        "run; submitting the question again starts fresh."
    )

    if st.session_state.get("saved_incomplete_id"):
        st.caption("Incomplete run saved on this computer.")
    else:
        st.warning(
            "This incomplete run has not been saved. Download its evidence "
            "or retry saving before leaving this page."
        )

        if st.button(
            "Save incomplete run",
            key="retry_save_incomplete",
        ):
            try:
                st.session_state["saved_incomplete_id"] = history.save(
                    partial
                )
            except HistoryError:
                st.error(
                    "Saving failed. You can still download the collected "
                    "evidence."
                )
            else:
                st.rerun()

    render_research_notes(partial)

    st.subheader("Collected sources and evidence")

    if not partial.sources:
        st.info("The run stopped before any sources were collected.")

    for source_index, source in enumerate(partial.sources, start=1):
        with st.container(border=True):
            st.markdown(
                f"**{source_index}. {source.title or source.domain}**"
            )
            st.caption("Fetch status: " + source.fetch_status)

            url = safe_source_url(source)
            if url:
                st.link_button("Visit page", url)

            excerpts = [
                item
                for item in partial.evidence
                if item.source_id == source.id
            ]

            if excerpts:
                with st.expander(
                    "Collected excerpts · " + source.id
                ):
                    for item in excerpts:
                        st.text(item.excerpt)

    with st.expander("Run details", expanded=False):
        if getattr(partial, "depth", None) is not None:
            st.caption("This run: " + partial.depth.summary)
        render_run_summary(partial)

    st.download_button(
        "Download incomplete research (.json)",
        partial.json_export(),
        file_name="incomplete-research.json",
        mime="application/json",
    )

    st.stop()


def retry_completed_save(result, history) -> None:
    """Preserve the existing retry path when history persistence fails."""

    if st.session_state.get("saved_report_id"):
        return

    if st.session_state.get("history_save_failed"):
        st.warning(
            "This report could not be saved to history. Download it now or "
            "retry saving before closing the page."
        )

    if st.button(
        "Save report to history",
        key="retry_save_history",
    ):
        try:
            st.session_state["saved_report_id"] = history.save(result)
        except HistoryError:
            st.session_state["history_save_failed"] = True
            st.error(
                "Saving failed. Your report is still available to "
                "download below."
            )
        else:
            st.session_state["history_save_failed"] = False
            st.rerun()


def render_citations(result) -> None:
    """Render verified claims first, with source and evidence detail on demand."""

    if not result.report.citations:
        st.info("No verified citations are available for this report.")
        return

    evidence_lookup = {
        item.id: item
        for item in result.evidence
    }

    for index, citation in enumerate(
        result.report.citations,
        start=1,
    ):
        with st.container(border=True):
            st.markdown(f"**Claim {index}**")
            st.write(citation.claim_text)

            sources = result.citation_sources(citation)
            if sources:
                st.caption("Sources")

            for source in sources:
                st.write(source.title or source.domain)
                url = safe_source_url(source)
                if url:
                    st.link_button("Open source", url)
                else:
                    st.caption("Source link unavailable.")

            with st.expander(
                f"Supporting evidence for claim {index}",
                expanded=False,
            ):
                for evidence_id in citation.evidence_ids:
                    item = evidence_lookup.get(evidence_id)
                    if item is not None:
                        st.text(item.excerpt)


def render_all_sources(result) -> None:
    """Keep every discovered source available without overwhelming the report."""

    if not result.sources:
        return

    read_count = sum(
        source.fetch_status == "success"
        for source in result.sources
    )

    st.caption(
        f"Read {read_count} of {len(result.sources)} discovered sources. "
        f"Collected {len(result.evidence)} evidence excerpts."
    )

    labels = {
        "success": "Read successfully",
        "failed": "Could not read",
        "skipped": "Skipped",
        "pending": "Not read",
    }

    with st.expander("All researched sources", expanded=False):
        for source in result.sources:
            st.write(
                f"**{source.title or source.domain}** — "
                f"{labels[source.fetch_status]}"
            )
            url = safe_source_url(source)
            if url:
                st.link_button("Visit page", url)


def render_completed_research(result) -> None:
    """Render the answer first and keep technical detail secondary."""

    st.divider()
    st.markdown(
        '<div class="ra-section-label">Research report</div>',
        unsafe_allow_html=True,
    )
    render_status_badge(result)

    if st.session_state.get("saved_report_id"):
        created = result.report.created_at.astimezone(timezone.utc)
        st.caption(
            "Saved on this computer · "
            + created.strftime("%d %b %Y, %H:%M UTC")
        )

    st.markdown(
        f'<div class="ra-question">{escape(result.report.question)}</div>',
        unsafe_allow_html=True,
    )

    if not result.report.citations:
        st.warning(
            "This run did not produce an answer with verified citations."
        )

    st.subheader("Answer")
    render_answer_text(result.report.content)

    render_research_notes(result)

    st.subheader("Sources & evidence")
    render_citations(result)
    render_all_sources(result)

    with st.expander("Run details", expanded=False):
        if getattr(result, "depth", None) is not None:
            st.caption("This run: " + result.depth.summary)
        render_run_summary(result)

    left, right = st.columns(2)
    left.download_button(
        "Download report (.txt)",
        result.text_export(),
        file_name="research-report.txt",
        mime="text/plain",
        use_container_width=True,
    )
    right.download_button(
        "Download evidence (.json)",
        result.json_export(),
        file_name="research-evidence.json",
        mime="application/json",
        use_container_width=True,
    )


def render_how_it_works() -> None:
    """Preserve the existing workflow explanation without dominating the page."""

    with st.expander("How the research process works", expanded=False):
        a, b, c = st.columns(3)
        a.markdown(
            "**1 · Explore**\n\n"
            "Break down the question and search for relevant sources."
        )
        b.markdown(
            "**2 · Examine**\n\n"
            "Read the evidence and investigate remaining gaps."
        )
        c.markdown(
            "**3 · Verify**\n\n"
            "Check the final answer and trace claims to their sources."
        )


apply_ui_styles()

st.markdown(
    '<div class="ra-eyebrow">Research Agent · Your research workspace</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h1 class="ra-title">Ask a question.<br>Follow the evidence.</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    (
        '<div class="ra-subtitle">'
        "Explore the web, compare evidence, and get a report checked "
        "against its sources."
        "</div>"
    ),
    unsafe_allow_html=True,
)

# Public demo-only mode must short-circuit before local history, providers,
# developer tools, or normal research widgets are touched.
render_public_demo_only_if_requested()

# User-selected offline demo/evaluation must also short-circuit before local
# history, provider configuration, or normal research widgets are touched.
render_active_offline_view_if_requested()

history = get_history_store()
compare_runs = render_sidebar(history)

retry_notice = st.session_state.pop(
    "research_again_notice",
    None,
)

viewing_research = (
    st.session_state.get("completed_research") is not None
    or st.session_state.get("incomplete_research") is not None
)

if viewing_research:
    with st.expander(
        "Start new research",
        expanded=bool(retry_notice),
    ):
        render_developer_tools(compact=True)
        question, submitted, depth_name = render_research_controls(
            compact=True
        )
else:
    render_developer_tools()
    question, submitted, depth_name = render_research_controls()

if retry_notice:
    level, message = retry_notice
    (
        st.warning
        if level == "warning"
        else st.info
    )(message)

run_submitted_research(
    question=question,
    submitted=submitted,
    depth_name=depth_name,
    history=history,
)

if compare_runs:
    st.divider()
    st.subheader("Compare saved research")
    from research_agent.comparison_ui import render_comparison

    render_comparison(history)

render_unsaved_protection(history)

partial = st.session_state.get("incomplete_research")
if partial is not None:
    render_incomplete_research(
        partial,
        history,
    )

result = st.session_state.get("completed_research")

if result is not None:
    retry_completed_save(
        result,
        history,
    )
    render_completed_research(result)
else:
    render_how_it_works()