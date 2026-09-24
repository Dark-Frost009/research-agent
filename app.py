"""Local Streamlit entry point. Start with: python -m streamlit run app.py."""
from pathlib import Path
from datetime import timezone
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st
from research_agent.ui_service import run_question, friendly_error, safe_source_url
from research_agent.history import get_history_store, HistoryError

history = get_history_store()

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="centered")
st.caption("RESEARCH AGENT  /  YOUR RESEARCH WORKSPACE")
st.title("Ask a question. Follow the evidence.")
st.write("Explore the web, compare evidence, and get a report checked against its sources.")

with st.form("research", clear_on_submit=False):
    question = st.text_area("What would you like to research?", key="question", height=120,
                            placeholder="For example: What are the main limitations of retrieval-augmented generation?")
    submitted = st.form_submit_button("Start research", type="primary")

st.caption("Uses your configured Gemini and Tavily services. Completed reports are saved on this computer. Open them from Saved research in the sidebar.")

if submitted:
    if not question.strip():
        st.warning("Enter a research question to begin.")
    else:
        with st.status("Starting your research", expanded=True) as status:
            progress = st.empty()
            def update(message):
                status.update(label=message)
                progress.write(message)
            try:
                result = run_question(question, update)
            except Exception as exc:
                status.update(label="Research could not finish", state="error", expanded=True)
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
                status.update(label="Research finished", state="complete", expanded=False)

with st.sidebar:
    st.header("Saved research")
    st.caption("Stored on this computer. Opening a report does not use Gemini.")
    search = st.text_input("Find a saved question", key="history_search")
    try:
        saved = history.list_reports(search)
    except HistoryError:
        st.warning("Local history could not be read. You can still research and download reports.")
        saved = []
    if saved:
        labels = {item.id: item.label for item in saved}
        selected = st.selectbox("Saved reports", options=list(labels),
                                format_func=labels.__getitem__, key="history_selection")
        if st.button("Open report", key="open_history"):
            try:
                opened = history.load(selected)
            except HistoryError as exc:
                st.error(str(exc))
            else:
                st.session_state["completed_research"] = opened
                st.session_state["saved_report_id"] = selected
                st.session_state["history_save_failed"] = False
    else:
        st.caption("No matching saved reports." if search.strip() else "Your completed reports will appear here.")

result = st.session_state.get("completed_research")
if result is not None and not st.session_state.get("saved_report_id"):
    if st.session_state.get("history_save_failed"):
        st.warning("This report could not be saved to history. Download it now or retry saving before closing the page.")
    # Also lets an existing pre-history browser session preserve its report.
    if st.button("Save report to history", key="retry_save_history"):
        try:
            st.session_state["saved_report_id"] = history.save(result)
        except HistoryError:
            st.session_state["history_save_failed"] = True
            st.error("Saving failed. Your report is still available to download below.")
        else:
            st.session_state["history_save_failed"] = False
            st.rerun()

if result is not None:
    st.divider()
    st.subheader("Research report")
    if st.session_state.get("saved_report_id"):
        st.caption("Saved on this computer · " + result.report.created_at.astimezone(timezone.utc).strftime("%d %b %Y, %H:%M UTC"))
    st.text(result.report.question)
    a, b, c = st.columns(3)
    a.metric("Research rounds", result.iterations)
    b.metric("Searches", result.searches)
    c.metric("Sources found", len(result.sources))
    if not result.report.citations:
        st.warning("This run did not produce an answer with verified citations.")
    if result.warning_count:
        st.info("Some research steps could not complete. The report uses the evidence that was available.")
    for issue in getattr(result, "issues", ()):
        st.warning(issue)
    report_tab, sources_tab = st.tabs(["Report", "Sources & evidence"])
    with report_tab:
        # Provider text is plain text: it cannot embed remote images or HTML.
        st.text(result.report.content)
    with sources_tab:
        if not result.report.citations:
            st.info("No verified citations are available for this report.")
        if result.sources:
            read_count = sum(source.fetch_status == "success" for source in result.sources)
            st.caption(f"Read {read_count} of {len(result.sources)} discovered sources. Collected {len(result.evidence)} evidence excerpts.")
            with st.expander("All discovered sources"):
                labels = {"success": "Read successfully", "failed": "Could not read", "skipped": "Skipped", "pending": "Not read"}
                for source in result.sources:
                    st.text(f"{source.title or source.domain} — {labels[source.fetch_status]}")
                    url = safe_source_url(source)
                    if url:
                        st.link_button("Visit page", url)
        evidence = {e.id: e for e in result.evidence}
        for index, citation in enumerate(result.report.citations, 1):
            st.markdown(f"**Claim {index}**")
            st.text(citation.claim_text)
            for source in result.citation_sources(citation):
                st.text(source.title or source.domain)
                url = safe_source_url(source)
                if url:
                    st.link_button("Open source", url)
                else:
                    st.caption("Source link unavailable.")
            with st.expander(f"Supporting evidence for claim {index}"):
                for evidence_id in citation.evidence_ids:
                    st.text(evidence[evidence_id].excerpt)
    left, right = st.columns(2)
    left.download_button("Download report (.txt)", result.text_export(),
                         file_name="research-report.txt", mime="text/plain")
    right.download_button("Download evidence (.json)", result.json_export(),
                          file_name="research-evidence.json", mime="application/json")
else:
    st.divider()
    st.subheader("From question to supported answer")
    a, b, c = st.columns(3)
    a.markdown("**1 · Explore**\n\nBreak down the question and search for relevant sources.")
    b.markdown("**2 · Examine**\n\nRead the evidence and investigate remaining gaps.")
    c.markdown("**3 · Verify**\n\nCheck the final answer and trace claims to their sources.")
