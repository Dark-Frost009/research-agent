"""Read-only comparisons of saved runs; never replace active session results."""
from datetime import timezone

import streamlit as st

from research_agent.history import HistoryError, HistoryStore
from research_agent.incomplete import IncompleteResearch
from research_agent.ui_service import safe_source_url
from research_agent.comparison_export import comparison_markdown, source_urls as _source_urls


def _render_run(result, label, other_urls):
    incomplete = isinstance(result, IncompleteResearch)
    st.subheader(label)
    st.text(result.question if incomplete else result.report.question)
    created_at = result.created_at if incomplete else result.report.created_at
    st.caption(created_at.astimezone(timezone.utc).strftime("%d %b %Y, %H:%M UTC"))
    st.caption("Status: " + ("Incomplete" if incomplete else "Completed"))
    depth = getattr(result, "depth", None)
    st.caption("Research depth: " + (depth.name if depth else "Not recorded"))
    if depth:
        st.text(depth.summary)
    summary = getattr(result, "summary", None)
    st.markdown("**Outcome**")
    if incomplete:
        st.text("Interrupted run. No verified answer was released.")
        st.text(result.stop_message)
    elif summary:
        st.text(summary.outcome_message)
    else:
        st.text("Outcome not recorded for this older report.")
    st.markdown("**Answer**")
    if incomplete:
        st.text("No completed answer is available for this run.")
    else:
        st.text(result.report.content)
    st.markdown("**Budget usage**")
    if summary:
        st.text(summary.stop_message)
        st.table(summary.rows())
    else:
        st.text("Detailed budget usage and limits were not recorded.")
        st.text(f"Recorded rounds: {result.iterations}; searches: {result.searches}.")
    st.caption(f"Sources: {len(result.sources)} · Evidence excerpts: {len(result.evidence)} · Warnings: {result.warning_count}")
    for issue in result.issues:
        st.text(issue)
    st.markdown("**Sources**")
    if not result.sources:
        st.text("No sources were collected.")
    for source in result.sources:
        st.text(source.title or source.domain)
        url = safe_source_url(source)
        if url:
            st.caption("Shared source URL" if url in other_urls else "Only in this run")
            st.text(url)
        else:
            st.caption("Source URL unavailable for comparison")
        st.caption("Fetch status: " + source.fetch_status)


def render_comparison(history: HistoryStore):
    st.subheader("Compare saved runs")
    st.caption("Choose two entries from all saved history. Sidebar filters do not restrict this list. Comparing does not start research or replace your current result.")
    try:
        entries = history.list_reports()
    except HistoryError as exc:
        st.warning(str(exc))
        return
    if len(entries) < 2:
        st.info("Save at least two runs to compare them.")
        return
    labels = {entry.id: entry.label for entry in entries}
    first = st.selectbox("First run", list(labels), index=None,
                         format_func=labels.__getitem__, key="compare_first")
    second = st.selectbox("Second run", [key for key in labels if key != first], index=None,
                          format_func=labels.__getitem__, key="compare_second")
    if not first or not second:
        st.caption("Select both runs to see their comparison.")
        return
    try:
        left, right = history.load(first), history.load(second)
    except HistoryError as exc:
        st.warning(str(exc))
        return
    left_question = left.question if isinstance(left, IncompleteResearch) else left.report.question
    right_question = right.question if isinstance(right, IncompleteResearch) else right.report.question
    if left_question.strip() != right_question.strip():
        st.info("These runs used different questions. Keep that difference in mind when comparing their answers.")
    left_urls, right_urls = _source_urls(left), _source_urls(right)
    st.caption(f"Source URLs: {len(left_urls & right_urls)} shared · {len(left_urls - right_urls)} only in first · {len(right_urls - left_urls)} only in second.")
    st.caption("Source overlap uses exact available HTTP(S) URLs, not page-content equality. Budget usage counts saved reservations, not provider billing or token totals. This view does not score answer quality.")
    st.download_button("Download comparison (.md)", comparison_markdown(first, left, second, right),
                       file_name="research-comparison.md", mime="text/markdown",
                       key="download_comparison")
    for column, result, label, other_urls in zip(st.columns(2), (left, right),
                                                ("First run", "Second run"), (right_urls, left_urls)):
        with column:
            _render_run(result, label, other_urls)
