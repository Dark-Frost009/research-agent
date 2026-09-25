"""Sidebar backup controls; no provider calls or active-result replacement."""
import streamlit as st

from research_agent.backup import export_backup, read_backup, restore_backup, MAX_BACKUP_BYTES
from research_agent.history import HistoryError, HistoryStore


def render_backup_controls(history: HistoryStore):
    with st.expander("History backup & restore"):
        st.caption("Backups contain saved reports and source excerpts. Keep the file private. Unsaved session results are not included.")
        notice = st.session_state.pop("history_restore_notice", None)
        if notice:
            st.success(notice)
        if st.button("Prepare history backup", key="prepare_history_backup"):
            st.session_state.pop("history_backup_bytes", None)
            try:
                st.session_state["history_backup_bytes"] = export_backup(history)
            except HistoryError as exc:
                st.error(str(exc))
        raw = st.session_state.get("history_backup_bytes")
        if raw is not None:
            timestamp = read_backup(raw).created_at.strftime("%Y-%m-%d %H:%M UTC")
            st.caption("Snapshot prepared " + timestamp + ". Prepare again to include newer entries.")
            st.download_button("Download history backup (.json)", raw,
                               file_name="research-history-backup.json", mime="application/json",
                               key="download_history_backup")
        upload = st.file_uploader("Choose a history backup", type=["json"], key="history_backup_upload")
        if upload is not None:
            try:
                if upload.size > MAX_BACKUP_BYTES:
                    raise HistoryError("This backup exceeds the 20 MB limit.")
                data = upload.getvalue()
                backup = read_backup(data)
            except HistoryError as exc:
                st.error(str(exc))
            else:
                completed = sum(record.kind == "completed" for record in backup.records)
                st.caption(f"Backup contains {completed} completed reports and {len(backup.records) - completed} incomplete runs.")
                st.caption("Restore adds missing entries and skips identical copies. Conflicts cancel the whole restore. Existing history and unsaved work are kept. Restoring validates the file structure, not the truth of its research claims.")
                if st.button("Restore missing history", key="restore_history_backup"):
                    try:
                        added, skipped = restore_backup(history, data)
                    except HistoryError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["history_restore_notice"] = f"Restored {added} entries; {skipped} identical entries were already present."
                        st.session_state.pop("history_backup_bytes", None)
                        st.rerun()
