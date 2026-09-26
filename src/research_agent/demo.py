"""A fictional, bundled demonstration. No providers, configuration, or history."""

import json


NOTICE = (
    "OFFLINE DEMO — fictional sample data. "
    "No web search or AI verification was performed."
)


def demo_data() -> dict:
    """Return fresh sample data, separate from real research result schemas."""

    return {
        "demo": True,
        "notice": NOTICE,
        "question": (
            "At the fictional Meadow Library, how many books may a member "
            "borrow, and when can a loan be renewed?"
        ),
        "answer": (
            "A member may borrow up to four books at a time, with a loan "
            "period of 21 days. [1]\n\n"
            "A loan may be renewed once for 14 more days, provided no other "
            "member has placed a hold on the book. [2]"
        ),
        "sources": [
            {
                "id": "demo-source-1",
                "title": "Meadow Library — sample borrowing policy",
                "text": (
                    "Members may borrow up to four books at a time. "
                    "Each book has a loan period of 21 days."
                ),
            },
            {
                "id": "demo-source-2",
                "title": "Meadow Library — sample renewal policy",
                "text": (
                    "A loan may be renewed once for 14 additional days, "
                    "provided no other member has placed a hold on the book."
                ),
            },
        ],
        "evidence": [
            {
                "id": "demo-evidence-1",
                "source_id": "demo-source-1",
                "excerpt": (
                    "Members may borrow up to four books at a time. "
                    "Each book has a loan period of 21 days."
                ),
            },
            {
                "id": "demo-evidence-2",
                "source_id": "demo-source-2",
                "excerpt": (
                    "A loan may be renewed once for 14 additional days, "
                    "provided no other member has placed a hold on the book."
                ),
            },
        ],
        "citations": [
            {
                "id": "1",
                "claim": "Members may borrow four books for 21 days.",
                "evidence_ids": ["demo-evidence-1"],
            },
            {
                "id": "2",
                "claim": (
                    "One 14-day renewal is allowed when no other member "
                    "has placed a hold."
                ),
                "evidence_ids": ["demo-evidence-2"],
            },
        ],
        "actual_provider_calls": 0,
        "example_summary": {
            "simulated": True,
            "stop_reason": (
                "Example: the evidence was judged sufficient after one "
                "research round."
            ),
            "outcome": (
                "Prepared sample answer; the application did not run "
                "verification."
            ),
            "budget": [
                {
                    "Resource": "Research rounds",
                    "Example budget used": 1,
                    "Example limit": 1,
                },
                {
                    "Resource": "Searches",
                    "Example budget used": 2,
                    "Example limit": 2,
                },
                {
                    "Resource": "Sources",
                    "Example budget used": 2,
                    "Example limit": 4,
                },
                {
                    "Resource": "Page reads",
                    "Example budget used": 2,
                    "Example limit": 4,
                },
                {
                    "Resource": "AI calls",
                    "Example budget used": 6,
                    "Example limit": 12,
                },
            ],
        },
    }


def demo_text(data: dict) -> str:
    lines = [
        NOTICE,
        "",
        data["question"],
        "",
        data["answer"],
        "",
        "FICTIONAL SAMPLE SOURCES",
    ]

    for source in data["sources"]:
        lines += [
            source["title"],
            source["text"],
            "",
        ]

    lines += [
        "ILLUSTRATIVE RUN SUMMARY",
        data["example_summary"]["stop_reason"],
        data["example_summary"]["outcome"],
        "Actual provider calls: 0",
    ]

    lines += [
        (
            f"{row['Resource']}: "
            f"{row['Example budget used']} / "
            f"{row['Example limit']} (simulated)"
        )
        for row in data["example_summary"]["budget"]
    ]

    return "\n".join(lines)


def render_demo(*, public_mode: bool = False) -> None:
    import streamlit as st

    data = demo_data()

    st.warning(NOTICE)

    if public_mode:
        st.caption(
            "This fixed portfolio example works without API keys or internet "
            "access. Live research and saved research history are disabled "
            "on this public deployment."
        )
    else:
        st.caption(
            "This fixed example works without API keys or internet access. "
            "It is never saved to research history. Turn off Offline demo "
            "to return to your work."
        )

    st.subheader("Sample research question")
    st.text(data["question"])

    st.caption(
        "Example workflow: explore the question → collect excerpts → "
        "check claims → present the report."
    )

    report_tab, sources_tab, summary_tab = st.tabs(
        [
            "Sample report",
            "Sample sources & evidence",
            "Sample run summary",
        ]
    )

    with report_tab:
        st.text(data["answer"])
        st.caption(
            "Citation numbers link the prepared sample claims to the excerpts "
            "in the sources tab. These are not live-verified findings."
        )

    with sources_tab:
        sources = {
            item["id"]: item
            for item in data["sources"]
        }
        evidence = {
            item["id"]: item
            for item in data["evidence"]
        }

        for citation in data["citations"]:
            st.markdown(
                f"**Sample claim {citation['id']}**"
            )
            st.text(citation["claim"])

            for evidence_id in citation["evidence_ids"]:
                item = evidence[evidence_id]
                st.text(
                    sources[item["source_id"]]["title"]
                )

                with st.expander(
                    f"Sample supporting excerpt {citation['id']}",
                    expanded=True,
                ):
                    st.text(item["excerpt"])

        st.caption(
            "Both source documents are fictional and bundled with the app; "
            "no pages were fetched."
        )

    with summary_tab:
        st.metric(
            "Actual provider calls",
            0,
        )
        st.text(
            data["example_summary"]["stop_reason"]
        )
        st.text(
            data["example_summary"]["outcome"]
        )
        st.table(
            data["example_summary"]["budget"]
        )
        st.caption(
            "The table illustrates budget accounting for a hypothetical run. "
            "It is not usage from this demo."
        )

    left, right = st.columns(2)

    left.download_button(
        "Download demo report (.txt)",
        demo_text(data),
        file_name="DEMO-research-report.txt",
        mime="text/plain",
    )

    right.download_button(
        "Download demo evidence (.json)",
        json.dumps(
            data,
            indent=2,
        ),
        file_name="DEMO-research-evidence.json",
        mime="application/json",
    )