"""Why a synthesized answer was rejected, as a fixed code.

When synthesis or semantic verification rejects an answer, the answer and all
verifier text are discarded on purpose: they are untrusted model output. What
is kept is one of the codes below, chosen by Python at the check that failed.
A code never contains model text, so it is safe to store, export and show.

The codes, from the earliest check to the last:

* ``quote_mismatch``: a supporting quote was not found word for word in the
  evidence excerpt it cited (deterministic check).
* ``synthesis_format``: the synthesis broke another structural rule, such as a
  claim missing from the answer text or an unknown evidence handle
  (deterministic check).
* ``verifier_format``: the verifier's own reply was malformed or inconsistent,
  so it could not be trusted either way.
* ``unsupported_claim``: the verifier judged at least one claim not supported
  by its cited evidence.
* ``incomplete_coverage``: the verifier found factual statements in the answer
  that no citation covers.

If several problems exist, the first check that fails decides the code.
"""

from __future__ import annotations

from typing import Literal, get_args

RejectionReason = Literal[
    "quote_mismatch",
    "synthesis_format",
    "verifier_format",
    "unsupported_claim",
    "incomplete_coverage",
]

REJECTION_REASONS: frozenset[str] = frozenset(get_args(RejectionReason))


def require_rejection_reason(reason: object) -> RejectionReason:
    """Return ``reason`` if it is a known code, otherwise fail loudly.

    An unknown code here is a programming error, not a model error.
    """

    if reason not in REJECTION_REASONS:
        raise ValueError(f"Unknown rejection reason: {reason!r}")

    return reason  # type: ignore[return-value]
