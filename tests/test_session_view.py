from __future__ import annotations

from milknado.app.session_view import (
    action_options,
    error_text,
    permission_options,
    short_title,
    transcript_text,
)
from milknado.domains.common import SessionEvent, SessionView


def test_short_title_is_deterministic_without_truncating_the_brief() -> None:
    description = "# Build the release artifact\n\nKeep the complete acceptance brief here."

    assert short_title(description) == "Build the release artifact"
    title = short_title("a " * 60, limit=12)
    assert title.startswith("a a")
    assert title.endswith("…")
    assert len(title) <= 12


def test_transcript_and_errors_are_separate_rich_renderables() -> None:
    view = SessionView(
        events=(
            SessionEvent(kind="assistant", text="Starting", event_id="a", state="streaming"),
            SessionEvent(kind="tool", text="pytest passed", event_id="t"),
            SessionEvent(kind="error", text="permission denied", event_id="e"),
        )
    )

    transcript = transcript_text(view)
    errors = error_text(view)
    assert "Assistant [streaming]: Starting" in transcript.plain
    assert "Tool: pytest passed" in transcript.plain
    assert "permission denied" not in transcript.plain
    assert errors.plain == "Error: permission denied"


def test_permission_choices_keep_exact_request_ids() -> None:
    view = SessionView(
        actions=("approve", "deny"),
        permissions=(
            SessionEvent(
                kind="permission", text="read source", event_id="request-1", state="requested"
            ),
            SessionEvent(
                kind="permission", text="old request", event_id="request-2", state="approved"
            ),
        ),
    )

    assert action_options(view.actions) == (
        ("Approve permission request", "approve"),
        ("Deny permission request", "deny"),
    )
    assert permission_options(view) == (("request-1: read source", "request-1"),)
