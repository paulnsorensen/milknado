"""#17 residual: mcp_server.main() wires configure_stderr_logging() before
mcp.run() so the long-lived stdio server logs to stderr, not stdout."""

from __future__ import annotations

import pytest


def test_main_calls_configure_stderr_logging_before_mcp_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import milknado.domains.execution.run_loop._logging as logging_mod
    from milknado import mcp_server

    calls: list[str] = []

    def fake_configure_stderr_logging():
        calls.append("configure_stderr_logging")
        import logging

        return logging.NullHandler()

    def fake_run() -> None:
        calls.append("mcp.run")

    monkeypatch.setattr(logging_mod, "configure_stderr_logging", fake_configure_stderr_logging)
    monkeypatch.setattr(mcp_server.mcp, "run", fake_run)

    mcp_server.main()

    assert calls == ["configure_stderr_logging", "mcp.run"], (
        "configure_stderr_logging must run exactly once, before mcp.run()"
    )
