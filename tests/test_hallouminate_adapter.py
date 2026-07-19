from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from milknado.adapters import hallouminate
from milknado.adapters.hallouminate import HallouminateIndexer
from milknado.domains.wiki.ports import WikiIndexResult, WikiIndexStatus


def test_refresh_distinguishes_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(hallouminate.shutil, "which", lambda _name: None)

    result = HallouminateIndexer().refresh(tmp_path)

    assert result == WikiIndexResult(
        WikiIndexStatus.UNAVAILABLE,
        "hallouminate executable was not found on PATH",
    )


def test_refresh_preserves_actionable_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(hallouminate.shutil, "which", lambda _name: "/opt/bin/hallouminate")
    calls: list[tuple[list[str], Path]] = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs["cwd"]))
        return subprocess.CompletedProcess(
            argv,
            returncode=2,
            stdout="",
            stderr="index database locked: /repo/.hallouminate/index.lance\n",
        )

    monkeypatch.setattr(hallouminate.subprocess, "run", run)

    result = HallouminateIndexer().refresh(tmp_path)

    assert result == WikiIndexResult(
        WikiIndexStatus.FAILED,
        "index database locked: /repo/.hallouminate/index.lance",
    )
    assert calls == [(["/opt/bin/hallouminate", "index"], tmp_path)]


def test_refresh_reports_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hallouminate.shutil, "which", lambda _name: "/opt/bin/hallouminate")
    monkeypatch.setattr(
        hallouminate.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv, returncode=0, stdout="indexed", stderr=""
        ),
    )

    assert HallouminateIndexer().refresh(tmp_path) == WikiIndexResult(WikiIndexStatus.REFRESHED)
