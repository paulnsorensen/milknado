from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from milknado.adapters.hallouminate import HallouminateIndexer
from milknado.domains.wiki.ports import WikiIndexResult, WikiIndexStatus


def _which_unavailable(_name: str) -> str | None:
    return None


def _which_hallouminate(_name: str) -> str | None:
    return "/opt/bin/hallouminate"


def _run_success(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, returncode=0, stdout="indexed", stderr="")


def test_refresh_distinguishes_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(shutil, "which", _which_unavailable)

    result = HallouminateIndexer().refresh(tmp_path)

    assert result == WikiIndexResult(
        WikiIndexStatus.UNAVAILABLE,
        "hallouminate executable was not found on PATH",
    )


def test_refresh_preserves_actionable_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(shutil, "which", _which_hallouminate)
    calls: list[tuple[list[str], object]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs["cwd"]))
        return subprocess.CompletedProcess(
            argv,
            returncode=2,
            stdout="",
            stderr="index database locked: /repo/.hallouminate/index.lance\n",
        )

    monkeypatch.setattr(subprocess, "run", run)

    result = HallouminateIndexer().refresh(tmp_path)

    assert result == WikiIndexResult(
        WikiIndexStatus.FAILED,
        "index database locked: /repo/.hallouminate/index.lance",
    )
    assert calls == [(["/opt/bin/hallouminate", "index"], tmp_path)]


def test_refresh_reports_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shutil, "which", _which_hallouminate)
    monkeypatch.setattr(subprocess, "run", _run_success)

    assert HallouminateIndexer().refresh(tmp_path) == WikiIndexResult(WikiIndexStatus.REFRESHED)
