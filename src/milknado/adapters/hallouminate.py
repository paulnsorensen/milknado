"""Process adapter for refreshing the hallouminate wiki index."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from milknado.domains.wiki import WikiIndexResult, WikiIndexStatus

_MAX_DETAIL = 2_000


class HallouminateIndexer:
    def refresh(self, wiki_root: Path) -> WikiIndexResult:
        binary = shutil.which("hallouminate")
        if binary is None:
            return WikiIndexResult(
                WikiIndexStatus.UNAVAILABLE,
                "hallouminate executable was not found on PATH",
            )
        try:
            completed = subprocess.run(
                [binary, "index"],
                cwd=wiki_root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return WikiIndexResult(WikiIndexStatus.FAILED, _bounded(str(exc)))
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            if not detail:
                detail = f"hallouminate index exited with status {completed.returncode}"
            return WikiIndexResult(WikiIndexStatus.FAILED, _bounded(detail))
        return WikiIndexResult(WikiIndexStatus.REFRESHED)


def _bounded(detail: str) -> str:
    if len(detail) <= _MAX_DETAIL:
        return detail
    return detail[: _MAX_DETAIL - 1] + "…"
