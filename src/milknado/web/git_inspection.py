"""Git inspection of a run's session worktree for the web host."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.adapters import ChangedFile
    from milknado.domains.common import GitPort, SessionContext


class ProjectGitInspection:
    def __init__(self, project_root: Path) -> None:
        from milknado.adapters import GitAdapter

        self._git: GitAdapter = GitAdapter(project_root)

    @property
    def port(self) -> GitPort:
        return self._git

    def changes(self, context: SessionContext) -> tuple[ChangedFile, ...]:
        return self._git.session_changes(context)

    def diff(self, context: SessionContext, path: str) -> str:
        return self._git.session_diff(context, path)
