from __future__ import annotations

from pathlib import Path

from milknado.loop.sessions._protocol import SessionProtocol

_SUPPORTED_FAMILIES = {"omp", "claude", "codex"}


def _family(argv: tuple[str, ...]) -> str | None:
    if not argv:
        return None
    stem = Path(argv[0]).name.lower()
    if stem.endswith(".exe"):
        stem = stem[:-4]
    return stem if stem in _SUPPORTED_FAMILIES else None


def create_protocol(argv: tuple[str, ...], cwd: Path) -> SessionProtocol | None:
    """Construct a supported family protocol without spawning a process."""
    family = _family(argv)
    if family == "omp":
        from milknado.loop.sessions._omp import OmpSession

        return OmpSession(argv, cwd)
    if family == "claude":
        from milknado.loop.sessions._claude import ClaudeSession

        return ClaudeSession(argv, cwd)
    if family == "codex":
        from milknado.loop.sessions._codex import CodexSession

        return CodexSession(argv, cwd)
    return None


def is_supported(argv: tuple[str, ...]) -> bool:
    """Return whether *argv* names one of the structured session families."""
    return _family(argv) is not None
