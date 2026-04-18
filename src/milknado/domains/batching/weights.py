from __future__ import annotations

import functools
from pathlib import Path

import tiktoken

from milknado.domains.batching.change import FileChange

TOKENS_PER_LINE: dict[str, int] = {
    "py": 10, "ts": 8, "tsx": 8, "js": 7, "jsx": 7,
    "rs": 11, "go": 9, "java": 11, "kt": 10,
    "rb": 9, "php": 9, "c": 9, "cpp": 10, "h": 9,
    "md": 5, "toml": 6, "yaml": 6, "yml": 6, "json": 6,
}
NEW_FILE_LINES: dict[str, int] = {
    "py": 150, "ts": 120, "tsx": 120, "js": 120, "jsx": 120,
    "rs": 200, "go": 180, "java": 200, "kt": 180,
    "rb": 120, "php": 150, "c": 180, "cpp": 200, "h": 100,
    "md": 80, "toml": 40, "yaml": 50, "yml": 50, "json": 40,
}
FLAT_COST: dict[str, int] = {"delete": 80, "rename": 120}
HEADROOM: float = 1.25


@functools.lru_cache(maxsize=1)
def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def _extension(path: str) -> str:
    base = path.rsplit("/", 1)[-1]
    if "." not in base or base.startswith("."):
        return ""
    return base.rsplit(".", 1)[-1].lower()


def _tiktoken_count(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return len(_get_encoder().encode(text))


def estimate_tokens(change: FileChange, root: Path) -> int:
    if change.edit_kind in FLAT_COST:
        return FLAT_COST[change.edit_kind]
    ext = _extension(change.path)
    tpl = TOKENS_PER_LINE.get(ext, 8)
    if change.edit_kind == "modify":
        real = _tiktoken_count(root / change.path)
        if real is not None:
            return int(real * HEADROOM)
    lines = NEW_FILE_LINES.get(ext, 150)
    return int(lines * tpl * HEADROOM)
