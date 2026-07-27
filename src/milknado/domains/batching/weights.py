from __future__ import annotations

import functools
import hashlib
import math
import os
from pathlib import Path

import tiktoken

from milknado.domains.batching.change import FileChange

TOKENS_PER_LINE: dict[str, int] = {
    "py": 10,
    "ts": 8,
    "tsx": 8,
    "js": 7,
    "jsx": 7,
    "rs": 11,
    "go": 9,
    "java": 11,
    "kt": 10,
    "rb": 9,
    "php": 9,
    "c": 9,
    "cpp": 10,
    "h": 9,
    "md": 5,
    "toml": 6,
    "yaml": 6,
    "yml": 6,
    "json": 6,
}
NEW_FILE_LINES: dict[str, int] = {
    "py": 150,
    "ts": 120,
    "tsx": 120,
    "js": 120,
    "jsx": 120,
    "rs": 200,
    "go": 180,
    "java": 200,
    "kt": 180,
    "rb": 120,
    "php": 150,
    "c": 180,
    "cpp": 200,
    "h": 100,
    "md": 80,
    "toml": 40,
    "yaml": 50,
    "yml": 50,
    "json": 40,
}
FLAT_COST: dict[str, int] = {"delete": 80, "rename": 120}
HEADROOM: float = 1.25
TIKTOKEN_BLOB_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
TIKTOKEN_CACHE_KEY = hashlib.sha1(TIKTOKEN_BLOB_URL.encode(), usedforsecurity=False).hexdigest()


def _configure_tiktoken_cache() -> None:
    cache_dir = Path(__file__).resolve().parents[2] / "_vendor" / "tiktoken-cache"
    if (cache_dir / TIKTOKEN_CACHE_KEY).is_file():
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache_dir))


@functools.lru_cache(maxsize=1)
def _get_encoder() -> tiktoken.Encoding:
    _configure_tiktoken_cache()
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


RALPH_STARTUP_TOKENS = 2000


def batch_size_cost(k: int) -> int:
    """Fixed ralph-loop startup overhead for a batch of k changes.

    Models system-prompt + tool-setup tokens that don't scale with file content.
    k=0 means no ralph invocation, so no overhead.
    """
    return RALPH_STARTUP_TOKENS if k > 0 else 0


def _estimate_path_level(change: FileChange, root: Path) -> int:
    ext = _extension(change.path)
    resolved_root = root.resolve()
    resolved_path = (root / change.path).resolve()
    if resolved_path.is_relative_to(resolved_root):
        real = _tiktoken_count(resolved_path)
        if real is not None:
            return math.ceil(real * HEADROOM)
    lines = NEW_FILE_LINES.get(ext, 150)
    tpl = TOKENS_PER_LINE.get(ext, 8)
    return math.ceil(lines * tpl * HEADROOM)


def estimate_tokens(change: FileChange, root: Path) -> int:
    if change.edit_kind in FLAT_COST:
        return FLAT_COST[change.edit_kind]
    if change.edit_kind == "modify":
        return _estimate_path_level(change, root)
    ext = _extension(change.path)
    lines = NEW_FILE_LINES.get(ext, 150)
    tpl = TOKENS_PER_LINE.get(ext, 8)
    return math.ceil(lines * tpl * HEADROOM)
