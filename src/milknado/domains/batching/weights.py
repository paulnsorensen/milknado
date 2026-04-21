from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import TYPE_CHECKING

import tiktoken

from milknado.domains.batching.change import FileChange

if TYPE_CHECKING:
    from milknado.domains.common.protocols import TilthPort

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


# ---------------------------------------------------------------------------
# Token estimation — design notes
# ---------------------------------------------------------------------------
#
# PRIMARY: per-symbol tiktoken slicing via TilthPort
#
# `estimate_tokens_per_symbols` is the main entry point. For edit_kind=modify
# with symbols present and a live TilthPort, it resolves each SymbolRef to a
# line range by calling tilth_port.search_symbol(name, glob=file), fetches the
# source slice with tilth_port.read_section, tiktoken-encodes the concatenated
# slices, and multiplies by HEADROOM.  This gives a token count proportional
# to the actual symbols touched rather than the whole file.
#
# FALLBACK: path-level tiktoken on the whole file
#
# When tilth_port is None, when change.symbols is empty, or when any symbol
# resolution step raises, the function falls back to reading the whole file
# with tiktoken (same behaviour as the old estimate_tokens).  Every degradation
# is appended to .milknado/planning-context-warn.log so callers can detect when
# per-symbol costing is unavailable.
#
# NON-MODIFY HEURISTIC
#
# edit_kind != modify (add, and the flat-cost kinds delete/rename) still uses
# the NEW_FILE_LINES × TOKENS_PER_LINE × HEADROOM heuristic.  No file read
# or symbol resolution is attempted because the target file does not yet exist
# (add) or the cost is inherently flat (delete/rename).
#
# CALIBRATION TRIGGER
#
# If calibration.jsonl shows batches consistently coming in under 60% of their
# token budget, bump HEADROOM upward.  Over-fragmented batches (many single-
# change batches) suggest HEADROOM is too aggressive.
# ---------------------------------------------------------------------------


RALPH_STARTUP_TOKENS = 2000


def batch_size_cost(k: int) -> int:
    """Fixed ralph-loop startup overhead for a batch of k changes.

    Models system-prompt + tool-setup tokens that don't scale with file content.
    k=0 means no ralph invocation, so no overhead.
    """
    return RALPH_STARTUP_TOKENS if k > 0 else 0


def _estimate_via_symbols(
    change: FileChange,
    tilth_port: TilthPort,
) -> int:
    seen: dict[tuple[str, str], str] = {}
    for sym in change.symbols:
        key = (sym.name, sym.file)
        if key in seen:
            continue
        locs = tilth_port.search_symbol(sym.name, glob=sym.file)
        if not locs:
            raise ValueError(f"symbol not found: {sym.name!r} in {sym.file!r}")
        loc = locs[0]
        seen[key] = tilth_port.read_section(loc.path, loc.line_start, loc.line_end)
    combined = "\n".join(seen.values())
    return math.ceil(len(_get_encoder().encode(combined)) * HEADROOM)


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


def _log_degradation(root: Path, change: FileChange, reason: str) -> None:
    log_dir = root / ".milknado"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "planning-context-warn.log", "a") as f:
            f.write(f"[weights] symbol resolution failed for {change.path}: {reason}\n")
    except OSError:
        pass


def estimate_tokens_per_symbols(
    change: FileChange,
    root: Path,
    tilth_port: TilthPort | None,
) -> int:
    if change.edit_kind in FLAT_COST:
        return FLAT_COST[change.edit_kind]
    ext = _extension(change.path)
    if change.edit_kind != "modify":
        lines = NEW_FILE_LINES.get(ext, 150)
        tpl = TOKENS_PER_LINE.get(ext, 8)
        return math.ceil(lines * tpl * HEADROOM)
    if tilth_port is not None and change.symbols:
        try:
            return _estimate_via_symbols(change, tilth_port)
        except Exception as exc:
            _log_degradation(root, change, str(exc))
    return _estimate_path_level(change, root)


def estimate_tokens(change: FileChange, root: Path) -> int:
    return estimate_tokens_per_symbols(change, root, None)
