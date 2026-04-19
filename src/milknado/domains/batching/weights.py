from __future__ import annotations

import functools
import math
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


# ---------------------------------------------------------------------------
# Token estimation — design notes and calibration guidance
# ---------------------------------------------------------------------------
#
# CURRENT APPROACH: path dedup in `_tokens_per_scc`
#
# `_tokens_per_scc` (solver.py) calls `estimate_tokens` once per change, then
# sums the results over all changes in an SCC.  Before summing, it deduplicates
# by `FileChange.path`: if two changes share the same path, only the first is
# charged.  This prevents double-counting the same file's token mass when a
# manifest has multiple change IDs touching one path.
#
# HEURISTIC CAVEAT: same file ≠ same work
#
# The dedup gives each path exactly one vote regardless of how many symbols
# in that file are being changed.  Two symbol edits inside `utils.py` is more
# work than one, just not 2×.  The current approach under-counts in that
# scenario — which is the *safer* miscalibration: under-counting makes batches
# slightly larger than intended, but over-counting would fragment them
# unnecessarily, increasing round-trips and sequencing overhead.
#
# BETTER LONG-TERM: per-symbol costing
#
# The right model resolves each `SymbolRef` to a byte range via LSP or a
# tree-sitter parser, tiktoken-encodes only that slice, then sums across
# distinct symbols.  Delta-weighting would further refine estimates:
# `edit_kind=modify` on a small symbol is cheaper than `edit_kind=add` in a
# large file.  This requires structural dependency resolution (the tilth MCP
# or an LSP plugin) and is gated on A2 (tilth integration) being available.
# Until then, path-level dedup is the right tradeoff.
#
# CALIBRATION TRIGGER
#
# If `calibration.jsonl` shows batches consistently coming in well under their
# token budget (< 60% utilisation), bump `HEADROOM` upward (e.g. 1.35 → 1.5)
# or migrate to per-symbol costing.  Over-fragmented batches (many batches with
# one change each) suggest the opposite: HEADROOM is too aggressive or the
# path-dedup under-counts too severely for the workload.
# ---------------------------------------------------------------------------


RALPH_STARTUP_TOKENS = 2000


def batch_size_cost(k: int) -> int:
    """Fixed ralph-loop startup overhead for a batch of k changes.

    Models system-prompt + tool-setup tokens that don't scale with file content.
    k=0 means no ralph invocation, so no overhead.
    """
    return RALPH_STARTUP_TOKENS if k > 0 else 0


def estimate_tokens(change: FileChange, root: Path) -> int:
    if change.edit_kind in FLAT_COST:
        return FLAT_COST[change.edit_kind]
    ext = _extension(change.path)
    tpl = TOKENS_PER_LINE.get(ext, 8)
    if change.edit_kind == "modify":
        resolved_root = root.resolve()
        resolved_path = (root / change.path).resolve()
        if resolved_path.is_relative_to(resolved_root):
            real = _tiktoken_count(resolved_path)
            if real is not None:
                return math.ceil(real * HEADROOM)
    lines = NEW_FILE_LINES.get(ext, 150)
    return math.ceil(lines * tpl * HEADROOM)
