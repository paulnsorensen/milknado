#!/usr/bin/env python3
"""Enforce the source file-size budget, with explicit measured baseline waivers."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import cast


def _quality_config(root: Path) -> tuple[int, dict[str, dict[str, object]]]:
    with (root / "pyproject.toml").open("rb") as stream:
        config = cast(dict[str, object], tomllib.load(stream))
    tool = cast(dict[str, object], config["tool"])
    milknado = cast(dict[str, object], tool["milknado"])
    quality = cast(dict[str, object], milknado["quality"])
    return cast(int, quality["max_file_lines"]), cast(
        dict[str, dict[str, object]], quality["file_exceptions"]
    )


def _violations(
    root: Path,
    max_lines: int,
    exceptions: dict[str, dict[str, object]],
) -> list[str]:
    failures: list[str] = []
    source_root = root / "src" / "milknado"
    for path in sorted(source_root.rglob("*.py")):
        if "_vendor" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        actual = len(path.read_text(encoding="utf-8").splitlines())
        waiver = exceptions.get(relative)
        limit = cast(int, waiver["max_lines"]) if waiver else max_lines
        if actual > limit:
            failures.append(f"{relative}: {actual} lines (limit {limit})")
    return failures


def _stale_waivers(root: Path, exceptions: dict[str, dict[str, object]]) -> list[str]:
    return [relative for relative in exceptions if not (root / relative).is_file()]


def main(root: Path | None = None) -> int:
    root = root if root is not None else Path(__file__).resolve().parents[1]
    max_lines, exceptions = _quality_config(root)
    missing_reasons = [path for path, waiver in exceptions.items() if not waiver.get("reason")]
    if missing_reasons:
        print("File-size waivers need reasons: " + ", ".join(missing_reasons))
        return 1
    stale = _stale_waivers(root, exceptions)
    if stale:
        print("File-size waivers reference files that no longer exist: " + ", ".join(stale))
        return 1
    failures = _violations(root, max_lines, exceptions)
    if failures:
        print(f"Source files exceed the {max_lines}-line budget:")
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
