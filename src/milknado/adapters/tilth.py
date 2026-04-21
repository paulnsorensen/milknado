from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from milknado.domains.common.protocols import SymbolLocation
from milknado.domains.common.types import DegradationMarker, TilthMap

_MATCH_HEADER = re.compile(r"^## (.+):(\d+)(?:-(\d+))? \[")


def _run_tilth_json(cmd: list[str]) -> dict | None:
    """Run a tilth command expecting JSON output. Returns None on any failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _parse_symbol_headers(output: str) -> list[SymbolLocation]:
    """Parse ## path:start-end [kind] header lines into SymbolLocation list."""
    locations: list[SymbolLocation] = []
    for line in output.splitlines():
        m = _MATCH_HEADER.match(line)
        if not m:
            continue
        path_str, start_str, end_str = m.group(1), m.group(2), m.group(3)
        try:
            start = int(start_str)
            locations.append(
                SymbolLocation(
                    path=Path(path_str),
                    line_start=start,
                    line_end=int(end_str) if end_str else start,
                )
            )
        except ValueError:
            continue
    return locations


class TilthAdapter:
    def structural_map(
        self,
        scope: Path,
        budget_tokens: int,
    ) -> TilthMap | DegradationMarker:
        if shutil.which("tilth") is None:
            return DegradationMarker(
                source="tilth",
                reason="binary_missing",
                detail="tilth not found on PATH",
            )
        try:
            result = subprocess.run(
                [
                    "tilth",
                    "--map",
                    "--json",
                    "--scope",
                    str(scope),
                    "--budget",
                    str(budget_tokens),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return DegradationMarker(
                source="tilth",
                reason="exec_failed",
                detail="tilth execution timed out after 30 seconds",
            )
        if result.returncode != 0:
            return DegradationMarker(
                source="tilth",
                reason="exec_failed",
                detail=(result.stderr or "").strip()[:500],
            )
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            return DegradationMarker(
                source="tilth",
                reason="invalid_json",
                detail=str(exc)[:500],
            )
        if not isinstance(data, dict):
            return DegradationMarker(
                source="tilth",
                reason="invalid_json",
                detail="top-level JSON is not an object",
            )
        return TilthMap(scope=scope, budget_tokens=budget_tokens, data=data)

    def search_symbol(
        self,
        keyword: str,
        glob: str | None = None,
    ) -> list[SymbolLocation]:
        if shutil.which("tilth") is None:
            return []
        cmd = ["tilth", keyword, "--json"]
        if glob:
            cmd += ["--glob", glob]
        data = _run_tilth_json(cmd)
        if data is None:
            return []
        output_text = data.get("output", "")
        if not isinstance(output_text, str):
            return []
        return _parse_symbol_headers(output_text)

    def read_section(self, path: Path, line_start: int, line_end: int) -> str:
        if shutil.which("tilth") is None:
            return ""
        try:
            result = subprocess.run(
                ["tilth", str(path), "--section", f"{line_start}-{line_end}"],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return ""
        if result.returncode != 0:
            return ""
        return result.stdout
