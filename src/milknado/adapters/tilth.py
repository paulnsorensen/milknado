from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from milknado.domains.common import DegradationMarker, SymbolLocation, TilthMap

_JSON_OBJECT = TypeAdapter(dict[str, object])


class _TilthSearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    output: str = ""


_MATCH_HEADER = re.compile(r"^## (.+):(\d+)(?:-(\d+))? \[")


def _parse_json_object(output: str) -> tuple[dict[str, object] | None, str | None]:
    try:
        raw = json.loads(output)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(raw, dict):
        return None, "top-level JSON is not an object"
    try:
        return _JSON_OBJECT.validate_python(raw, strict=True), None
    except ValidationError as exc:
        return None, str(exc)


def _run_tilth_json(cmd: list[str]) -> dict[str, object] | None:
    """Run a tilth command expecting a JSON object. Returns None on failure."""
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
    data, _ = _parse_json_object(result.stdout)
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
        binary = shutil.which("tilth")
        if binary is None:
            return DegradationMarker(
                source="tilth",
                reason="binary_missing",
                detail="tilth not found on PATH",
            )
        try:
            result = subprocess.run(
                [
                    binary,
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
        data, detail = _parse_json_object(result.stdout)
        if data is None:
            return DegradationMarker(
                source="tilth",
                reason="invalid_json",
                detail=(detail or "invalid JSON object")[:500],
            )
        return TilthMap(scope=scope, budget_tokens=budget_tokens, data=data)

    def search_symbol(
        self,
        keyword: str,
        glob: str | None = None,
    ) -> list[SymbolLocation]:
        binary = shutil.which("tilth")
        if binary is None:
            return []
        cmd = [binary, keyword, "--json"]
        if glob:
            cmd += ["--glob", glob]
        data = _run_tilth_json(cmd)
        if data is None:
            return []
        try:
            response = _TilthSearchResponse.model_validate(data)
        except ValidationError:
            return []
        return _parse_symbol_headers(response.output)

    def read_section(self, path: Path, line_start: int, line_end: int) -> str:
        binary = shutil.which("tilth")
        if binary is None:
            return ""
        try:
            result = subprocess.run(
                [binary, str(path), "--section", f"{line_start}-{line_end}"],
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
