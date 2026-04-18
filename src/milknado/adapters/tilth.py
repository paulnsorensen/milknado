from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from milknado.domains.common.types import DegradationMarker, TilthMap


class TilthAdapter:
    def structural_map(
        self, scope: Path, budget_tokens: int,
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
                    "tilth", "--map", "--json",
                    "--scope", str(scope),
                    "--budget", str(budget_tokens),
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
