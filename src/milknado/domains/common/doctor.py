from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.common.config import MilknadoConfig

PROBED_TOOLS = ("git", "ralphify", "code-review-graph")


@dataclass(frozen=True)
class ToolProbe:
    name: str
    path: str | None
    version: str


@dataclass(frozen=True)
class DoctorReport:
    milknado_version: str
    config_path: Path
    db_path: Path
    db_status: str
    tools: tuple[ToolProbe, ...]


def _probe(name: str) -> ToolProbe:
    path = shutil.which(name)
    if path is None:
        return ToolProbe(name=name, path=None, version="unknown")
    try:
        proc = subprocess.run(
            [name, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        raw = (proc.stdout or proc.stderr).strip()
        ver = raw.splitlines()[0] if raw else "unknown"
    except Exception:  # noqa: BLE001
        ver = "unknown"
    return ToolProbe(name=name, path=path, version=ver)


def run_doctor(config_path: Path, config: MilknadoConfig) -> DoctorReport:
    try:
        milknado_version = pkg_version("milknado")
    except PackageNotFoundError:
        milknado_version = "unknown"

    db_path = config.db_path.resolve()
    db_status = "OK" if db_path.exists() else "MISSING"
    tools = tuple(_probe(t) for t in PROBED_TOOLS)

    return DoctorReport(
        milknado_version=milknado_version,
        config_path=config_path.resolve(),
        db_path=db_path,
        db_status=db_status,
        tools=tools,
    )


def render_report(report: DoctorReport) -> tuple[str, int]:
    lines = [
        f"milknado: {report.milknado_version}",
        f"config: {report.config_path}",
        f"db_path: {report.db_path} ({report.db_status})",
    ]
    issues = 0
    for tool in report.tools:
        if tool.path is None:
            lines.append(f"{tool.name}: not found")
            issues += 1
        else:
            lines.append(f"{tool.name}: {tool.path}  {tool.version}")

    summary = "doctor: ok" if issues == 0 else f"doctor: {issues} issue(s)"
    lines.append(summary)
    return "\n".join(lines), issues
