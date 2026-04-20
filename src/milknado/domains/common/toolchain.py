from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RustTool:
    name: str
    install_args: tuple[str, ...]
    supports_binstall: bool = True


# rtk is rtk-ai/rtk (token-optimizing CLI), not the unrelated `rtk` crate on crates.io.
REQUIRED_RUST_TOOLS: tuple[RustTool, ...] = (
    RustTool(name="tilth", install_args=("tilth",)),
    RustTool(name="mergiraf", install_args=("mergiraf",)),
    RustTool(
        name="rtk",
        install_args=("--git", "https://github.com/rtk-ai/rtk"),
        supports_binstall=False,
    ),
)


@dataclass(frozen=True)
class ToolStatus:
    name: str
    installed: bool
    path: str | None = None


def get_required_tool_status() -> list[ToolStatus]:
    statuses: list[ToolStatus] = []
    for tool in REQUIRED_RUST_TOOLS:
        path = shutil.which(tool.name)
        statuses.append(
            ToolStatus(name=tool.name, installed=path is not None, path=path),
        )
    return statuses


def install_missing_rust_tools() -> tuple[list[str], list[str]]:
    installed: list[str] = []
    current_status = get_required_tool_status()

    has_cargo = shutil.which("cargo") is not None
    if not has_cargo:
        failed_without_cargo = [status.name for status in current_status if not status.installed]
        return installed, failed_without_cargo

    use_binstall = _cargo_subcommand_exists("binstall")
    tool_by_name = {t.name: t for t in REQUIRED_RUST_TOOLS}
    failed: list[str] = []
    for status in current_status:
        if status.installed:
            continue
        tool = tool_by_name[status.name]
        if use_binstall and tool.supports_binstall:
            cmd = ["cargo", "binstall", "--no-confirm", *tool.install_args]
        else:
            cmd = ["cargo", "install", "--locked", *tool.install_args]
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0 and _cargo_bin_exists(status.name):
            installed.append(status.name)
        else:
            failed.append(status.name)
    return installed, failed


def _cargo_bin_exists(name: str) -> bool:
    if shutil.which(name) is not None:
        return True
    cargo_home = os.environ.get("CARGO_HOME", os.path.expanduser("~/.cargo"))
    return (Path(cargo_home) / "bin" / name).exists()


def _cargo_subcommand_exists(subcommand: str) -> bool:
    result = subprocess.run(
        ["cargo", subcommand, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0
