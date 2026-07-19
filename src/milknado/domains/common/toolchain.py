from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common.protocols import ToolchainPort


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


def get_required_tool_status(toolchain: ToolchainPort) -> list[ToolStatus]:
    return [
        ToolStatus(
            name=tool.name,
            installed=(path := toolchain.find_executable(tool.name)) is not None,
            path=path,
        )
        for tool in REQUIRED_RUST_TOOLS
    ]


def install_missing_rust_tools(
    toolchain: ToolchainPort,
) -> tuple[list[str], list[str]]:
    installed: list[str] = []
    current_status = get_required_tool_status(toolchain)

    if toolchain.find_executable("cargo") is None:
        failed_without_cargo = [status.name for status in current_status if not status.installed]
        return installed, failed_without_cargo

    use_binstall = _cargo_subcommand_exists("binstall", toolchain)
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
        result = toolchain.run(cmd)
        if result.returncode == 0 and _cargo_bin_exists(status.name, toolchain):
            installed.append(status.name)
        else:
            failed.append(status.name)
    return installed, failed


def _cargo_bin_exists(name: str, toolchain: ToolchainPort) -> bool:
    if toolchain.find_executable(name) is not None:
        return True
    cargo_home = toolchain.environment("CARGO_HOME")
    cargo_root = Path(cargo_home) if cargo_home else toolchain.home() / ".cargo"
    return (cargo_root / "bin" / name).exists()


def _cargo_subcommand_exists(subcommand: str, toolchain: ToolchainPort) -> bool:
    return toolchain.run(["cargo", subcommand, "--help"]).returncode == 0
