from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common.protocols import CommandResult
from milknado.domains.common.toolchain import (
    REQUIRED_RUST_TOOLS,
    get_required_tool_status,
    install_missing_rust_tools,
)


@dataclass(frozen=True)
class _CommandResult(CommandResult):
    returncode: int
    stdout: str
    stderr: str


class FakeToolchain:
    def __init__(
        self,
        executables: dict[str, str] | None = None,
        *,
        cargo_home: Path | None = None,
        home: Path | None = None,
        materialize_installs: bool = True,
    ) -> None:
        self.executables: dict[str, str] = executables or {}
        self.commands: list[list[str]] = []
        self.returncodes: dict[tuple[str, ...], int] = {}
        self.cargo_home: Path | None = cargo_home
        self._home: Path = home or Path("/home/test")
        self.materialize_installs: bool = materialize_installs

    def find_executable(self, command: str) -> str | None:
        return self.executables.get(command)

    def run(self, argv: list[str], *, check: bool = False) -> _CommandResult:
        del check
        self.commands.append(argv)
        returncode = self.returncodes.get(tuple(argv), 0)
        if (
            self.materialize_installs
            and returncode == 0
            and argv[1:2] in (["binstall"], ["install"])
            and "--help" not in argv
        ):
            self.executables[argv[-1]] = f"/home/test/.cargo/bin/{argv[-1]}"
        return _CommandResult(returncode=returncode, stdout="", stderr="")

    def system(self) -> str:
        return "Linux"

    def machine(self) -> str:
        return "x86_64"

    def environment(self, name: str) -> str | None:
        return str(self.cargo_home) if name == "CARGO_HOME" and self.cargo_home else None

    def home(self) -> Path:
        return self._home


def test_status_uses_toolchain_port() -> None:
    toolchain = FakeToolchain({"mergiraf": "/bin/mergiraf"})

    statuses = get_required_tool_status(toolchain)

    assert [(status.name, status.installed, status.path) for status in statuses] == [
        ("mergiraf", True, "/bin/mergiraf"),
        ("rtk", False, None),
    ]


def test_no_cargo_marks_only_missing_tools_failed() -> None:
    toolchain = FakeToolchain(
        {tool.name: f"/bin/{tool.name}" for tool in REQUIRED_RUST_TOOLS if tool.name != "mergiraf"}
    )

    installed, failed = install_missing_rust_tools(toolchain)

    assert installed == []
    assert failed == ["mergiraf"]
    assert toolchain.commands == []


def test_binstall_and_cargo_install_are_selected_per_tool() -> None:
    toolchain = FakeToolchain({"cargo": "/bin/cargo", "rtk": "/home/test/.cargo/bin/rtk"})

    installed, failed = install_missing_rust_tools(toolchain)

    assert installed == ["mergiraf"]
    assert failed == []
    assert toolchain.commands == [
        ["cargo", "binstall", "--help"],
        ["cargo", "binstall", "--no-confirm", "mergiraf"],
    ]


def test_install_failure_is_reported() -> None:
    toolchain = FakeToolchain({"cargo": "/bin/cargo", "rtk": "/bin/rtk"})
    toolchain.returncodes[("cargo", "binstall", "--no-confirm", "mergiraf")] = 1

    installed, failed = install_missing_rust_tools(toolchain)

    assert installed == []
    assert failed == ["mergiraf"]


def test_successful_command_without_installed_binary_is_failure(tmp_path: Path) -> None:
    toolchain = FakeToolchain(
        {"cargo": "/bin/cargo", "rtk": "/bin/rtk"},
        home=tmp_path,
        materialize_installs=False,
    )

    installed, failed = install_missing_rust_tools(toolchain)

    assert installed == []
    assert failed == ["mergiraf"]


def test_cargo_home_binary_proves_install_success(tmp_path: Path) -> None:
    cargo_home = tmp_path / "cargo"
    binary = cargo_home / "bin" / "mergiraf"
    binary.parent.mkdir(parents=True)
    binary.touch()
    toolchain = FakeToolchain(
        {"cargo": "/bin/cargo", "rtk": "/bin/rtk"},
        cargo_home=cargo_home,
        materialize_installs=False,
    )

    installed, failed = install_missing_rust_tools(toolchain)

    assert installed == ["mergiraf"]
    assert failed == []
