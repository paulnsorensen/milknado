from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from milknado.domains.common.toolchain import (
    REQUIRED_RUST_TOOLS,
    get_required_tool_status,
    install_missing_rust_tools,
)


class FakeToolchain:
    def __init__(
        self,
        executables: dict[str, str] | None = None,
        *,
        cargo_home: Path | None = None,
        home: Path = Path("/home/test"),
        materialize_installs: bool = True,
    ) -> None:
        self.executables = executables or {}
        self.commands: list[list[str]] = []
        self.returncodes: dict[tuple[str, ...], int] = {}
        self.cargo_home = cargo_home
        self._home = home
        self.materialize_installs = materialize_installs

    def find_executable(self, command: str) -> str | None:
        return self.executables.get(command)

    def run(self, argv: list[str], *, check: bool = False):
        self.commands.append(argv)
        returncode = self.returncodes.get(tuple(argv), 0)
        if (
            self.materialize_installs
            and returncode == 0
            and argv[1:2] in (["binstall"], ["install"])
            and "--help" not in argv
        ):
            self.executables[argv[-1]] = f"/home/test/.cargo/bin/{argv[-1]}"
        return SimpleNamespace(returncode=returncode, stdout="", stderr="")

    def system(self) -> str:
        return "Linux"

    def machine(self) -> str:
        return "x86_64"

    def environment(self, name: str) -> str | None:
        return str(self.cargo_home) if name == "CARGO_HOME" and self.cargo_home else None

    def home(self) -> Path:
        return self._home


def test_status_uses_toolchain_port() -> None:
    toolchain = FakeToolchain({"tilth": "/bin/tilth"})

    statuses = get_required_tool_status(toolchain)

    assert [(status.name, status.installed, status.path) for status in statuses] == [
        ("tilth", True, "/bin/tilth"),
        ("mergiraf", False, None),
        ("rtk", False, None),
    ]


def test_default_status_composes_system_adapter() -> None:
    toolchain = FakeToolchain({"tilth": "/bin/tilth"})
    with patch(
        "milknado.adapters.toolchain.SystemToolchainAdapter",
        return_value=toolchain,
    ):
        statuses = get_required_tool_status()

    assert statuses[0].path == "/bin/tilth"


def test_no_cargo_marks_only_missing_tools_failed() -> None:
    toolchain = FakeToolchain(
        {tool.name: f"/bin/{tool.name}" for tool in REQUIRED_RUST_TOOLS if tool.name != "mergiraf"}
    )

    installed, failed = install_missing_rust_tools(toolchain)

    assert installed == []
    assert failed == ["mergiraf"]
    assert toolchain.commands == []


def test_binstall_and_cargo_install_are_selected_per_tool() -> None:
    toolchain = FakeToolchain({"cargo": "/bin/cargo"})
    toolchain.executables["tilth"] = "/home/test/.cargo/bin/tilth"
    toolchain.executables["rtk"] = "/home/test/.cargo/bin/rtk"

    installed, failed = install_missing_rust_tools(toolchain)

    assert installed == ["mergiraf"]
    assert failed == []
    assert toolchain.commands == [
        ["cargo", "binstall", "--help"],
        ["cargo", "binstall", "--no-confirm", "mergiraf"],
    ]


def test_install_failure_is_reported() -> None:
    toolchain = FakeToolchain(
        {"cargo": "/bin/cargo", "mergiraf": "/bin/mergiraf", "rtk": "/bin/rtk"}
    )
    toolchain.returncodes[("cargo", "binstall", "--no-confirm", "tilth")] = 1

    installed, failed = install_missing_rust_tools(toolchain)

    assert installed == []
    assert failed == ["tilth"]


def test_successful_command_without_installed_binary_is_failure(tmp_path: Path) -> None:
    toolchain = FakeToolchain(
        {"cargo": "/bin/cargo", "mergiraf": "/bin/mergiraf", "rtk": "/bin/rtk"},
        home=tmp_path,
        materialize_installs=False,
    )

    installed, failed = install_missing_rust_tools(toolchain)

    assert installed == []
    assert failed == ["tilth"]


def test_cargo_home_binary_proves_install_success(tmp_path: Path) -> None:
    cargo_home = tmp_path / "cargo"
    binary = cargo_home / "bin" / "tilth"
    binary.parent.mkdir(parents=True)
    binary.touch()
    toolchain = FakeToolchain(
        {"cargo": "/bin/cargo", "mergiraf": "/bin/mergiraf", "rtk": "/bin/rtk"},
        cargo_home=cargo_home,
        materialize_installs=False,
    )

    installed, failed = install_missing_rust_tools(toolchain)

    assert installed == ["tilth"]
    assert failed == []
