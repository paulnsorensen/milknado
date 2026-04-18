from __future__ import annotations

from unittest.mock import MagicMock, patch

from milknado.domains.common.toolchain import ToolStatus, install_missing_rust_tools


class TestInstallMissingRustTools:
    @patch("milknado.domains.common.toolchain.get_required_tool_status")
    @patch("milknado.domains.common.toolchain.shutil.which")
    def test_no_cargo_all_tools_already_installed(
        self,
        mock_which,
        mock_status,
    ) -> None:
        mock_status.return_value = [
            ToolStatus(name="tilth", installed=True, path="/usr/local/bin/tilth"),
            ToolStatus(name="mergiraf", installed=True, path="/usr/local/bin/mergiraf"),
        ]
        mock_which.side_effect = lambda name: None if name == "cargo" else "/usr/local/bin/tool"

        installed, failed = install_missing_rust_tools()

        assert installed == []
        assert failed == []

    @patch("milknado.domains.common.toolchain.get_required_tool_status")
    @patch("milknado.domains.common.toolchain.shutil.which")
    def test_no_cargo_only_missing_tools_marked_failed(
        self,
        mock_which,
        mock_status,
    ) -> None:
        mock_status.return_value = [
            ToolStatus(name="tilth", installed=True, path="/usr/local/bin/tilth"),
            ToolStatus(name="mergiraf", installed=False, path=None),
        ]
        mock_which.side_effect = lambda name: None if name == "cargo" else "/usr/local/bin/tool"

        installed, failed = install_missing_rust_tools()

        assert installed == []
        assert failed == ["mergiraf"]

    @patch("milknado.domains.common.toolchain.subprocess.run")
    @patch("milknado.domains.common.toolchain._cargo_bin_exists")
    @patch("milknado.domains.common.toolchain._cargo_subcommand_exists")
    @patch("milknado.domains.common.toolchain.get_required_tool_status")
    @patch("milknado.domains.common.toolchain.shutil.which")
    def test_cargo_binstall_used_when_available(
        self,
        mock_which,
        mock_status,
        mock_subcommand,
        mock_bin_exists,
        mock_run,
    ) -> None:
        mock_which.side_effect = lambda name: "/usr/bin/cargo" if name == "cargo" else None
        mock_status.return_value = [
            ToolStatus(name="tilth", installed=False, path=None),
        ]
        mock_subcommand.return_value = True
        mock_run.return_value = MagicMock(returncode=0)
        mock_bin_exists.return_value = True

        installed, failed = install_missing_rust_tools()

        assert installed == ["tilth"]
        assert failed == []
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[1] == "binstall"

    @patch("milknado.domains.common.toolchain.subprocess.run")
    @patch("milknado.domains.common.toolchain._cargo_bin_exists")
    @patch("milknado.domains.common.toolchain._cargo_subcommand_exists")
    @patch("milknado.domains.common.toolchain.get_required_tool_status")
    @patch("milknado.domains.common.toolchain.shutil.which")
    def test_cargo_install_fallback_when_no_binstall(
        self,
        mock_which,
        mock_status,
        mock_subcommand,
        mock_bin_exists,
        mock_run,
    ) -> None:
        mock_which.side_effect = lambda name: "/usr/bin/cargo" if name == "cargo" else None
        mock_status.return_value = [
            ToolStatus(name="mergiraf", installed=False, path=None),
        ]
        mock_subcommand.return_value = False
        mock_run.return_value = MagicMock(returncode=0)
        mock_bin_exists.return_value = True

        installed, failed = install_missing_rust_tools()

        assert installed == ["mergiraf"]
        assert failed == []
        cmd = mock_run.call_args[0][0]
        assert cmd[1] == "install"
        assert "--locked" in cmd

    @patch("milknado.domains.common.toolchain.subprocess.run")
    @patch("milknado.domains.common.toolchain._cargo_subcommand_exists")
    @patch("milknado.domains.common.toolchain.get_required_tool_status")
    @patch("milknado.domains.common.toolchain.shutil.which")
    def test_cargo_install_failure_marks_tool_failed(
        self,
        mock_which,
        mock_status,
        mock_subcommand,
        mock_run,
    ) -> None:
        mock_which.side_effect = lambda name: "/usr/bin/cargo" if name == "cargo" else None
        mock_status.return_value = [
            ToolStatus(name="tilth", installed=False, path=None),
        ]
        mock_subcommand.return_value = False
        mock_run.return_value = MagicMock(returncode=1)

        installed, failed = install_missing_rust_tools()

        assert installed == []
        assert failed == ["tilth"]
