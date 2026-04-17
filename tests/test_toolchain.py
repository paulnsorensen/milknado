from __future__ import annotations

from unittest.mock import patch

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
