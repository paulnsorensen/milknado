from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from milknado.cli import app

runner = CliRunner()

_CLI_VERSION_OUTPUTS = {
    "git": "git version 2.40.0",
    "code-review-graph": "code-review-graph 0.5.0",
}


def _fake_which(name: str) -> str | None:
    return f"/usr/bin/{name}"


def _fake_run(cmd: list[str], **_kwargs: object) -> MagicMock:
    result = MagicMock()
    result.stdout = _CLI_VERSION_OUTPUTS.get(cmd[0], "") + "\n"
    result.stderr = ""
    return result


def _fake_pkg_version(_name: str) -> str:
    return "1.0.0"


class TestDoctorHappyPath:
    @patch("milknado.domains.common.doctor.pkg_version", side_effect=_fake_pkg_version)
    @patch("milknado.domains.common.doctor.subprocess.run", side_effect=_fake_run)
    @patch("milknado.domains.common.doctor.shutil.which", side_effect=_fake_which)
    def test_exit_0_all_ok(
        self, _mock_which: MagicMock, _mock_run: MagicMock, _mock_ver: MagicMock, tmp_path: Path
    ) -> None:
        db = tmp_path / ".milknado" / "milknado.db"
        db.parent.mkdir(parents=True)
        db.touch()

        result = runner.invoke(app, ["doctor", str(tmp_path)])

        assert result.exit_code == 0
        assert "doctor: ok" in result.output

    @patch("milknado.domains.common.doctor.pkg_version", side_effect=_fake_pkg_version)
    @patch("milknado.domains.common.doctor.subprocess.run", side_effect=_fake_run)
    @patch("milknado.domains.common.doctor.shutil.which", side_effect=_fake_which)
    def test_version_lines_present(
        self, _mock_which: MagicMock, _mock_run: MagicMock, _mock_ver: MagicMock, tmp_path: Path
    ) -> None:
        db = tmp_path / ".milknado" / "milknado.db"
        db.parent.mkdir(parents=True)
        db.touch()

        result = runner.invoke(app, ["doctor", str(tmp_path)])

        assert "git version 2.40.0" in result.output
        assert "code-review-graph 0.5.0" in result.output

    @patch("milknado.domains.common.doctor.pkg_version", side_effect=_fake_pkg_version)
    @patch("milknado.domains.common.doctor.subprocess.run", side_effect=_fake_run)
    @patch("milknado.domains.common.doctor.shutil.which", side_effect=_fake_which)
    def test_no_ralphify_probe_in_output(
        self, _mock_which: MagicMock, _mock_run: MagicMock, _mock_ver: MagicMock, tmp_path: Path
    ) -> None:
        """WHY: the ralphify dep is vendored; its absence is not an issue."""
        db = tmp_path / ".milknado" / "milknado.db"
        db.parent.mkdir(parents=True)
        db.touch()

        result = runner.invoke(app, ["doctor", str(tmp_path)])

        # Confirm no "ralphify: ..." probe line — the format is "name: ..."
        assert not any(
            line.lstrip().startswith("ralphify:") for line in result.output.splitlines()
        )


class TestDoctorDbMissing:
    @patch("milknado.domains.common.doctor.pkg_version", side_effect=_fake_pkg_version)
    @patch("milknado.domains.common.doctor.subprocess.run", side_effect=_fake_run)
    @patch("milknado.domains.common.doctor.shutil.which", side_effect=_fake_which)
    def test_exit_0_db_missing_not_an_issue(
        self, _mock_which: MagicMock, _mock_run: MagicMock, _mock_ver: MagicMock, tmp_path: Path
    ) -> None:
        result = runner.invoke(app, ["doctor", str(tmp_path)])

        assert result.exit_code == 0
        assert "MISSING" in result.output
        assert "doctor: ok" in result.output
