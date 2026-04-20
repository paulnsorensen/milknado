from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from milknado.cli import app

runner = CliRunner()

_VERSION_OUTPUTS = {
    "git": "git version 2.40.0",
    "ralphify": "ralphify 1.0.0",
    "code-review-graph": "code-review-graph 0.5.0",
}


def _fake_which(name: str) -> str | None:
    return f"/usr/bin/{name}"


def _fake_run(cmd: list[str], **_kwargs: object) -> MagicMock:
    result = MagicMock()
    result.stdout = _VERSION_OUTPUTS.get(cmd[0], "") + "\n"
    result.stderr = ""
    return result


def _which_missing_ralphify(name: str) -> str | None:
    return None if name == "ralphify" else f"/usr/bin/{name}"


class TestDoctorHappyPath:
    @patch("milknado.domains.common.doctor.subprocess.run", side_effect=_fake_run)
    @patch("milknado.domains.common.doctor.shutil.which", side_effect=_fake_which)
    def test_exit_0_all_ok(
        self, _mock_which: MagicMock, _mock_run: MagicMock, tmp_path: Path
    ) -> None:
        db = tmp_path / ".milknado" / "milknado.db"
        db.parent.mkdir(parents=True)
        db.touch()

        result = runner.invoke(app, ["doctor", str(tmp_path)])

        assert result.exit_code == 0
        assert "doctor: ok" in result.output

    @patch("milknado.domains.common.doctor.subprocess.run", side_effect=_fake_run)
    @patch("milknado.domains.common.doctor.shutil.which", side_effect=_fake_which)
    def test_version_lines_present(
        self, _mock_which: MagicMock, _mock_run: MagicMock, tmp_path: Path
    ) -> None:
        db = tmp_path / ".milknado" / "milknado.db"
        db.parent.mkdir(parents=True)
        db.touch()

        result = runner.invoke(app, ["doctor", str(tmp_path)])

        assert "git version 2.40.0" in result.output
        assert "ralphify 1.0.0" in result.output
        assert "code-review-graph 0.5.0" in result.output


class TestDoctorMissingTool:
    @patch("milknado.domains.common.doctor.subprocess.run", side_effect=_fake_run)
    @patch("milknado.domains.common.doctor.shutil.which", side_effect=_which_missing_ralphify)
    def test_exit_1_when_tool_missing(
        self, _mock_which: MagicMock, _mock_run: MagicMock, tmp_path: Path
    ) -> None:
        db = tmp_path / ".milknado" / "milknado.db"
        db.parent.mkdir(parents=True)
        db.touch()

        result = runner.invoke(app, ["doctor", str(tmp_path)])

        assert result.exit_code == 1
        assert "not found" in result.output
        assert "doctor: 1 issue(s)" in result.output


class TestDoctorDbMissing:
    @patch("milknado.domains.common.doctor.subprocess.run", side_effect=_fake_run)
    @patch("milknado.domains.common.doctor.shutil.which", side_effect=_fake_which)
    def test_exit_0_db_missing_not_an_issue(
        self, _mock_which: MagicMock, _mock_run: MagicMock, tmp_path: Path
    ) -> None:
        result = runner.invoke(app, ["doctor", str(tmp_path)])

        assert result.exit_code == 0
        assert "MISSING" in result.output
        assert "doctor: ok" in result.output
