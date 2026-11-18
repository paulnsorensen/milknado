import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

import msgspec
import pytest

from scripts.spikes.t3_planning.__main__ import main

pytestmark = pytest.mark.real_process


class Batch(msgspec.Struct, frozen=True):
    index: int
    change_ids: list[str]
    depends_on: list[int]
    oversized: bool


class Plan(msgspec.Struct, frozen=True):
    batches: list[Batch]


class Result(msgspec.Struct, frozen=True):
    plan: Plan
    crg_index_present: bool


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    _ = subprocess.run(["git", "init", "-q", str(root)], check=True)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    return root


def _run(project: Path, request: Mapping[str, object], *flags: str) -> int:
    manifest = project.parent / "request.json"
    _ = manifest.write_text(json.dumps(request))
    output = project.parent / "plan.json"
    return main(["--project-root", str(project), str(manifest), str(output), *flags])


def test_plan_preserves_dependency_without_crg(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    changes = [
        {"id": "use", "path": "use.py", "description": "Use API", "depends_on": ["api"]},
        {"id": "api", "path": "api.py", "description": "Add API"},
    ]
    assert _run(project, {"version": 1, "budget": 1, "changes": changes}) == 0
    result = msgspec.json.decode((project.parent / "plan.json").read_bytes(), type=Result)
    assert result.plan.batches == [
        Batch(index=0, change_ids=["api"], depends_on=[], oversized=True),
        Batch(index=1, change_ids=["use"], depends_on=[0], oversized=True),
    ]
    assert result.crg_index_present is False
    assert "proceeding without graph" in caplog.text
    assert {path.name for path in project.iterdir()} == {".git", ".milknado"}
    assert {path.name for path in (project / ".milknado").iterdir()} == {"calibration.jsonl"}
    assert len((project / ".milknado" / "calibration.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize(
    ("request_data", "error"),
    [
        ({"version": 2, "budget": 1, "changes": []}, "Invalid enum value"),
        ({"version": 1, "budget": 0, "changes": []}, "Expected `int` >= 1"),
        (
            {
                "version": 1,
                "budget": 1,
                "changes": [{"id": "bad", "path": "../bad.py", "description": "Add module"}],
            },
            "manifest is not a valid milknado.plan.v2 object",
        ),
    ],
)
def test_invalid_request(
    project: Path, capsys: pytest.CaptureFixture[str], request_data: dict[str, object], error: str
) -> None:
    assert _run(project, request_data) == 1
    assert error in capsys.readouterr().err
    assert not (project.parent / "plan.json").exists()
    assert not (project / ".milknado").exists()


def test_mega_batch_requires_explicit_override(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    changes = [
        {
            "id": str(i),
            "path": f"change{i}.py",
            "description": "Add module",
            "depends_on": [str((i + 1) % 6)],
        }
        for i in range(6)
    ]
    request = {"version": 1, "budget": 150_000, "changes": changes}
    assert _run(project, request) == 1
    assert "Mega-batch detected: 6 changes" in capsys.readouterr().err
    assert not (project / ".milknado").exists()
    assert _run(project, request, "--force-single-batch") == 0
    result = msgspec.json.decode((project.parent / "plan.json").read_bytes(), type=Result)
    assert [set(batch.change_ids) for batch in result.plan.batches] == [{str(i) for i in range(6)}]
