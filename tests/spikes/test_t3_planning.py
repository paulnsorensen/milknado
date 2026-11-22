import json
import os
import resource
import signal
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from contextlib import closing
from pathlib import Path
from typing import Literal, ParamSpec, TypeVar

import msgspec
import pytest

from milknado.domains.batching import plan_batches
from scripts.spikes.t3_planning import __main__ as planning
from scripts.spikes.t3_planning.__main__ import main

P = ParamSpec("P")
R = TypeVar("R")

pytestmark = pytest.mark.real_process

REPO = Path(__file__).resolve().parents[2]
CHANGES = [
    {"id": "use", "path": "use.py", "description": "Use API", "depends_on": ["api"]},
    {"id": "api", "path": "api.py", "description": "Add API"},
]
DEPENDENT = {"version": 1, "budget": 1, "changes": CHANGES}


class Batch(msgspec.Struct, frozen=True):
    index: int
    change_ids: list[str]
    depends_on: list[int]
    oversized: bool


class Plan(msgspec.Struct, frozen=True):
    batches: list[Batch]
    solver_status: str


class Result(msgspec.Struct, frozen=True):
    version: int
    plan: Plan
    crg_index_present: bool


DEPENDENT_BATCHES = [
    Batch(index=0, change_ids=["api"], depends_on=[], oversized=True),
    Batch(index=1, change_ids=["use"], depends_on=[0], oversized=True),
]


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    _ = subprocess.run(["git", "init", "-q", str(root)], check=True)
    _ = (root / "api.py").write_text("def answer() -> int:\n    return 42\n")
    _ = (root / "use.py").write_text("from api import answer\n\nprint(answer())\n")
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    return root


def _request(project: Path, request: Mapping[str, object]) -> Path:
    manifest = project.parent / "request.json"
    _ = manifest.write_text(json.dumps(request))
    return manifest


def _run(project: Path, request: Mapping[str, object], *flags: str) -> int:
    manifest = _request(project, request)
    output = project.parent / "plan.json"
    return main(["--project-root", str(project), str(manifest), str(output), *flags])


def _tree(base: Path) -> dict[str, tuple[int, bytes] | None]:
    return {
        str(path.relative_to(base)): (path.stat().st_mtime_ns, path.read_bytes())
        if path.is_file()
        else None
        for path in base.rglob("*")
    }


def _changed(before: Mapping[str, object], after: Mapping[str, object]) -> set[str]:
    return {path for path in before.keys() | after.keys() if before.get(path) != after.get(path)}


def _build_index(project: Path, home: Path) -> None:
    home.mkdir()
    crg = Path(sys.executable).parent / "code-review-graph"
    _ = subprocess.run(
        [str(crg), "build", "--repo", str(project)],
        cwd=project,
        env={"PATH": "/usr/bin:/bin", "HOME": str(home)},
        capture_output=True,
        check=True,
    )
    assert (project / ".code-review-graph" / "graph.db").is_file()


def _corrupt_index(project: Path) -> None:
    index = project / ".code-review-graph" / "graph.db"
    index.parent.mkdir()
    _ = index.write_bytes(b"not a SQLite database\n" * 8)


def _damage_file_lookup(project: Path) -> None:
    """Overwrite the file-path index page; the header and schema stay readable."""
    index = project / ".code-review-graph" / "graph.db"
    query = "SELECT page_size, rootpage FROM pragma_page_size, sqlite_master WHERE name = ?"
    with closing(sqlite3.connect(index)) as db:
        cursor = db.execute(query, ("idx_nodes_file",))
        page_size, page = msgspec.convert(cursor.fetchone(), type=tuple[int, int], strict=True)
    with index.open("r+b") as stream:
        _ = stream.seek((page - 1) * page_size)
        _ = stream.write(b"\xff" * page_size)


@pytest.mark.parametrize("index", ["none", "built", "corrupt", "damaged"])
def test_plan_writes_only_output(
    project: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    index: Literal["none", "built", "corrupt", "damaged"],
) -> None:
    match index:
        case "built":
            _build_index(project, tmp_path / "home")
        case "corrupt":
            _corrupt_index(project)
        case "damaged":
            _build_index(project, tmp_path / "home")
            _damage_file_lookup(project)
        case "none":
            pass
    manifest = _request(project, DEPENDENT)
    output = tmp_path / "plan.json"
    before = _tree(tmp_path)
    assert main(["--project-root", str(project), str(manifest), str(output)]) == 0
    assert _changed(before, _tree(tmp_path)) == {"plan.json"}
    result = msgspec.json.decode(output.read_bytes(), type=Result)
    assert result.version == 1
    assert result.plan.batches == DEPENDENT_BATCHES
    assert result.plan.solver_status == "OPTIMAL"
    assert result.crg_index_present is (index == "built")
    warnings = capsys.readouterr().err.count("CRG index unreadable")
    assert warnings == (index in {"corrupt", "damaged"})
    umask = _umask()
    assert output.stat().st_mode & 0o777 == 0o600 & ~umask


def _umask() -> int:
    current = os.umask(0)
    _ = os.umask(current)
    return current


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


def test_existing_output_stops_before_planning(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = project.parent / "plan.json"
    _ = output.write_text("keep\n")
    assert _run(project, DEPENDENT) == 1
    assert "File exists" in capsys.readouterr().err
    assert output.read_text() == "keep\n"
    assert not (project / ".milknado").exists()


def _then(step: Callable[P, R], effect: Callable[[], object]) -> Callable[P, R]:
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        result = step(*args, **kwargs)
        _ = effect()
        return result

    return wrapped


def test_plan_published_during_solve_is_kept(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = project.parent / "plan.json"
    publish_other = _then(plan_batches, lambda: output.write_text("other\n"))
    monkeypatch.setattr(planning, "plan_batches", publish_other)
    assert _run(project, DEPENDENT) == 1
    assert "File exists" in capsys.readouterr().err
    assert output.read_text() == "other\n"
    assert sorted(path.name for path in project.parent.iterdir()) == [
        "plan.json",
        "project",
        "request.json",
        "tmp",
    ]


def _limit_file_size() -> None:
    _ = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (64, 64))


def test_failed_write_leaves_no_output(project: Path, tmp_path: Path) -> None:
    manifest = _request(project, DEPENDENT)
    output = tmp_path / "plan.json"
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.spikes.t3_planning"]
        + ["--project-root", str(project), str(manifest), str(output)],
        cwd=REPO,
        env={"PATH": "", "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp_path / "tmp")},
        preexec_fn=_limit_file_size,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "File too large" in completed.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == ["project", "request.json", "tmp"]


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
    assert not (project.parent / "plan.json").exists()
    assert _run(project, request, "--force-single-batch") == 0
    result = msgspec.json.decode((project.parent / "plan.json").read_bytes(), type=Result)
    assert [set(batch.change_ids) for batch in result.plan.batches] == [{str(i) for i in range(6)}]
    assert not (project / ".milknado").exists()
