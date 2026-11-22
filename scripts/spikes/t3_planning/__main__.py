"""Plan batches with the installed Milknado solver and write only the JSON plan."""

import argparse
import errno
import os
import shutil
import sys
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal, TypedDict, cast

import msgspec

from milknado.adapters.crg import CrgAdapter
from milknado.domains.batching import MEGA_BATCH_THRESHOLD, BatchPlan, plan_batches
from milknado.domains.common import MegaBatchAborted
from milknado.domains.planning import MANIFEST_VERSION, decode_manifest

_INDEX = Path(".code-review-graph") / "graph.db"


class Request(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    version: Literal[1]
    budget: Annotated[int, msgspec.Meta(ge=1)]
    changes: list[dict[str, object]]
    new_relationships: list[dict[str, object]] | None = None


class Result(msgspec.Struct, frozen=True, kw_only=True):
    version: Literal[1] = 1
    plan: BatchPlan
    crg_index_present: bool


class Arguments(TypedDict):
    project_root: Path
    force_single_batch: bool
    request: Path
    output: Path


@contextmanager
def _index_copy(root: Path) -> Generator[CrgAdapter | None, None, None]:
    """Yield an adapter over a temporary index copy; opening a GraphStore writes WAL files."""
    index = root / _INDEX
    if not index.is_file():
        yield None
        return
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / _INDEX
        copy.parent.mkdir()
        for source in (index, index.with_name(f"{index.name}-wal")):
            if source.is_file():
                _ = shutil.copyfile(source, copy.with_name(source.name))
        yield CrgAdapter(Path(scratch))


def _plan(root: Path, request: Request, *, force_single_batch: bool) -> Result:
    manifest = decode_manifest(
        {
            "manifest_version": MANIFEST_VERSION,
            "goal": "(t3-planning-spike)",
            "goal_summary": "T3 planning spike request",
            "spec_path": None,
            "changes": request.changes,
            "new_relationships": request.new_relationships or [],
        }
    )
    with _index_copy(root) as crg:
        plan = plan_batches(
            manifest.changes,
            request.budget,
            crg=crg,
            new_relationships=manifest.new_relationships,
            root=root,
        )
    count = plan.mega_batch_change_count
    if count is not None and not force_single_batch:
        raise MegaBatchAborted(change_count=count, threshold=MEGA_BATCH_THRESHOLD)
    return Result(plan=plan, crg_index_present=crg is not None)


def _publish(output: Path, document: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    draft = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            _ = stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        _ = draft.replace(output)
    except BaseException:
        draft.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--project-root", type=Path, required=True)
    _ = parser.add_argument("--force-single-batch", action="store_true")
    _ = parser.add_argument("request", type=Path)
    _ = parser.add_argument("output", type=Path)
    args = cast(Arguments, cast(object, vars(parser.parse_args(argv))))
    output = args["output"]
    try:
        root = args["project_root"].resolve(strict=True)
        if not root.is_dir():
            raise ValueError("project root must be a directory")
        request = msgspec.json.decode(args["request"].read_bytes(), type=Request)
        if output.exists() or output.is_symlink():
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(output))
        result = _plan(root, request, force_single_batch=args["force_single_batch"])
        _publish(output, msgspec.json.format(msgspec.json.encode(result), indent=2) + b"\n")
    except (OSError, ValueError, MegaBatchAborted) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
