"""Run the existing batch planner without applying its plan."""

import argparse
import json
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict, cast

import msgspec

from milknado.app.plan_batches import plan_batches
from milknado.domains.common import MegaBatchAborted


class Request(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    version: Literal[1]
    budget: Annotated[int, msgspec.Meta(ge=1)]
    changes: list[dict[str, object]]
    new_relationships: list[dict[str, object]] | None = None


class Arguments(TypedDict):
    project_root: Path
    force_single_batch: bool
    request: Path
    output: Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--project-root", type=Path, required=True)
    _ = parser.add_argument("--force-single-batch", action="store_true")
    _ = parser.add_argument("request", type=Path)
    _ = parser.add_argument("output", type=Path)
    args = cast(Arguments, cast(object, vars(parser.parse_args(argv))))
    try:
        root = args["project_root"].resolve(strict=True)
        if not root.is_dir():
            raise ValueError("project root must be a directory")
        request = msgspec.json.decode(args["request"].read_bytes(), type=Request)
        plan = plan_batches(
            request.changes,
            request.budget,
            root,
            request.new_relationships,
            force_single_batch=args["force_single_batch"],
        )
        result = {
            "version": 1,
            "plan": plan,
            "crg_index_present": (root / ".code-review-graph" / "graph.db").is_file(),
        }
        with args["output"].open("x", encoding="utf-8") as output:
            _ = output.write(json.dumps(result, indent=2) + "\n")
    except (OSError, ValueError, MegaBatchAborted) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
