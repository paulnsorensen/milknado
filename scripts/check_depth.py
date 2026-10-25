#!/usr/bin/env python3
"""Depth ratchet: measure six depth keys per slice and compare each slice with its own baseline.

Ported from sliced-bread-depth-ratchet (paulnsorensen/sliced-bread-architecture#71) through the
Python `ast` port in paulnsorensen/skillz-that-grillz. Every key is lower-is-better with
tolerance 0. A slice is a `[[slice]]` of `slices.toml`; its crust is its `crust` list. The metric
definitions live in `scripts/depth_metrics.py`.

    python -m scripts.check_depth measure [--json]
    python -m scripts.check_depth init [--mode report|block]   refuses to overwrite the baseline
    python -m scripts.check_depth check                        exit 1 on a regression in block mode
    python -m scripts.check_depth tighten [--accept-denominator-gains] [--adopt-new]
    python -m scripts.check_depth diff-baseline --base REF     list loosening; informs only

Exit 0 is a pass or report mode, 1 a regression in block mode or a refused tighten, and 2 an
input error. Dead internal code stays in the denominators: the vulture lane removes it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import cast

from scripts import slice_map
from scripts.depth_metrics import Measure, Value, computed_imports, measure
from scripts.slice_map import Graph, MapError

ROOT = Path(__file__).resolve().parents[1]
BASELINE_FILE = "depth-ratchet.json"
# Increment when a definition changes a number; check then refuses an old baseline.
MEASUREMENT_VERSION = 1
SHARES = {"M08.share", "M10.share"}
KEYS = (
    "M08.crust_logic_statements",
    "M08.share",
    "M09.public_surface",
    "M10.crust_complexity",
    "M10.share",
    "M12.unused_exports",
)
MODES = ("report", "block")
SIZE_AWARE_STATEMENTS = 200  # Advisory only: never blocks, in any mode.
Caps = dict[str, dict[str, Value]]  # slice -> key -> cap


def _show(value: Value) -> str:
    return f"{value[0]}/{value[1]}" if isinstance(value, tuple) else str(value)


def _number(value: Value) -> Fraction:
    return Fraction(value[0], value[1] or 1) if isinstance(value, tuple) else Fraction(value)


def _numerator(value: Value) -> int:
    return value if isinstance(value, int) else value[0]


def _parse_value(key: str, raw: object, where: str) -> Value:
    if key in SHARES:
        parts = raw.split("/") if isinstance(raw, str) else []
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            return int(parts[0]), int(parts[1])
        raise MapError(f"{where}: {key} must be a 'numerator/denominator' string")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw
    raise MapError(f"{where}: {key} must be a non-negative integer")


@dataclass(frozen=True)
class Baseline:
    mode: str
    slices: Caps


def _document(text: str, where: str) -> dict[str, object]:
    try:
        document = cast(object, json.loads(text))
    except json.JSONDecodeError as error:
        raise MapError(f"{where}: {error}") from error
    fields = {"version", "measurement_version", "mode", "slices"}
    return slice_map.table(document, where, set(), fields)


def decode(text: str, where: str) -> Baseline:
    top = _document(text, where)
    if top.get("version") != 1 or top.get("mode") not in MODES or "slices" not in top:
        raise MapError(f"{where}: version must be 1, mode one of {MODES}, and slices present")
    if top.get("measurement_version") != MEASUREMENT_VERSION:
        version = top.get("measurement_version")
        raise MapError(
            f"{where}: measurement_version {version} is not {MEASUREMENT_VERSION}; "
            + "a human re-baselines after a definition change"
        )
    caps: Caps = {}
    for name, row in slice_map.table(top["slices"], f"{where} slices", set(), None).items():
        fields = slice_map.table(row, f"{where} slice {name}", set(), set(KEYS))
        caps[name] = {
            key: _parse_value(key, raw, f"{where} {name}") for key, raw in fields.items()
        }
    return Baseline(str(top["mode"]), caps)


def encode(baseline: Baseline) -> str:
    slices = {
        name: {
            key: _show(caps[key]) if key in SHARES else caps[key] for key in KEYS if key in caps
        }
        for name, caps in sorted(baseline.slices.items())
    }
    document = {
        "version": 1,
        "measurement_version": MEASUREMENT_VERSION,
        "mode": baseline.mode,
        "slices": slices,
    }
    return json.dumps(document, indent=2) + "\n"


@dataclass
class Comparison:
    accept_gains: bool = False
    lines: list[str] = field(default_factory=list[str])
    regressed: bool = False
    orphaned: bool = False
    improved: Caps = field(default_factory=dict[str, dict[str, Value]])  # slice -> key -> cap
    held: list[str] = field(default_factory=list[str])
    new: list[str] = field(default_factory=list[str])

    def judge(self, slot: tuple[str, str], value: Value, cap: Value) -> None:
        """Record one `(slice, key)` slot against its cap."""
        name, key = slot
        if _number(value) > _number(cap):
            self.regressed = True
            note = ""
            if key in SHARES and _numerator(value) <= _numerator(cap):
                note = " denominator-shrink: the crust did not grow; internal code shrank"
            self.lines.append(f"regressed {name} {key} {_show(cap)} -> {_show(value)}{note}")
        elif _number(value) < _number(cap):
            if key in SHARES and not self.accept_gains and _numerator(value) >= _numerator(cap):
                self.held.append(
                    f"held {name} {key} {_show(cap)} -> {_show(value)}: the share fell only "
                    + "because internal code grew; a human may pass --accept-denominator-gains"
                )
            else:
                self.improved.setdefault(name, {})[key] = value
                self.lines.append(
                    f"improved {name} {key} {_show(cap)} -> {_show(value)}: run `tighten`"
                )


def compare(
    found: dict[str, Measure], baseline: Baseline, accept_gains: bool = False
) -> Comparison:
    result = Comparison(accept_gains)
    for name, measured in sorted(found.items()):
        caps = baseline.slices.get(name)
        if caps is None:
            continue
        for key in KEYS:
            value = measured.values[key]
            if key in caps:
                result.judge((name, key), value, caps[key])
            else:
                result.lines.append(f"new {name} {key} = {_show(value)} (no cap)")
    result.new = sorted(set(found) - set(baseline.slices))
    result.lines += [
        f"new {name}: no baseline; `tighten --adopt-new` adds it" for name in result.new
    ]
    orphans = sorted(set(baseline.slices) - set(found))
    result.lines += [
        f"orphan {name}: {BASELINE_FILE} holds an undeclared slice" for name in orphans
    ]
    result.orphaned = bool(orphans)
    return result


def _advisories(found: dict[str, Measure], graph: Graph) -> list[str]:
    lines = [
        f"advisory M08.size-aware {name}: crust holds {_show(share)} of the slice's statements"
        for name, measured in sorted(found.items())
        if _number(measured.values["M08.crust_logic_statements"]) > SIZE_AWARE_STATEMENTS
        and _number(share := measured.values["M08.share"]) > Fraction(1, 2)
    ]
    if computed := computed_imports(graph):
        lines.append(f"coverage: computed imports that no number follows: {', '.join(computed)}")
    return lines


def _table(found: dict[str, Measure]) -> str:
    header = ["slice", *KEYS, "module_init"]
    rows = [
        [name, *(_show(measured.values[key]) for key in KEYS), str(measured.module_init)]
        for name, measured in sorted(found.items())
    ]
    widths = [max(len(row[index]) for row in [header, *rows]) for index in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in [header, *rows]
    )


class _Args(argparse.Namespace):
    command: str = ""
    json: bool = False
    mode: str = "report"
    accept_denominator_gains: bool = False
    adopt_new: bool = False
    base: str = ""


@dataclass(frozen=True)
class _Run:
    root: Path
    args: _Args
    graph: Graph
    found: dict[str, Measure]

    @property
    def path(self) -> Path:
        return self.root / BASELINE_FILE

    def baseline(self) -> Baseline:
        if not self.path.is_file():
            raise MapError(f"{BASELINE_FILE} is missing; run `init` in the adoption change")
        return decode(self.path.read_text(encoding="utf-8"), BASELINE_FILE)

    def write(self, baseline: Baseline) -> None:
        _ = self.path.write_text(encode(baseline), encoding="utf-8")


def _measure(run: _Run) -> int:
    if not run.args.json:
        print(_table(run.found))
        return 0
    document = {
        name: {
            "values": {key: _show(value) for key, value in measured.values.items()},
            "exports": measured.exports,
            "unused_exports": measured.unused,
            "module_init": measured.module_init,
        }
        for name, measured in sorted(run.found.items())
    }
    print(json.dumps(document, indent=2))
    return 0


def _init(run: _Run) -> int:
    if run.path.exists():
        print(f"{BASELINE_FILE} exists; init refuses to overwrite it", file=sys.stderr)
        return 1
    run.write(Baseline(run.args.mode, {n: dict(m.values) for n, m in run.found.items()}))
    print(_table(run.found))
    return 0


def _check(run: _Run) -> int:
    baseline = run.baseline()
    result = compare(run.found, baseline)
    lines = result.lines + _advisories(run.found, run.graph)
    passed = f"depth ratchet: pass ({len(run.found)} slices, mode {baseline.mode})"
    print("\n".join(lines) if lines else passed)
    if (result.regressed or result.orphaned) and baseline.mode == "block":
        print(
            "depth ratchet: a slice is shallower than its baseline. Move the logic behind the "
            + f"crust, or ask the human owner to raise the cap in {BASELINE_FILE}"
        )
        return 1
    return 0


def _tighten(run: _Run) -> int:
    baseline = run.baseline()
    result = compare(run.found, baseline, run.args.accept_denominator_gains)
    if result.regressed or result.orphaned:
        print("\n".join(result.lines))
        print("tighten refuses while a metric regressed or a slice is orphaned; nothing written")
        return 1
    slices = {name: dict(caps) for name, caps in baseline.slices.items()}
    for name, caps in result.improved.items():
        slices[name].update(caps)
    if run.args.adopt_new:
        slices.update({name: dict(run.found[name].values) for name in result.new})
    run.write(Baseline(baseline.mode, slices))
    improved = [
        line.replace(": run `tighten`", "") for line in result.lines if line.startswith("improved")
    ]
    print("\n".join(improved + result.held) or "tighten: no cap changed")
    return 0


def diff_baseline(root: Path, base: str) -> int:
    """Print each loosening of the baseline against `base`; it informs and approves nothing."""
    shown = subprocess.run(
        ["git", "show", f"{base}:{BASELINE_FILE}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if shown.returncode:
        raise MapError(f"{base} has no {BASELINE_FILE}")
    where = f"{base}:{BASELINE_FILE}"
    version = _document(shown.stdout, where).get("measurement_version")
    if version != MEASUREMENT_VERSION:
        print(f"loosened measurement_version {version} -> {MEASUREMENT_VERSION}")
        return 0
    before = decode(shown.stdout, where)
    after = decode((root / BASELINE_FILE).read_text(encoding="utf-8"), BASELINE_FILE)
    loosened_mode = before.mode == "block" and after.mode == "report"
    lines = ["loosened mode block -> report"] if loosened_mode else []
    for name, caps in sorted(before.slices.items()):
        if name not in after.slices:
            lines.append(f"loosened removed slice {name}")
            continue
        for key, cap in caps.items():
            if key not in after.slices[name]:
                lines.append(f"loosened removed {name} {key}")
            elif _number(raised := after.slices[name][key]) > _number(cap):
                lines.append(f"loosened raised {name} {key} {_show(cap)} -> {_show(raised)}")
    print("\n".join(lines) or "no loosening")
    return 0


COMMANDS: dict[str, Callable[[_Run], int]] = {
    "measure": _measure,
    "init": _init,
    "check": _check,
    "tighten": _tighten,
}


def run(root: Path, args: _Args) -> int:
    if args.command == "diff-baseline":
        return diff_baseline(root, args.base)
    slices = slice_map.load(root)
    graph = slice_map.build(root, slices)
    return COMMANDS[args.command](_Run(root, args, graph, measure(slices, graph)))


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(prog="check_depth", description="Ratchet slice depth.")
    commands = parser.add_subparsers(dest="command", required=True)
    _ = commands.add_parser("measure").add_argument("--json", action="store_true")
    _ = commands.add_parser("init").add_argument("--mode", choices=MODES, default="report")
    _ = commands.add_parser("check")
    tighten = commands.add_parser("tighten")
    _ = tighten.add_argument("--accept-denominator-gains", action="store_true")
    _ = tighten.add_argument("--adopt-new", action="store_true")
    _ = commands.add_parser("diff-baseline").add_argument("--base", required=True)
    try:
        return run(root, parser.parse_args(argv, namespace=_Args()))
    except MapError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
