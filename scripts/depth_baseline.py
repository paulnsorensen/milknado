"""Read and write `depth-ratchet.json`, the per-slice caps of the depth ratchet."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

from scripts import slice_map
from scripts.depth_metrics import Value
from scripts.slice_map import MapError

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
Caps = dict[str, dict[str, Value]]  # slice -> key -> cap


def show(value: Value) -> str:
    return f"{value[0]}/{value[1]}" if isinstance(value, tuple) else str(value)


def number(value: Value) -> Fraction:
    return Fraction(value[0], value[1] or 1) if isinstance(value, tuple) else Fraction(value)


def numerator(value: Value) -> int:
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


def read_document(text: str, where: str) -> dict[str, object]:
    try:
        document = cast(object, json.loads(text))
    except json.JSONDecodeError as error:
        raise MapError(f"{where}: {error}") from error
    fields = {"version", "measurement_version", "mode", "slices"}
    return slice_map.table(document, where, set(), fields)


def decode(text: str, where: str) -> Baseline:
    top = read_document(text, where)
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
        name: {key: show(caps[key]) if key in SHARES else caps[key] for key in KEYS if key in caps}
        for name, caps in sorted(baseline.slices.items())
    }
    document = {
        "version": 1,
        "measurement_version": MEASUREMENT_VERSION,
        "mode": baseline.mode,
        "slices": slices,
    }
    return json.dumps(document, indent=2) + "\n"
