"""Slice map: each invalid `slices.toml` is an input error."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import slice_map
from scripts.slice_map import MapError
from tests.slice_fixture import MAP, SRC, TREE


@pytest.mark.parametrize(
    "broken",
    [
        MAP.replace(f'paths = ["{SRC}/pricing"]', f'paths = ["{SRC}/pri*"]'),
        MAP.replace(
            f'paths = ["{SRC}/pricing"]', f'paths = ["{SRC}/pricing", "{SRC}/orders/_place.py"]'
        ),
        MAP.replace('shape = "thin-facade"', 'shape = "thin-facade"\nowner = "x"', 1),
        MAP.replace(f'crust = ["{SRC}/pricing/__init__.py"]', f'crust = ["{SRC}/_cli.py"]'),
        MAP + '\n[[suppress]]\nrule = "SB-00"\nfrom = "a"\nto = "b"\nreason = "r"\n',
        MAP + '\n[[suppress]]\nrule = "SB-04"\nfrom = "a"\nto = "b"\nreason = ""\n',
        MAP.replace('requires = ["pricing"]', 'requires = ["app"]'),
        MAP.replace("may_depend_on = []", 'may_depend_on = ["pricing"]'),
        MAP.replace('language = "python"', 'language = "typescript"'),
        MAP.replace("version = 1", "version = 2"),
        MAP + '\n[[spine]]\nname = "app"\npaths = ["src"]\nmay_depend_on = []\n',
        "version = ",
    ],
    ids=[
        "glob in paths",
        "overlap",
        "unknown field",
        "crust outside",
        "SB-00 suppression",
        "empty reason",
        "requires outside may_depend_on",
        "depends on itself",
        "wrong language",
        "wrong version",
        "duplicate name",
        "invalid toml",
    ],
)
def test_invalid_maps_are_input_errors(tmp_path: Path, broken: str) -> None:
    for name, text in TREE.items():
        (tmp_path / SRC / name).parent.mkdir(parents=True, exist_ok=True)
        _ = (tmp_path / SRC / name).write_text(text, encoding="utf-8")
    assert slice_map.parse(MAP, tmp_path).nodes
    with pytest.raises(MapError):
        _ = slice_map.parse(broken, tmp_path)
