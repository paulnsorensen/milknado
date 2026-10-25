"""The source tree, slice map, and git helper that the slice guard tests share."""

from __future__ import annotations

import subprocess
from pathlib import Path

SRC = "src/shop"
MAP = f"""version = 1

[policy]
language = "python"
unmapped_files = "error"
suppressions = "human-only"
roots = ["src"]

[[slice]]
name = "orders"
paths = ["{SRC}/orders"]
crust = ["{SRC}/orders/__init__.py"]
shape = "thin-facade"
may_depend_on = ["pricing"]
requires = ["pricing"]

[[slice]]
name = "pricing"
paths = ["{SRC}/pricing"]
crust = ["{SRC}/pricing/__init__.py"]
shape = "thin-facade"
may_depend_on = []

[[spine]]
name = "app"
paths = ["{SRC}/__init__.py", "{SRC}/_cli.py"]
may_depend_on = ["orders", "pricing"]
"""
TREE = {
    "__init__.py": "",
    "_cli.py": "from shop.orders import place\n\nprint(place)\n",
    "orders/__init__.py": "from shop.orders._place import place\n\n__all__ = ['place']\n",
    "orders/_place.py": "from shop.pricing import price\n\n\ndef place():\n    return price()\n",
    "pricing/__init__.py": "from shop.pricing._tiers import price\n\n__all__ = ['price']\n",
    "pricing/_tiers.py": "def price():\n    return 1\n",
}
BACK_EDGE = "from shop.orders import place\n\n\ndef price():\n    return 1\n"


def git(root: Path, *args: str) -> None:
    """Run git in `root` with a fixed identity and without commit signing."""
    _ = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )
