"""The repo's own milknado.toml must load through the real config loader.

A shipped config that the loader cannot parse can sail through green because no
test in the suite loads `milknado.toml` itself. These tests close that gap by
loading the actual file and asserting its supported codex/OMP configuration
contract. Separate fixture tests below keep the supported single-list grammar
and the rejected legacy form covered independently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.domains.common import load_config

# tests/ lives directly under the repo root; the shipped config sits beside it.
REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_CONFIG = REPO_ROOT / "milknado.toml"


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_shipped_config_file_exists() -> None:
    assert SHIPPED_CONFIG.is_file(), f"expected shipped config at {SHIPPED_CONFIG}"


def test_shipped_config_parses_through_real_loader() -> None:
    # include_global=False keeps this reproducible regardless of the host's
    # ~/.config/milknado/milknado.toml — we are asserting about the repo file only.
    cfg = load_config(SHIPPED_CONFIG, include_global=False)

    assert cfg.agent_family == "codex"
    assert cfg.worker_tools == {}
    assert cfg.worktree is True


def test_shipped_config_preserves_explicit_codex_execution_agent() -> None:
    """The shipped codex config uses the supported codex worker command."""
    cfg = load_config(SHIPPED_CONFIG, include_global=False)

    assert cfg.execution_agent == "codex"


def test_old_extend_grammar_fails_to_parse(tmp_path: Path) -> None:
    """The unsupported `extend=` form must raise — this is what shipped broken.

    Guards against a revert to the table-with-`extend` grammar the loader never
    supported. If this stops raising, the loader silently grew `extend=` support
    (out of scope, YAGNI) or the test is no longer exercising the parse path.
    """
    bad = _write(
        tmp_path / "milknado.toml",
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.worker.tools.claude]\n"
        'extend = ["Bash(just:*)"]\n',
    )
    # Match the full defect signature, not just the context prefix: the
    # `extend=` table collapses the family value to a dict, and the loader
    # rejects it with "...must be a list of strings, got dict". Pinning "dict"
    # ties the red to the exact regression rather than any error mentioning the
    # table.
    with pytest.raises(
        ValueError,
        match=r"\[milknado\.worker\.tools\.claude\] must be a list of strings, got dict",
    ):
        load_config(bad, include_global=False)


def test_new_single_list_grammar_parses(tmp_path: Path) -> None:
    """The supported single-list form parses where the old one failed."""
    good = _write(
        tmp_path / "milknado.toml",
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.worker.tools]\n"
        'claude = ["...", "Bash(just:*)"]\n',
    )
    cfg = load_config(good, include_global=False)
    assert cfg.worker_tools["claude"] == ("...", "Bash(just:*)")
