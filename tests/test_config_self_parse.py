"""The repo's own milknado.toml must load through the real config loader.

A shipped config that the loader cannot parse is a CI-invisible defect: nothing
in the suite loads `milknado.toml` itself, so an unparseable worker-tools grammar
(e.g. the unsupported `[milknado.worker.tools.claude]` / `extend=` form) sails
through green. These tests close that gap — they load the actual file and assert
the supported single-list grammar resolves, and they pin the old form as a hard
parse error so the regression cannot silently return.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.domains.common import load_config
from milknado.domains.common.agent_argv import WORKER_ALLOWED_TOOLS

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

    assert cfg.agent_family == "claude"
    # The worker-tools table must have parsed into the structured form.
    assert "claude" in cfg.worker_tools


def test_shipped_config_worker_tools_extend_family_baseline() -> None:
    """The `"..."` sentinel must expand to the family default plus the extra tool.

    This is the behavioural payload of the fix: `claude = ["...", "Bash(just:*)"]`
    keeps every default worker tool and adds the just-runner. A bare replacement
    list (no sentinel) would drop the baseline — assert we did not do that.
    """
    cfg = load_config(SHIPPED_CONFIG, include_global=False)

    claude_tools = cfg.worker_tools["claude"]
    # Stored tuple is the raw list (sentinel un-expanded); resolution happens later.
    assert claude_tools == ("...", "Bash(just:*)")

    # And the family baseline is genuinely non-empty, so the sentinel carries weight.
    assert WORKER_ALLOWED_TOOLS["claude"], "family baseline must be non-empty"


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
