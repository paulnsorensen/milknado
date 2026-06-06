"""Hardening tests for the vendor-loop-engine seam.

WHY: the ralphify fork was vendored into milknado.loop; these tests lock
the four changed behaviours so any regression fails fast:

1. Crust contract — milknado.loop exports exactly the five promised names.
2. Seam rename — LoopAdapter / LoopPort are the public names; the old
   RalphifyAdapter / RalphPort names are not importable from public paths.
3. Dep removal — import ralphify raises ImportError (the git dep is gone).
4. Label change — cli_agents check prints "execution (loop)", not "ralphify".
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# 1. Crust contract
# ---------------------------------------------------------------------------


class TestLoopCrustContract:
    def test_all_exports_exact_set(self) -> None:
        """WHY: __all__ is the public contract; any drift (addition or removal)
        must require an explicit spec change, not happen silently."""
        import milknado.loop as loop_pkg

        assert set(loop_pkg.__all__) == {
            "EventType",
            "QueueEmitter",
            "RunConfig",
            "RunManager",
            "RunStatus",
        }

    def test_no_extra_public_names(self) -> None:
        """WHY: names not in __all__ should not be accidentally importable as
        part of the public API (e.g. leaked cli entrypoints or main)."""
        import milknado.loop as loop_pkg

        public_names = {n for n in dir(loop_pkg) if not n.startswith("_")}
        # Modules re-exported as subpackage attrs are allowed; only top-level
        # callable/class names that escaped __all__ are the concern.
        extra = public_names - set(loop_pkg.__all__)
        # Filter out submodule references and well-known dunder attrs that
        # Python itself adds to packages.
        submodule_noise = {"loop"}  # the package name itself, if present
        extra -= submodule_noise
        # Nothing should be left that isn't an imported submodule name.
        # Importable submodule names are acceptable (e.g. 'adapters', 'manager');
        # what's banned is cli entrypoints like 'main' or 'cli'.
        assert "main" not in extra, "'main' leaked into milknado.loop public namespace"
        assert "cli" not in extra, "'cli' leaked into milknado.loop public namespace"

    def test_attribution_docstring_cites_upstream(self) -> None:
        """WHY: MIT attribution is required; the docstring must cite the upstream
        repo so provenance is machine-readable."""
        import milknado.loop as loop_pkg

        doc = loop_pkg.__doc__ or ""
        assert "computerlovetech/ralphify" in doc, (
            "Attribution docstring must cite upstream: computerlovetech/ralphify"
        )

    def test_attribution_docstring_cites_fork(self) -> None:
        """WHY: the vendored code came from the paulnsorensen fork, not upstream
        directly; both must be cited so the diff is auditable."""
        import milknado.loop as loop_pkg

        doc = loop_pkg.__doc__ or ""
        assert "paulnsorensen/ralphify" in doc, (
            "Attribution docstring must cite fork: paulnsorensen/ralphify"
        )

    def test_five_names_importable(self) -> None:
        """WHY: the spec acceptance criterion is 'from milknado.loop import
        EventType, QueueEmitter, RunConfig, RunManager, RunStatus' — each name
        must resolve to a real object, not None."""
        from milknado.loop import EventType, QueueEmitter, RunConfig, RunManager, RunStatus

        for name, obj in [
            ("EventType", EventType),
            ("QueueEmitter", QueueEmitter),
            ("RunConfig", RunConfig),
            ("RunManager", RunManager),
            ("RunStatus", RunStatus),
        ]:
            assert obj is not None, f"milknado.loop.{name} resolved to None"


# ---------------------------------------------------------------------------
# 2. Seam rename
# ---------------------------------------------------------------------------


class TestSeamRename:
    def test_loop_adapter_importable_from_adapters(self) -> None:
        """WHY: milknado.adapters is the public adapter surface; LoopAdapter
        must be there so downstream code that does 'from milknado.adapters
        import LoopAdapter' keeps working."""
        from milknado.adapters import LoopAdapter

        assert LoopAdapter is not None

    def test_adapters_all_contains_loop_adapter(self) -> None:
        """WHY: __all__ is the machine-readable public contract; LoopAdapter must
        be listed so 'from milknado.adapters import *' pulls it in and linters
        do not flag it as unused."""
        import milknado.adapters as adapters_pkg

        assert "LoopAdapter" in adapters_pkg.__all__

    def test_adapters_all_does_not_contain_ralphify_adapter(self) -> None:
        """WHY: the rename is complete; any old name in __all__ would re-export
        a name that no longer exists and break callers."""
        import milknado.adapters as adapters_pkg

        assert "RalphifyAdapter" not in adapters_pkg.__all__

    def test_loop_port_importable_from_protocols(self) -> None:
        """WHY: LoopPort is the hexagonal boundary; domains reference it via
        protocols.py — it must be importable from there."""
        from milknado.domains.common.protocols import LoopPort

        assert LoopPort is not None

    def test_ralph_port_not_in_protocols(self) -> None:
        """WHY: the rename from RalphPort to LoopPort must be complete; any
        residual RalphPort attribute on the protocols module would mean the old
        name was silently re-added."""
        import milknado.domains.common.protocols as proto_mod

        assert not hasattr(proto_mod, "RalphPort"), (
            "RalphPort still present on protocols module — rename incomplete"
        )

    def test_loop_adapter_satisfies_loop_port(self) -> None:
        """WHY: LoopPort is the protocol boundary; LoopAdapter is the only
        implementation — they must be structurally compatible at import time."""
        from milknado.adapters.loop import LoopAdapter

        # Runtime isinstance check against a Protocol requires runtime_checkable.
        # We verify compatibility by asserting the adapter exposes every method
        # the port declares — the exhaustive set from protocols.py.
        port_methods = {
            "create_run",
            "start_run",
            "stop_run",
            "list_runs",
            "get_run",
            "get_run_stdout",
            "wait_for_next_completion",
            "poll_progress_events",
            "verify_spec",
            "generate_ralph_md",
        }
        missing = port_methods - set(dir(LoopAdapter))
        assert not missing, f"LoopAdapter missing LoopPort methods: {missing}"

    def test_ralphify_adapter_not_importable_from_adapters_module(self) -> None:
        """WHY: the old adapter name must not be re-exportable; an accidental
        re-add would be a silent regression."""
        import milknado.adapters as adapters_pkg

        assert not hasattr(adapters_pkg, "RalphifyAdapter"), (
            "RalphifyAdapter is accessible on milknado.adapters — it should have been removed"
        )


# ---------------------------------------------------------------------------
# 3. Dep removal — ralphify not importable
# ---------------------------------------------------------------------------


class TestRalphifyNotImportable:
    def test_import_ralphify_raises_import_error(self) -> None:
        """WHY: the git dependency was deleted; if ralphify were somehow
        re-installed (e.g. accidental `uv add`), this test would catch it
        immediately and prevent the vendored suite from silently testing the
        installed wheel instead of the vendored copy."""
        # Guard: if already imported in this session, remove from sys.modules
        # so we get a fresh import attempt.
        sys.modules.pop("ralphify", None)
        with pytest.raises(ImportError):
            importlib.import_module("ralphify")

    def test_ralphify_not_in_installed_packages(self) -> None:
        """WHY: the pyproject.toml dep was removed; this test verifies the lock
        file was regenerated and ralphify is absent from the environment."""
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as pkg_version

        with pytest.raises(PackageNotFoundError):
            pkg_version("ralphify")


# ---------------------------------------------------------------------------
# 4. Label change — "execution (loop)" in cli_agents check output
# ---------------------------------------------------------------------------


class TestCliAgentsLoopLabel:
    def test_agents_check_prints_loop_label(self, tmp_path: Path) -> None:
        """WHY: cli_agents.py line 43 was changed from 'ralphify' to 'loop';
        this test locks the output label so a rename rollback is caught."""
        from milknado.cli import app

        result = runner.invoke(app, ["agents", "check", "--project-root", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert "execution (loop)" in result.output

    def test_agents_check_does_not_print_ralphify_label(self, tmp_path: Path) -> None:
        """WHY: if 'ralphify' appears as the execution label, the rename
        regressed — this assertion catches it even if 'loop' is also present."""
        from milknado.cli import app

        result = runner.invoke(app, ["agents", "check", "--project-root", str(tmp_path)])

        assert result.exit_code == 0, result.output
        # The label must not reference the old dep name.
        assert "execution (ralphify)" not in result.output
