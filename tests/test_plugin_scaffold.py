from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import pytest

from milknado.plugins.loader import load_plugins
from milknado.plugins.scaffold import scaffold_plugin


class TestScaffoldPlugin:
    def test_creates_directory_structure(self, tmp_path: Path) -> None:
        result = scaffold_plugin("my_plugin", tmp_path)
        assert result.plugin_dir == tmp_path / "my_plugin"
        assert result.plugin_dir.is_dir()
        assert (result.plugin_dir / "__init__.py").exists()
        assert (result.plugin_dir / "plugin.py").exists()
        assert (result.plugin_dir / "README.md").exists()

    def test_files_created_list(self, tmp_path: Path) -> None:
        result = scaffold_plugin("test_plug", tmp_path)
        assert set(result.files_created) == {"__init__.py", "plugin.py", "README.md"}

    def test_plugin_class_name_from_snake_case(self, tmp_path: Path) -> None:
        scaffold_plugin("my_cool_plugin", tmp_path)
        content = (tmp_path / "my_cool_plugin" / "plugin.py").read_text()
        assert "class MyCoolPlugin:" in content

    def test_plugin_class_name_from_kebab_case(self, tmp_path: Path) -> None:
        scaffold_plugin("my-cool-plugin", tmp_path)
        content = (tmp_path / "my-cool-plugin" / "plugin.py").read_text()
        assert "class MyCoolPlugin:" in content

    def test_plugin_has_hook_method(self, tmp_path: Path) -> None:
        scaffold_plugin("demo", tmp_path)
        content = (tmp_path / "demo" / "plugin.py").read_text()
        assert "def on_node_status_change" in content
        assert "PluginHook" in content

    def test_init_imports_plugin_class(self, tmp_path: Path) -> None:
        scaffold_plugin("demo", tmp_path)
        content = (tmp_path / "demo" / "__init__.py").read_text()
        assert "from demo.plugin import Demo" in content

    def test_readme_contains_name(self, tmp_path: Path) -> None:
        scaffold_plugin("analytics", tmp_path)
        content = (tmp_path / "analytics" / "README.md").read_text()
        assert "# analytics" in content
        assert 'plugins = ["analytics"]' in content

    def test_raises_if_directory_exists(self, tmp_path: Path) -> None:
        (tmp_path / "existing").mkdir()
        with pytest.raises(FileExistsError, match="already exists"):
            scaffold_plugin("existing", tmp_path)

    def test_generated_plugin_is_importable(self, tmp_path: Path) -> None:
        scaffold_plugin("importable_plug", tmp_path)
        sys.path.insert(0, str(tmp_path))
        try:
            mod = importlib.import_module("importable_plug")
            cls = getattr(mod, "ImportablePlug")
            instance = cls()
            assert instance.meta.name == "importable_plug"
            assert instance.meta.version == "0.1.0"
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("importable_plug", None)
            sys.modules.pop("importable_plug.plugin", None)


class TestLoadPlugins:
    def _make_plugin_module(self, tmp_path: Path, name: str) -> None:
        scaffold_plugin(name, tmp_path)
        if str(tmp_path) not in sys.path:
            sys.path.insert(0, str(tmp_path))

    def _cleanup_modules(self, *names: str) -> None:
        for name in names:
            sys.modules.pop(name, None)
            sys.modules.pop(f"{name}.plugin", None)

    def test_loads_valid_plugin(self, tmp_path: Path) -> None:
        self._make_plugin_module(tmp_path, "valid_plug")
        try:
            plugins = load_plugins(("valid_plug",))
            assert len(plugins) == 1
            assert plugins[0].meta.name == "valid_plug"
        finally:
            sys.path.remove(str(tmp_path))
            self._cleanup_modules("valid_plug")

    def test_skips_missing_plugin(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="milknado.plugins"):
            plugins = load_plugins(("nonexistent_plugin_xyz",))
        assert len(plugins) == 0
        assert "Could not import" in caplog.text

    def test_loads_multiple_plugins(self, tmp_path: Path) -> None:
        self._make_plugin_module(tmp_path, "plug_a")
        self._make_plugin_module(tmp_path, "plug_b")
        try:
            plugins = load_plugins(("plug_a", "plug_b"))
            assert len(plugins) == 2
            names = {p.meta.name for p in plugins}
            assert names == {"plug_a", "plug_b"}
        finally:
            sys.path.remove(str(tmp_path))
            self._cleanup_modules("plug_a", "plug_b")

    def test_skips_module_without_plugin_class(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pkg = tmp_path / "empty_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("X = 1\n")
        sys.path.insert(0, str(tmp_path))
        try:
            with caplog.at_level(logging.WARNING, logger="milknado.plugins"):
                plugins = load_plugins(("empty_pkg",))
            assert len(plugins) == 0
            assert "No plugin class found" in caplog.text
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("empty_pkg", None)

    def test_empty_plugins_tuple(self) -> None:
        assert load_plugins(()) == []

    def test_plugin_hook_fires(self, tmp_path: Path) -> None:
        self._make_plugin_module(tmp_path, "hook_test")
        try:
            plugins = load_plugins(("hook_test",))
            from milknado.domains.common import MikadoNode, NodeStatus

            node = MikadoNode(id=1, description="test")
            plugins[0].on_node_status_change(node, NodeStatus.PENDING, NodeStatus.RUNNING)
        finally:
            sys.path.remove(str(tmp_path))
            self._cleanup_modules("hook_test")

    def test_logs_loaded_plugin(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        self._make_plugin_module(tmp_path, "log_plug")
        try:
            with caplog.at_level(logging.INFO, logger="milknado.plugins"):
                load_plugins(("log_plug",))
            assert "Loaded plugin: log_plug v0.1.0" in caplog.text
        finally:
            sys.path.remove(str(tmp_path))
            self._cleanup_modules("log_plug")


class TestPluginInitCli:
    def test_plugin_init_creates_directory(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from milknado.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["plugin", "init", "my_plug", "-d", str(tmp_path)])
        assert result.exit_code == 0
        assert "Created plugin" in result.stdout
        assert (tmp_path / "my_plug" / "plugin.py").exists()

    def test_plugin_init_fails_if_exists(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from milknado.cli import app

        (tmp_path / "exists_plug").mkdir()
        runner = CliRunner()
        result = runner.invoke(app, ["plugin", "init", "exists_plug", "-d", str(tmp_path)])
        assert result.exit_code == 1
        assert "already exists" in result.stdout

    def test_plugin_init_lists_files(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from milknado.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["plugin", "init", "list_test", "-d", str(tmp_path)])
        assert result.exit_code == 0
        assert "__init__.py" in result.stdout
        assert "plugin.py" in result.stdout
        assert "README.md" in result.stdout
