"""
Manifest-drift tests for the portable plugin packaging.

Asserts that all four JSON artifacts are valid, consistent with each other,
and pinned to the version declared in pyproject.toml.  Runs under the normal
pytest suite so `just check-llm` gates drift.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

REPO = Path(__file__).parent.parent

# All four JSON artifacts that must stay in sync
CLAUDE_MARKETPLACE = REPO / ".claude-plugin" / "marketplace.json"
CODEX_MARKETPLACE = REPO / ".agents" / "plugins" / "marketplace.json"
CLAUDE_PLUGIN = REPO / "plugins" / "milknado" / ".claude-plugin" / "plugin.json"
CODEX_PLUGIN = REPO / "plugins" / "milknado" / ".codex-plugin" / "plugin.json"
MCP_JSON = REPO / "plugins" / "milknado" / ".mcp.json"


def _pyproject_version() -> str:
    with open(REPO / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


class TestManifestsExist:
    def test_claude_marketplace_exists(self) -> None:
        assert CLAUDE_MARKETPLACE.exists(), f"Missing: {CLAUDE_MARKETPLACE}"

    def test_codex_marketplace_exists(self) -> None:
        assert CODEX_MARKETPLACE.exists(), f"Missing: {CODEX_MARKETPLACE}"

    def test_claude_plugin_json_exists(self) -> None:
        assert CLAUDE_PLUGIN.exists(), f"Missing: {CLAUDE_PLUGIN}"

    def test_codex_plugin_json_exists(self) -> None:
        assert CODEX_PLUGIN.exists(), f"Missing: {CODEX_PLUGIN}"

    def test_mcp_json_exists(self) -> None:
        assert MCP_JSON.exists(), f"Missing: {MCP_JSON}"


class TestManifestsParse:
    def test_claude_marketplace_valid_json(self) -> None:
        data = json.loads(CLAUDE_MARKETPLACE.read_text())
        assert isinstance(data, dict)

    def test_codex_marketplace_valid_json(self) -> None:
        data = json.loads(CODEX_MARKETPLACE.read_text())
        assert isinstance(data, dict)

    def test_claude_plugin_valid_json(self) -> None:
        data = json.loads(CLAUDE_PLUGIN.read_text())
        assert isinstance(data, dict)

    def test_codex_plugin_valid_json(self) -> None:
        data = json.loads(CODEX_PLUGIN.read_text())
        assert isinstance(data, dict)

    def test_mcp_json_valid_json(self) -> None:
        data = json.loads(MCP_JSON.read_text())
        assert isinstance(data, dict)


class TestVersionAgreement:
    def test_claude_plugin_version_matches_pyproject(self) -> None:
        expected = _pyproject_version()
        data = json.loads(CLAUDE_PLUGIN.read_text())
        assert data["version"] == expected, (
            f"plugins/milknado/.claude-plugin/plugin.json version {data['version']!r} "
            f"!= pyproject.toml version {expected!r}"
        )

    def test_codex_plugin_version_matches_pyproject(self) -> None:
        expected = _pyproject_version()
        data = json.loads(CODEX_PLUGIN.read_text())
        assert data["version"] == expected, (
            f"plugins/milknado/.codex-plugin/plugin.json version {data['version']!r} "
            f"!= pyproject.toml version {expected!r}"
        )


class TestMarketplaceSourcePaths:
    def test_claude_marketplace_source_points_at_plugins_milknado(self) -> None:
        data = json.loads(CLAUDE_MARKETPLACE.read_text())
        plugins = data["plugins"]
        assert len(plugins) >= 1, "Claude marketplace must have at least one plugin entry"
        sources = [p["source"] for p in plugins]
        assert "./plugins/milknado" in sources, (
            f"Claude marketplace plugins[].source must include './plugins/milknado'; got {sources}"
        )

    def test_codex_marketplace_source_points_at_plugins_milknado(self) -> None:
        data = json.loads(CODEX_MARKETPLACE.read_text())
        plugins = data["plugins"]
        assert len(plugins) >= 1, "Codex marketplace must have at least one plugin entry"
        paths = [p["source"]["path"] for p in plugins]
        assert "./plugins/milknado" in paths, (
            "Codex marketplace plugins[].source.path must include"
            f" './plugins/milknado'; got {paths}"
        )


class TestPluginNameAgreement:
    """Plugin name must be 'milknado' in all manifests that carry a name field."""

    EXPECTED_NAME = "milknado"

    def test_claude_plugin_name(self) -> None:
        data = json.loads(CLAUDE_PLUGIN.read_text())
        assert data["name"] == self.EXPECTED_NAME, (
            f"Claude plugin.json name={data['name']!r}, expected {self.EXPECTED_NAME!r}"
        )

    def test_codex_plugin_name(self) -> None:
        data = json.loads(CODEX_PLUGIN.read_text())
        assert data["name"] == self.EXPECTED_NAME, (
            f"Codex plugin.json name={data['name']!r}, expected {self.EXPECTED_NAME!r}"
        )

    def test_claude_marketplace_name(self) -> None:
        data = json.loads(CLAUDE_MARKETPLACE.read_text())
        assert data["name"] == self.EXPECTED_NAME, (
            f"Claude marketplace name={data['name']!r}, expected {self.EXPECTED_NAME!r}"
        )


class TestMcpJsonShape:
    def test_mcp_json_has_milknado_server(self) -> None:
        data = json.loads(MCP_JSON.read_text())
        assert "mcpServers" in data
        assert "milknado" in data["mcpServers"], (
            f"mcpServers must have a 'milknado' key; got {list(data['mcpServers'].keys())}"
        )

    def test_mcp_json_uses_uvx(self) -> None:
        data = json.loads(MCP_JSON.read_text())
        server = data["mcpServers"]["milknado"]
        assert server["command"] == "uvx", (
            f"mcpServers.milknado.command must be 'uvx'; got {server['command']!r}"
        )
        assert "milknado-mcp" in server["args"], (
            f"mcpServers.milknado.args must include 'milknado-mcp'; got {server['args']}"
        )
