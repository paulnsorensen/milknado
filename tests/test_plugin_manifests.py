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

import yaml

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


class TestSkillFilesExist:
    """Auto-discovery requires the skill files to be present in the payload."""

    SKILL_DIR = REPO / "plugins" / "milknado" / "skills" / "milknado-config"

    def test_skill_md_exists(self) -> None:
        skill_md = self.SKILL_DIR / "SKILL.md"
        assert skill_md.exists(), f"Missing skill file: {skill_md}"

    def test_flavor_presets_exists(self) -> None:
        preset = self.SKILL_DIR / "references" / "flavor-presets.md"
        assert preset.exists(), f"Missing skill reference: {preset}"


class TestPluginJsonNoSkillsField:
    """plugin.json must NOT have a 'skills' field.

    The spec locked this: default skills/ auto-discovery is used.  A 'skills'
    field with a '..' path would fail validation; locking absence prevents a
    future accidental addition from silently breaking install.
    """

    def test_claude_plugin_no_skills_field(self) -> None:
        data = json.loads(CLAUDE_PLUGIN.read_text())
        assert "skills" not in data, (
            "Claude plugin.json must not have a 'skills' field; "
            "default skills/ auto-discovery is used (spec decision)"
        )

    def test_codex_plugin_no_skills_field(self) -> None:
        data = json.loads(CODEX_PLUGIN.read_text())
        assert "skills" not in data, (
            "Codex plugin.json must not have a 'skills' field; "
            "default skills/ auto-discovery is used (spec decision)"
        )


class TestCodexMarketplaceSchema:
    """Lock the Codex-native marketplace shape: source.source, policy, interface.displayName."""

    def _entry(self) -> dict:
        data = json.loads(CODEX_MARKETPLACE.read_text())
        return data["plugins"][0]

    def test_codex_source_type_is_local(self) -> None:
        entry = self._entry()
        assert entry["source"]["source"] == "local", (
            f"Codex marketplace source.source must be 'local'; got {entry['source']['source']!r}"
        )

    def test_codex_policy_installation(self) -> None:
        entry = self._entry()
        assert entry["policy"]["installation"] == "AVAILABLE", (
            f"Codex marketplace policy.installation must be 'AVAILABLE'; "
            f"got {entry['policy']['installation']!r}"
        )

    def test_codex_policy_authentication(self) -> None:
        entry = self._entry()
        assert entry["policy"]["authentication"] == "ON_INSTALL", (
            f"Codex marketplace policy.authentication must be 'ON_INSTALL'; "
            f"got {entry['policy']['authentication']!r}"
        )

    def test_codex_interface_display_name(self) -> None:
        data = json.loads(CODEX_MARKETPLACE.read_text())
        assert data["interface"]["displayName"] == "milknado", (
            f"Codex marketplace interface.displayName must be 'milknado'; "
            f"got {data['interface']['displayName']!r}"
        )


class TestReleaseWorkflow:
    """Lock the release.yml shape: valid YAML, v* tag trigger, OIDC (no token secrets)."""

    RELEASE_YML = REPO / ".github" / "workflows" / "release.yml"

    def _workflow(self) -> dict:
        return yaml.safe_load(self.RELEASE_YML.read_text())

    def test_release_yml_exists(self) -> None:
        assert self.RELEASE_YML.exists(), f"Missing: {self.RELEASE_YML}"

    def test_release_yml_valid_yaml(self) -> None:
        wf = self._workflow()
        assert isinstance(wf, dict)

    def test_release_yml_triggers_on_version_tags(self) -> None:
        wf = self._workflow()
        # PyYAML parses the 'on' YAML key as boolean True
        tags = wf[True]["push"]["tags"]
        assert "v*" in tags, f"release.yml must trigger on 'v*' tags; got {tags}"

    def test_release_yml_uses_oidc_not_token_secret(self) -> None:
        """Trusted publishing uses id-token: write; no PASSWORD or API_TOKEN env var."""
        raw = self.RELEASE_YML.read_text()
        assert "id-token: write" in raw, (
            "release.yml must grant id-token: write for OIDC trusted publishing"
        )
        for forbidden in ("PASSWORD", "API_TOKEN", "PYPI_TOKEN", "TWINE_PASSWORD"):
            assert forbidden not in raw, (
                f"release.yml must not contain secret token '{forbidden}'; "
                "use OIDC trusted publishing instead"
            )
