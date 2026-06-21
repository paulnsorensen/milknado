"""
Manifest-drift tests for the portable plugin packaging.

Asserts that all five JSON artifacts are valid, consistent with each other,
and pinned to the version declared in pyproject.toml.  Runs under the normal
pytest suite so `just check-llm` gates drift.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import yaml

REPO = Path(__file__).parent.parent

# All five JSON artifacts that must stay in sync
CLAUDE_MARKETPLACE = REPO / ".claude-plugin" / "marketplace.json"
CODEX_MARKETPLACE = REPO / ".agents" / "plugins" / "marketplace.json"
CLAUDE_PLUGIN = REPO / "plugins" / "milknado" / ".claude-plugin" / "plugin.json"
CODEX_PLUGIN = REPO / "plugins" / "milknado" / ".codex-plugin" / "plugin.json"
MCP_JSON = REPO / "plugins" / "milknado" / ".mcp.json"


def _pyproject_version() -> str:
    with open(REPO / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def _accepted_launcher_args(version: str) -> tuple[list[str], list[str]]:
    """The two coherent .mcp.json launcher forms, by channel.

    main carries the git-ref form (server resolved from git @main); tag / stable
    commits carry the PyPI pin `--from milknado==<version>` (version
    from pyproject.toml). Any other args list is channel drift. Shared by the live
    manifest assertion and the acceptance-discrimination tests so the contract has
    exactly one source of truth.
    """
    main_form = [
        "--from",
        "git+https://github.com/paulnsorensen/milknado@main",
        "milknado-mcp",
    ]
    pinned_form = ["--from", f"milknado=={version}", "milknado-mcp"]
    return main_form, pinned_form


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
        main_form, pinned_form = _accepted_launcher_args(_pyproject_version())
        assert server["args"] in (main_form, pinned_form), (
            "mcpServers.milknado.args must launch the server from the SAME ref as the "
            f"checkout: the git-ref form {main_form} on main, or the "
            f"PyPI pin {pinned_form} on a tag / stable commit (version read from "
            f"pyproject.toml, never hard-coded). Got {server['args']}"
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
    """Lock the release.yml shape: workflow_dispatch trigger, contents: write (to
    create the off-main pinned tag + repoint stable), the .mcp.json pin rewrite, and
    OIDC publish (no token secrets). The tag must carry the pin and stay off main, so
    CI — not a hand-pushed `v*` tag — creates it; the trigger is therefore
    workflow_dispatch, not push: tags."""

    RELEASE_YML = REPO / ".github" / "workflows" / "release.yml"

    def _workflow(self) -> dict:
        return yaml.safe_load(self.RELEASE_YML.read_text())

    def test_release_yml_exists(self) -> None:
        assert self.RELEASE_YML.exists(), f"Missing: {self.RELEASE_YML}"

    def test_release_yml_valid_yaml(self) -> None:
        wf = self._workflow()
        assert isinstance(wf, dict)

    def test_release_yml_triggers_on_workflow_dispatch(self) -> None:
        """Release is dispatched by hand, not by a pushed tag — the workflow creates
        the off-main pinned tag itself, so a push: tags trigger would re-fire it."""
        wf = self._workflow()
        # YAML-1.1 loaders (PyYAML) parse the 'on' key as boolean True;
        # YAML-1.2 loaders keep it as the string 'on'. Accept both.
        trigger = wf.get(True) or wf.get("on")
        assert trigger is not None, f"release.yml has no 'on' trigger block; keys: {list(wf)}"
        assert "workflow_dispatch" in trigger, (
            f"release.yml must trigger on workflow_dispatch; got {list(trigger)}"
        )
        assert "push" not in trigger, (
            "release.yml must NOT trigger on push: tags — the workflow creates and "
            "pushes the vX.Y.Z tag itself, which would re-fire a push: tags trigger"
        )

    def test_release_yml_grants_contents_write(self) -> None:
        """Pushing the tag and force-pushing stable both need contents: write."""
        wf = self._workflow()
        job = wf["jobs"]["build-and-publish"]
        assert job["permissions"]["contents"] == "write", (
            "build-and-publish must grant contents: write to push the vX.Y.Z tag and "
            f"repoint stable; got {job['permissions'].get('contents')!r}"
        )

    def test_release_yml_creates_pinned_tag_and_repoints_stable(self) -> None:
        """The release writes the PyPI pin into .mcp.json on an off-main commit, tags
        it vX.Y.Z, and force-points stable at that same commit."""
        raw = self.RELEASE_YML.read_text()
        assert "milknado==" in raw, (
            "release.yml must rewrite plugins/milknado/.mcp.json to the pinned "
            "`--from milknado==<version>` launcher on the tag commit"
        )
        assert "git tag" in raw, "release.yml must create the vX.Y.Z tag"
        assert "refs/heads/stable" in raw, (
            "release.yml must repoint the stable branch (refs/heads/stable) at the "
            "pinned tag commit"
        )

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

    def test_release_yml_publishes_via_pypa_action_in_pypi_environment(self) -> None:
        """Publish step uses pypa/gh-action-pypi-publish; job sets environment: pypi."""
        wf = self._workflow()
        job = wf["jobs"]["build-and-publish"]
        assert job.get("environment") == "pypi", (
            f"build-and-publish job must set 'environment: pypi'; got {job.get('environment')!r}"
        )
        uses_values = [step["uses"] for step in job["steps"] if "uses" in step]
        assert any("pypa/gh-action-pypi-publish" in u for u in uses_values), (
            f"No publish step uses pypa/gh-action-pypi-publish; found: {uses_values}"
        )


class TestWheelExcludesPluginPayload:
    """The pinned .mcp.json rewrite must not leak into the PyPI wheel.

    Structurally guaranteed: hatchling packages only src/milknado, so the plugin
    payload (plugins/milknado/.mcp.json) is never in the wheel. A wheel built from
    the pinned tag commit is therefore byte-identical in package contents to one
    built from main's bump commit. Lock the build target so a future include of
    plugins/ can't silently break that property.
    """

    def test_wheel_packages_only_src_milknado(self) -> None:
        with open(REPO / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        packages = data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
        assert packages == ["src/milknado"], (
            "Wheel must package only src/milknado so the plugin payload "
            f"(plugins/milknado/.mcp.json) never leaks into the wheel; got {packages}"
        )


class TestLauncherFormAcceptance:
    """The both-forms acceptance must discriminate, not rubber-stamp: accept the two
    coherent channel launchers, reject channel drift. The rewrite-consistency test
    exercises the tag / stable pinned-form branch a live `main` checkout never reaches."""

    def test_unpinned_legacy_form_is_rejected(self) -> None:
        # the pre-channels launcher floated the server independently of the ref;
        # it must no longer satisfy acceptance, or the channels coherence is a no-op
        legacy = ["--from", "milknado", "milknado-mcp"]
        assert legacy not in _accepted_launcher_args(_pyproject_version())

    def test_version_mismatched_pin_is_rejected(self) -> None:
        # proves the pin is keyed to pyproject, not a hard-coded constant
        wrong = ["--from", "milknado==0.0.0", "milknado-mcp"]
        assert wrong not in _accepted_launcher_args(_pyproject_version())

    def test_release_rewrite_yields_an_accepted_pinned_form(self) -> None:
        """Mirror the release workflow's `.mcp.json` rewrite and prove a tag / stable
        checkout is self-consistent: the rewritten launcher passes the same manifest
        acceptance. Locks workflow-output ↔ test-acceptance against drift (e.g. a pin
        target change on one side only). Mirrors, does not exec, release.yml's jq."""
        version = _pyproject_version()
        data = json.loads(MCP_JSON.read_text())
        data["mcpServers"]["milknado"]["args"] = [
            "--from",
            f"milknado=={version}",
            "milknado-mcp",
        ]
        assert data["mcpServers"]["milknado"]["args"] in _accepted_launcher_args(version)


class TestReadmeDocumentsChannels:
    """Spec acceptance: README documents the three channels + the uvx refresh caveat.
    Substring locks (version-independent) so a revert to single-channel install fails."""

    README = REPO / "README.md"

    def test_readme_names_all_three_channels(self) -> None:
        text = self.README.read_text()
        for ref in ("@stable", "@vX.Y.Z", "@main"):
            assert ref in text, f"README must document the {ref} channel"

    def test_readme_documents_main_git_launcher(self) -> None:
        text = self.README.read_text()
        assert "git+https://github.com/paulnsorensen/milknado@main" in text, (
            "README must show the @main git-ref server launcher"
        )

    def test_readme_documents_uvx_refresh_caveat(self) -> None:
        text = self.README.read_text()
        assert "--refresh" in text, (
            "README must document the uvx --refresh caveat for the @main git-ref server"
        )
