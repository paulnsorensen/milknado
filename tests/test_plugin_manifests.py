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
from typing import cast

import yaml

JsonObject = dict[str, object]
YamlObject = dict[object, object]


def _read_json(path: Path) -> JsonObject:
    return cast(JsonObject, json.loads(path.read_text()))


def _read_toml(path: Path) -> JsonObject:
    with path.open("rb") as file:
        return cast(JsonObject, tomllib.load(file))


def _read_yaml(path: Path) -> YamlObject:
    return cast(YamlObject, yaml.safe_load(path.read_text()))


def _mapping(value: object) -> JsonObject:
    return cast(JsonObject, value)


def _objects(value: object) -> list[JsonObject]:
    return cast(list[JsonObject], value)


def _strings(value: object) -> list[str]:
    return cast(list[str], value)


def _yaml_mapping(value: object) -> YamlObject:
    return cast(YamlObject, value)


REPO = Path(__file__).parent.parent

# All five JSON artifacts that must stay in sync
CLAUDE_MARKETPLACE = REPO / ".claude-plugin" / "marketplace.json"
CODEX_MARKETPLACE = REPO / ".agents" / "plugins" / "marketplace.json"
CLAUDE_PLUGIN = REPO / "plugins" / "milknado" / ".claude-plugin" / "plugin.json"
CODEX_PLUGIN = REPO / "plugins" / "milknado" / ".codex-plugin" / "plugin.json"
MCP_JSON = REPO / "plugins" / "milknado" / ".mcp.json"


def _pyproject_version() -> str:
    data = _read_toml(REPO / "pyproject.toml")
    project = cast(JsonObject, data["project"])
    return cast(str, project["version"])


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
        data = _read_json(CLAUDE_MARKETPLACE)
        assert isinstance(data, dict)

    def test_codex_marketplace_valid_json(self) -> None:
        data = _read_json(CODEX_MARKETPLACE)
        assert isinstance(data, dict)

    def test_claude_plugin_valid_json(self) -> None:
        data = _read_json(CLAUDE_PLUGIN)
        assert isinstance(data, dict)

    def test_codex_plugin_valid_json(self) -> None:
        data = _read_json(CODEX_PLUGIN)
        assert isinstance(data, dict)

    def test_mcp_json_valid_json(self) -> None:
        data = _read_json(MCP_JSON)
        assert isinstance(data, dict)


class TestVersionAgreement:
    def test_claude_plugin_version_matches_pyproject(self) -> None:
        expected = _pyproject_version()
        data = _read_json(CLAUDE_PLUGIN)
        assert data["version"] == expected, (
            f"plugins/milknado/.claude-plugin/plugin.json version {data['version']!r} "
            f"!= pyproject.toml version {expected!r}"
        )

    def test_codex_plugin_version_matches_pyproject(self) -> None:
        expected = _pyproject_version()
        data = _read_json(CODEX_PLUGIN)
        assert data["version"] == expected, (
            f"plugins/milknado/.codex-plugin/plugin.json version {data['version']!r} "
            f"!= pyproject.toml version {expected!r}"
        )


class TestMarketplaceSourcePaths:
    def test_claude_marketplace_source_points_at_plugins_milknado(self) -> None:
        data = _read_json(CLAUDE_MARKETPLACE)
        plugins = _objects(data["plugins"])
        assert len(plugins) >= 1, "Claude marketplace must have at least one plugin entry"
        sources = [p["source"] for p in plugins]
        assert "./plugins/milknado" in sources, (
            f"Claude marketplace plugins[].source must include './plugins/milknado'; got {sources}"
        )

    def test_codex_marketplace_source_points_at_plugins_milknado(self) -> None:
        data = _read_json(CODEX_MARKETPLACE)
        plugins = _objects(data["plugins"])
        assert len(plugins) >= 1, "Codex marketplace must have at least one plugin entry"
        paths = [_mapping(p["source"])["path"] for p in plugins]
        assert "./plugins/milknado" in paths, (
            "Codex marketplace plugins[].source.path must include"
            f" './plugins/milknado'; got {paths}"
        )


class TestPluginNameAgreement:
    """Plugin name must be 'milknado' in all manifests that carry a name field."""

    EXPECTED_NAME: str = "milknado"

    def test_claude_plugin_name(self) -> None:
        data = _read_json(CLAUDE_PLUGIN)
        assert data["name"] == self.EXPECTED_NAME, (
            f"Claude plugin.json name={data['name']!r}, expected {self.EXPECTED_NAME!r}"
        )

    def test_codex_plugin_name(self) -> None:
        data = _read_json(CODEX_PLUGIN)
        assert data["name"] == self.EXPECTED_NAME, (
            f"Codex plugin.json name={data['name']!r}, expected {self.EXPECTED_NAME!r}"
        )

    def test_claude_marketplace_name(self) -> None:
        data = _read_json(CLAUDE_MARKETPLACE)
        assert data["name"] == self.EXPECTED_NAME, (
            f"Claude marketplace name={data['name']!r}, expected {self.EXPECTED_NAME!r}"
        )


class TestMcpJsonShape:
    def test_mcp_json_has_milknado_server(self) -> None:
        data = _read_json(MCP_JSON)
        assert "mcpServers" in data
        servers = _mapping(data["mcpServers"])
        assert "milknado" in servers, f"mcpServers must have a 'milknado' key; got {list(servers)}"

    def test_mcp_json_uses_uvx(self) -> None:
        data = _read_json(MCP_JSON)
        server = _mapping(_mapping(data["mcpServers"])["milknado"])
        assert server["command"] == "uvx", (
            f"mcpServers.milknado.command must be 'uvx'; got {server['command']!r}"
        )
        main_form, pinned_form = _accepted_launcher_args(_pyproject_version())
        assert _strings(server["args"]) in (main_form, pinned_form), (
            "mcpServers.milknado.args must launch the server from the SAME ref as the "
            f"checkout: the git-ref form {main_form} on main, or the "
            f"PyPI pin {pinned_form} on a tag / stable commit (version read from "
            f"pyproject.toml, never hard-coded). Got {server['args']}"
        )


class TestSkillFilesExist:
    """Auto-discovery requires the skill files to be present in the payload."""

    SKILL_DIR: Path = REPO / "plugins" / "milknado" / "skills" / "milknado-config"

    def test_skill_md_exists(self) -> None:
        skill_md = self.SKILL_DIR / "SKILL.md"
        assert skill_md.exists(), f"Missing skill file: {skill_md}"

    def test_flavor_presets_exists(self) -> None:
        preset = self.SKILL_DIR / "references" / "flavor-presets.md"
        assert preset.exists(), f"Missing skill reference: {preset}"

    def test_flavor_docs_use_string_registry(self) -> None:
        docs = (
            (self.SKILL_DIR / "SKILL.md").read_text(),
            (self.SKILL_DIR / "references" / "flavor-presets.md").read_text(),
        )
        assert all("TaskFlavor" not in text for text in docs)
        assert "BUILTIN_FLAVORS" in docs[0]
        assert "worktree" in docs[1]


class TestPluginJsonNoSkillsField:
    """plugin.json must NOT have a 'skills' field.

    The spec locked this: default skills/ auto-discovery is used.  A 'skills'
    field with a '..' path would fail validation; locking absence prevents a
    future accidental addition from silently breaking install.
    """

    def test_claude_plugin_no_skills_field(self) -> None:
        data = _read_json(CLAUDE_PLUGIN)
        assert "skills" not in data, (
            "Claude plugin.json must not have a 'skills' field; "
            "default skills/ auto-discovery is used (spec decision)"
        )

    def test_codex_plugin_no_skills_field(self) -> None:
        data = _read_json(CODEX_PLUGIN)
        assert "skills" not in data, (
            "Codex plugin.json must not have a 'skills' field; "
            "default skills/ auto-discovery is used (spec decision)"
        )


class TestCodexMarketplaceSchema:
    """Lock the Codex-native marketplace shape: source.source, policy, interface.displayName."""

    def _entry(self) -> JsonObject:
        data = _read_json(CODEX_MARKETPLACE)
        plugins = cast(list[JsonObject], data["plugins"])
        return plugins[0]

    def test_codex_source_type_is_local(self) -> None:
        entry = self._entry()
        source = _mapping(entry["source"])
        assert source["source"] == "local", (
            f"Codex marketplace source.source must be 'local'; got {source['source']!r}"
        )

    def test_codex_policy_installation(self) -> None:
        entry = self._entry()
        policy = _mapping(entry["policy"])
        assert policy["installation"] == "AVAILABLE", (
            f"Codex marketplace policy.installation must be 'AVAILABLE'; "
            f"got {policy['installation']!r}"
        )

    def test_codex_policy_authentication(self) -> None:
        entry = self._entry()
        policy = _mapping(entry["policy"])
        assert policy["authentication"] == "ON_INSTALL", (
            f"Codex marketplace policy.authentication must be 'ON_INSTALL'; "
            f"got {policy['authentication']!r}"
        )

    def test_codex_interface_display_name(self) -> None:
        data = _read_json(CODEX_MARKETPLACE)
        interface = cast(JsonObject, data["interface"])
        assert interface["displayName"] == "milknado", (
            f"Codex marketplace interface.displayName must be 'milknado'; "
            f"got {interface['displayName']!r}"
        )


class TestReleaseWorkflow:
    """Lock the release.yml shape: fires on every push to main plus workflow_dispatch,
    but publishes only when the version in pyproject.toml is not yet on PyPI.  A
    non-bump push runs a fast detect job that no-ops (build-and-publish is skipped).
    workflow_dispatch is retained as a manual escape hatch.  contents: write lives on
    build-and-publish only (least-privilege split); the detect job runs with
    contents: read."""

    RELEASE_YML: Path = REPO / ".github" / "workflows" / "release.yml"

    def _workflow(self) -> YamlObject:
        return _read_yaml(self.RELEASE_YML)

    def test_release_yml_exists(self) -> None:
        assert self.RELEASE_YML.exists(), f"Missing: {self.RELEASE_YML}"

    def test_release_yml_valid_yaml(self) -> None:
        wf = self._workflow()
        assert isinstance(wf, dict)

    def test_release_yml_triggers_on_main_push_and_dispatch(self) -> None:
        """Release fires on every push to main (version-gate in detect skips non-bumps)
        plus workflow_dispatch as a manual escape hatch.  No push: tags trigger — the
        vX.Y.Z tag is an output of the job (off-main commit), not an input."""
        workflow = self._workflow()
        trigger_value = workflow.get(True) or workflow.get("on")
        assert trigger_value is not None, (
            f"release.yml has no 'on' trigger block; keys: {list(workflow)}"
        )
        trigger = _yaml_mapping(trigger_value)
        assert "workflow_dispatch" in trigger, (
            f"release.yml must trigger on workflow_dispatch; got {list(trigger)}"
        )
        assert "push" in trigger, f"release.yml must trigger on push to main; got {list(trigger)}"
        push = _yaml_mapping(trigger["push"])
        assert _strings(push["branches"]) == ["main"], (
            f"release.yml push trigger must be branches: [main]; got {push['branches']}"
        )
        assert "tags" not in push, (
            "release.yml must NOT trigger on push: tags — the vX.Y.Z tag is an output "
            "of the job (off-main commit), not an input trigger"
        )

    def test_release_yml_version_gate_jobs(self) -> None:
        """detect job outputs release=true/false; build-and-publish gates on it.

        Locks the two-job version-gate shape so a future refactor can't silently
        publish on every push to main.
        """
        workflow = self._workflow()
        jobs = _yaml_mapping(workflow["jobs"])
        assert "detect" in jobs, f"release.yml must have a 'detect' job; got {list(jobs)}"
        detect = _yaml_mapping(jobs["detect"])
        detect_outputs = _yaml_mapping(detect.get("outputs", {}))
        assert "release" in detect_outputs, (
            f"detect job must declare a 'release' output; got {list(detect_outputs)}"
        )
        pub = _yaml_mapping(jobs["build-and-publish"])
        needs = pub.get("needs", [])
        needs_list = [needs] if isinstance(needs, str) else cast(list[object], needs)
        assert "detect" in needs_list, f"build-and-publish must need 'detect'; got needs={needs!r}"
        if_cond = cast(str, pub.get("if", ""))
        assert "needs.detect.outputs.release == 'true'" in if_cond, (
            f"build-and-publish 'if' must gate on needs.detect.outputs.release == 'true'; "
            f"got {if_cond!r}"
        )

    def test_release_yml_detect_job_is_read_only(self) -> None:
        """Top-level permissions floor is contents:read; detect grants no write.

        Locks the least-privilege split: detect only reads pyproject.toml and curls
        PyPI, so it must never hold contents:write.  Guards against a future edit
        adding write grants to detect OR deleting the top-level read floor.
        """
        workflow = self._workflow()
        permissions = _yaml_mapping(workflow.get("permissions", {}))
        assert permissions.get("contents") == "read", (
            "top-level permissions.contents must be 'read' so jobs inherit least privilege"
        )
        jobs = _yaml_mapping(workflow["jobs"])
        detect = _yaml_mapping(jobs["detect"])
        detect_permissions = _yaml_mapping(detect.get("permissions", {}))
        detect_contents = detect_permissions.get("contents")
        assert detect_contents != "write", (
            "detect must stay read-only — it only reads pyproject.toml and curls PyPI; "
            "a contents:write grant would be a silent privilege over-grant"
        )

    def test_release_yml_detect_gate_has_both_release_branches(self) -> None:
        """detect job emits release=false for already-published versions (no-op, not exit 1)
        and release=true for new versions.

        Locks the dual-branch no-op design: an already-published version sets release=false
        so build-and-publish is skipped and the run is green — this is what makes
        push-on-main safe on non-release pushes.
        """
        raw = self.RELEASE_YML.read_text()
        assert "release=false" in raw, (
            "detect step must write release=false to GITHUB_OUTPUT for already-published versions"
        )
        assert "release=true" in raw, (
            "detect step must write release=true to GITHUB_OUTPUT for new versions"
        )
        assert "pypi.org/pypi/milknado" in raw, (
            "detect step must curl PyPI to gate on version existence"
        )

    def test_release_yml_detect_fails_closed_on_unexpected_status(self) -> None:
        """detect reads the explicit PyPI HTTP status and only publishes on a definite 404.

        Fail-closed: a 200 means already-published (skip), a 404 means new (publish), and
        any other response (5xx, or a network error rendered as '000') fails the job loudly
        via the catch-all rather than guessing release=true. Locks the gate against a
        regression to the old `curl -sfL` boolean that treated every non-200 as 'publish'.
        """
        raw = self.RELEASE_YML.read_text()
        assert "%{http_code}" in raw, (
            "detect must read the explicit PyPI HTTP status code, not a boolean curl exit, "
            "so a transient failure can't masquerade as a 404"
        )
        assert 'case "$status" in' in raw, (
            "detect must branch on the explicit status (200 skip / 404 publish / other fail)"
        )
        assert "refusing to guess" in raw, (
            "an unexpected PyPI status (5xx / network error) must fail the job loudly "
            "(fail-closed), not silently set release=true"
        )

    def test_release_yml_grants_contents_write(self) -> None:
        """Pushing the tag and force-pushing stable both need contents: write."""
        wf = self._workflow()
        job = _yaml_mapping(_yaml_mapping(wf["jobs"])["build-and-publish"])
        permissions = _yaml_mapping(job["permissions"])
        assert permissions["contents"] == "write", (
            "build-and-publish must grant contents: write to push the vX.Y.Z tag and "
            f"repoint stable; got {permissions.get('contents')!r}"
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
        workflow = self._workflow()
        job = _yaml_mapping(_yaml_mapping(workflow["jobs"])["build-and-publish"])
        assert job.get("environment") == "pypi", (
            f"build-and-publish job must set 'environment: pypi'; got {job.get('environment')!r}"
        )
        steps = _objects(job["steps"])
        uses_values = [cast(str, step["uses"]) for step in steps if "uses" in step]
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
        data = _read_toml(REPO / "pyproject.toml")
        tool = _mapping(data["tool"])
        hatch = _mapping(tool["hatch"])
        build = _mapping(hatch["build"])
        targets = _mapping(build["targets"])
        wheel = _mapping(targets["wheel"])
        packages = _strings(wheel["packages"])
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
        target change on one side only). Mirrors, does not exec, release.yml's jq."""
        version = _pyproject_version()
        data = _read_json(MCP_JSON)
        servers = _mapping(data["mcpServers"])
        server = _mapping(servers["milknado"])
        server["args"] = [
            "--from",
            f"milknado=={version}",
            "milknado-mcp",
        ]
        assert _strings(server["args"]) in _accepted_launcher_args(version)


class TestReadmeDocumentsChannels:
    """Spec acceptance: README documents the three channels + the uvx refresh caveat.
    Substring locks (version-independent) so a revert to single-channel install fails."""

    README: Path = REPO / "README.md"

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
