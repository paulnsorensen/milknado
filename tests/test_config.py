from __future__ import annotations

from pathlib import Path

import msgspec
import pytest

from milknado.domains.common.config import (
    Gate,
    MilknadoConfig,
    MilknadoSection,
    decode_milknado_section,
    default_config,
    detect_project_gates,
    load_config,
    save_config,
)


class TestDefaultConfig:
    def test_returns_milknado_config(self, tmp_path: Path) -> None:
        cfg = default_config(tmp_path)
        assert isinstance(cfg, MilknadoConfig)
        assert cfg.project_root == tmp_path
        assert cfg.agent_family == "claude"
        assert cfg.completion_timeout_seconds is None

    def test_db_path_under_project_root(self, tmp_path: Path) -> None:
        cfg = default_config(tmp_path)
        assert cfg.db_path == tmp_path / ".milknado" / "milknado.db"

    def test_planning_agent_not_empty(self, tmp_path: Path) -> None:
        cfg = default_config(tmp_path)
        assert len(cfg.planning_agent) > 0

    def test_execution_agent_not_empty(self, tmp_path: Path) -> None:
        cfg = default_config(tmp_path)
        assert len(cfg.execution_agent) > 0


class TestLoadConfig:
    def _write_toml(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "milknado.toml"
        _ = p.write_text(content, encoding="utf-8")
        return p

    def test_loads_minimal_config(self, tmp_path: Path) -> None:
        toml = '[milknado]\nagent_family = "claude"\n'
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert cfg.agent_family == "claude"
        assert cfg.project_root == tmp_path

    def test_loads_custom_quality_gates(self, tmp_path: Path) -> None:
        toml = (
            "[milknado]\n"
            'agent_family = "claude"\n'
            'quality_gates = ["uv run pytest", "uv run ruff check"]\n'
        )
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert cfg.quality_gates is not None
        assert Gate(command="uv run pytest") in cfg.quality_gates

    def test_loads_concurrency_limit(self, tmp_path: Path) -> None:
        toml = '[milknado]\nagent_family = "claude"\nconcurrency_limit = 8\n'
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert cfg.concurrency_limit == 8

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("concurrency_limit", "8"),
            ("stall_threshold_seconds", "300"),
            ("dispatch_max_retries", "2"),
            ("dispatch_backoff_seconds", "5.0"),
            ("completion_timeout_seconds", "30.0"),
            ("eta_sample_size", "10"),
        ],
    )
    def test_rejects_coercible_numeric_strings(self, field: str, value: str) -> None:
        with pytest.raises(msgspec.ValidationError, match=field):
            _ = msgspec.convert({field: value}, type=MilknadoSection, strict=True)

    def test_invalid_family_error_hides_input(self) -> None:
        with pytest.raises(msgspec.ValidationError) as exc_info:
            _ = decode_milknado_section({"agent_family": "TOPSECRET"})

        assert "topsecret" not in str(exc_info.value).lower()

    def test_decoder_does_not_mutate_raw_worker_tools(self) -> None:
        raw: dict[str, object] = {"worker": {"tools": {"claude": ["Read"]}}}

        _ = decode_milknado_section(raw)

        assert raw == {"worker": {"tools": {"claude": ["Read"]}}}

    def test_loads_custom_planning_agent(self, tmp_path: Path) -> None:
        toml = '[milknado]\nagent_family = "claude"\nplanning_agent = "claude --model opus"\n'
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert "opus" in cfg.planning_agent

    def test_loads_custom_execution_agent(self, tmp_path: Path) -> None:
        toml = '[milknado]\nagent_family = "claude"\nexecution_agent = "claude --model haiku"\n'
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert "haiku" in cfg.execution_agent

    def test_loads_planning_validation_hook(self, tmp_path: Path) -> None:
        toml = (
            '[milknado]\nagent_family = "claude"\n'
            'planning_validation_hook = "python scripts/validate_plan.py"\n'
        )
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert cfg.planning_validation_hook == "python scripts/validate_plan.py"

    def test_invalid_family_raises(self, tmp_path: Path) -> None:
        toml = '[milknado]\nagent_family = "openai"\n'
        path = self._write_toml(tmp_path, toml)
        with pytest.raises(ValueError, match="Invalid agent_family"):
            _ = load_config(path)

    def test_db_path_relative_to_project(self, tmp_path: Path) -> None:
        toml = '[milknado]\nagent_family = "claude"\ndb_path = ".milknado/custom.db"\n'
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert cfg.db_path == tmp_path / ".milknado" / "custom.db"

    def test_loads_all_numeric_fields(self, tmp_path: Path) -> None:
        toml = (
            "[milknado]\n"
            'agent_family = "claude"\n'
            "stall_threshold_seconds = 600\n"
            "dispatch_max_retries = 5\n"
            "dispatch_backoff_seconds = 10.0\n"
            "completion_timeout_seconds = 3600.0\n"
            "eta_sample_size = 20\n"
        )
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert cfg.stall_threshold_seconds == 600
        assert cfg.dispatch_max_retries == 5
        assert cfg.dispatch_backoff_seconds == 10.0
        assert cfg.completion_timeout_seconds == 3600.0
        assert cfg.eta_sample_size == 20

    def test_loads_plugins_list(self, tmp_path: Path) -> None:
        toml = '[milknado]\nagent_family = "claude"\nplugins = ["plugin-a", "plugin-b"]\n'
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert "plugin-a" in cfg.plugins

    def test_loads_protected_branches(self, tmp_path: Path) -> None:
        toml = '[milknado]\nagent_family = "claude"\nprotected_branches = ["main", "develop"]\n'
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert "develop" in cfg.protected_branches

    def test_top_level_without_milknado_section(self, tmp_path: Path) -> None:
        toml = 'agent_family = "claude"\n'
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert cfg.agent_family == "claude"

    def test_db_path_dotdot_traversal_raises(self, tmp_path: Path) -> None:
        toml = '[milknado]\nagent_family = "claude"\ndb_path = "../../evil.db"\n'
        path = self._write_toml(tmp_path, toml)
        with pytest.raises(ValueError, match="escapes project_root"):
            _ = load_config(path)

    def test_db_path_absolute_path_raises(self, tmp_path: Path) -> None:
        toml = '[milknado]\nagent_family = "claude"\ndb_path = "/etc/evil.db"\n'
        path = self._write_toml(tmp_path, toml)
        with pytest.raises(ValueError, match="escapes project_root"):
            _ = load_config(path)

    def test_worktree_config_requires_boolean(self, tmp_path: Path) -> None:
        path = self._write_toml(
            tmp_path,
            '[milknado]\nagent_family = "claude"\nworktree = "nope"\n',
        )
        with pytest.raises(ValueError, match=r"\[milknado\] worktree must be a boolean"):
            _ = load_config(path)


class TestSaveConfig:
    def test_writes_toml_file(self, tmp_path: Path) -> None:
        cfg = default_config(tmp_path)
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        assert path.exists()
        content = path.read_text()
        assert "[milknado]" in content
        assert "agent_family" in content

    def test_roundtrip(self, tmp_path: Path) -> None:
        cfg = default_config(tmp_path)
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded.agent_family == cfg.agent_family
        assert loaded.concurrency_limit == cfg.concurrency_limit
        assert loaded.worktree is True
        assert loaded.planning_validation_hook is None
        assert loaded.completion_timeout_seconds is None

    def test_roundtrip_preserves_explicit_completion_timeout(self, tmp_path: Path) -> None:
        cfg = msgspec.structs.replace(default_config(tmp_path), completion_timeout_seconds=3600.0)
        path = tmp_path / "milknado.toml"

        save_config(cfg, path)

        loaded = load_config(path)
        assert loaded.completion_timeout_seconds == 3600.0

    def test_escapes_backslashes(self, tmp_path: Path) -> None:
        cfg = MilknadoConfig(
            planning_agent='cmd "quoted"',
            execution_agent="cmd",
            project_root=tmp_path,
            db_path=tmp_path / ".milknado" / "milknado.db",
        )
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        content = path.read_text()
        assert '\\"quoted\\"' in content

    def test_roundtrip_preserves_prompt_prepends(self, tmp_path: Path) -> None:
        cfg = MilknadoConfig(
            project_root=tmp_path,
            db_path=tmp_path / ".milknado" / "milknado.db",
            planning_prompt_prepend="team rule: just build first",
            worker_brief_prepend="touch only listed files\nfollow up via tracker",
        )
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        loaded = load_config(path, include_global=False)
        assert loaded.planning_prompt_prepend == "team rule: just build first"
        assert loaded.worker_brief_prepend == "touch only listed files\nfollow up via tracker"

    def test_roundtrip_preserves_worker_tools(self, tmp_path: Path) -> None:
        # Build via load_config (no explicit execution_agent) so execution_agent
        # is the command DERIVED from the override — the realistic shape that
        # save_config suppresses and a reload re-derives.
        src = tmp_path / "in.toml"
        _ = src.write_text(
            """[milknado]
agent_family = "claude"

[milknado.worker.tools]
claude = [\"...\", \"mcp__github__*\"]
""",
            encoding="utf-8",
        )
        cfg = load_config(src, include_global=False)
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        loaded = load_config(path, include_global=False)
        # The execution_agent rebuild should reflect the round-tripped override.
        assert "mcp__github__*" in loaded.execution_agent
        # And the structured override itself survives the round trip.
        assert "mcp__github__*" in loaded.worker_tools.get("claude", ())

    def test_roundtrip_preserves_explicit_empty_worker_tools(self, tmp_path: Path) -> None:
        # An explicit empty list means "no tools" (replaces the family default
        # entirely); dropping it on save would silently restore the default
        # tool set on reload.
        src = tmp_path / "in.toml"
        _ = src.write_text(
            '[milknado]\nagent_family = "claude"\n\n[milknado.worker.tools]\nclaude = []\n',
            encoding="utf-8",
        )
        cfg = load_config(src, include_global=False)
        assert cfg.worker_tools.get("claude") == ()
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        loaded = load_config(path, include_global=False)
        assert loaded.worker_tools.get("claude") == ()

    def test_config_roundtrips_commit_footer(self, tmp_path: Path) -> None:
        cfg = MilknadoConfig(
            project_root=tmp_path,
            db_path=tmp_path / ".milknado" / "milknado.db",
            commit_footer="Co-authored-by: Team <team@example.com>",
        )
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        loaded = load_config(path, include_global=False)
        assert loaded.commit_footer == "Co-authored-by: Team <team@example.com>"

    def test_commit_footer_omitted_stays_none(self, tmp_path: Path) -> None:
        cfg = default_config(tmp_path)
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        content = path.read_text()
        assert "commit_footer" not in content
        loaded = load_config(path, include_global=False)
        assert loaded.commit_footer is None

    def test_config_roundtrips_plan_reviewer_agent(self, tmp_path: Path) -> None:
        cfg = MilknadoConfig(
            project_root=tmp_path,
            db_path=tmp_path / ".milknado" / "milknado.db",
            plan_reviewer_agent="claude --model sonnet -p",
            plan_review_max_rounds=5,
        )
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        loaded = load_config(path, include_global=False)
        assert loaded.plan_reviewer_agent == "claude --model sonnet -p"
        assert loaded.plan_review_max_rounds == 5

    def test_plan_reviewer_agent_omitted_stays_none(self, tmp_path: Path) -> None:
        cfg = default_config(tmp_path)
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        content = path.read_text()
        assert "plan_reviewer_agent" not in content
        loaded = load_config(path, include_global=False)
        assert loaded.plan_reviewer_agent is None
        assert loaded.plan_review_max_rounds == 3

    def test_roundtrip_skips_empty_prompt_and_worker_sections(self, tmp_path: Path) -> None:
        cfg = default_config(tmp_path)
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        content = path.read_text()
        assert "[milknado.prompts]" not in content
        assert "[milknado.worker.tools." not in content

    def test_save_config_with_relative_project_root_and_absolute_db_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """os.path.relpath handles mixed absolute/relative roots; Path.relative_to raises."""
        monkeypatch.chdir(tmp_path)
        cfg = MilknadoConfig(
            project_root=Path("."),
            db_path=tmp_path / ".milknado" / "milknado.db",
        )
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        content = path.read_text()
        assert "milknado.db" in content


def _section_gates(raw: object) -> tuple[Gate, ...] | None:
    """Validate raw TOML ``quality_gates`` through the ``[milknado]`` schema."""
    return decode_milknado_section({"quality_gates": raw}).quality_gates


class TestParseGates:
    def test_none_returns_none(self) -> None:
        assert _section_gates(None) is None

    def test_empty_list_returns_empty_tuple(self) -> None:
        assert _section_gates([]) == ()

    def test_string_entries_become_gate_objects(self) -> None:
        result = _section_gates(["uv run pytest", "uv run ruff check"])
        assert result == (Gate(command="uv run pytest"), Gate(command="uv run ruff check"))

    def test_table_entry_with_fail_on_stdout(self) -> None:
        raw = [{"command": "godot --headless", "fail_on_stdout": "SCRIPT ERROR"}]
        result = _section_gates(raw)
        assert result == (Gate(command="godot --headless", fail_on_stdout="SCRIPT ERROR"),)

    def test_table_entry_without_fail_on_stdout(self) -> None:
        raw = [{"command": "cargo test"}]
        result = _section_gates(raw)
        assert result == (Gate(command="cargo test"),)

    def test_non_list_raises_value_error(self) -> None:
        with pytest.raises(msgspec.ValidationError, match="must be a list"):
            _ = _section_gates("uv run pytest")

    def test_empty_string_entry_raises_value_error(self) -> None:
        with pytest.raises(msgspec.ValidationError, match="non-empty string"):
            _ = _section_gates([""])

    def test_bad_regex_fail_on_stdout_raises(self) -> None:
        raw = [{"command": "godot", "fail_on_stdout": "[invalid"}]
        with pytest.raises(msgspec.ValidationError, match="not a valid regex"):
            _ = _section_gates(raw)

    def test_dict_missing_command_raises(self) -> None:
        with pytest.raises(msgspec.ValidationError, match="command"):
            _ = _section_gates([{"fail_on_stdout": "ERROR"}])

    def test_wrong_type_entry_raises(self) -> None:
        with pytest.raises(msgspec.ValidationError, match="string or a table"):
            _ = _section_gates([42])

    def test_non_string_fail_on_stdout_raises(self) -> None:
        with pytest.raises(msgspec.ValidationError, match="fail_on_stdout"):
            _ = _section_gates([{"command": "godot", "fail_on_stdout": 42}])

    def test_whitespace_only_string_entry_raises(self) -> None:
        with pytest.raises(msgspec.ValidationError, match="non-empty string"):
            _ = _section_gates(["   "])

    def test_whitespace_only_command_raises(self) -> None:
        with pytest.raises(msgspec.ValidationError, match="non-empty string"):
            _ = _section_gates([{"command": "   "}])

    def test_whitespace_only_fail_on_stdout_treated_as_absent(self) -> None:
        result = _section_gates([{"command": "godot", "fail_on_stdout": "   "}])
        assert result == (Gate(command="godot", fail_on_stdout=None),)


class TestDetectProjectGates:
    def test_python_project_returns_python_triple(self, tmp_path: Path) -> None:
        _ = (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
        result = detect_project_gates(tmp_path)
        assert result is not None
        commands = [g.command for g in result]
        assert commands == [
            "uv run pytest",
            "uv run ruff check",
            "uv run basedpyright",
        ]

    def test_rust_project_returns_cargo_gates(self, tmp_path: Path) -> None:
        _ = (tmp_path / "Cargo.toml").write_text("[package]\nname = 'x'\n", encoding="utf-8")
        result = detect_project_gates(tmp_path)
        assert result is not None
        commands = [g.command for g in result]
        assert any("cargo" in c for c in commands)

    def test_node_project_returns_npm_test(self, tmp_path: Path) -> None:
        _ = (tmp_path / "package.json").write_text('{"name":"x"}', encoding="utf-8")
        result = detect_project_gates(tmp_path)
        assert result is not None
        commands = [g.command for g in result]
        assert any("npm" in c for c in commands)

    def test_go_project_returns_go_gates(self, tmp_path: Path) -> None:
        _ = (tmp_path / "go.mod").write_text("module x\ngo 1.21\n", encoding="utf-8")
        result = detect_project_gates(tmp_path)
        assert result is not None
        commands = [g.command for g in result]
        assert any("go" in c for c in commands)

    def test_godot_project_returns_gate_with_fail_on_stdout(self, tmp_path: Path) -> None:
        _ = (tmp_path / "project.godot").write_text("[gd_resource]\n", encoding="utf-8")
        result = detect_project_gates(tmp_path)
        assert result is not None
        patterns = [g.fail_on_stdout for g in result if g.fail_on_stdout is not None]
        assert patterns, "expected at least one gate with fail_on_stdout"
        combined = "|".join(patterns)
        assert "SCRIPT ERROR" in combined
        assert "^ERROR:" in combined
        assert "FAILED" not in combined

    def test_empty_dir_returns_none(self, tmp_path: Path) -> None:
        result = detect_project_gates(tmp_path)
        assert result is None

    def test_pyproject_wins_over_cargo(self, tmp_path: Path) -> None:
        """First match wins: pyproject.toml before Cargo.toml."""
        _ = (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
        _ = (tmp_path / "Cargo.toml").write_text("[package]\nname = 'x'\n", encoding="utf-8")
        result = detect_project_gates(tmp_path)
        assert result is not None
        # Python triple includes uv run pytest
        assert any("uv" in g.command for g in result)


class TestSaveConfigGates:
    def test_string_gate_serialized_as_bare_string(self, tmp_path: Path) -> None:
        cfg = msgspec.structs.replace(
            default_config(tmp_path), quality_gates=(Gate(command="uv run pytest"),)
        )
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        content = path.read_text()
        assert '"uv run pytest"' in content

    def test_gate_with_fail_on_stdout_serialized_as_inline_table(self, tmp_path: Path) -> None:
        cfg = msgspec.structs.replace(
            default_config(tmp_path),
            quality_gates=(Gate(command="godot --headless", fail_on_stdout="SCRIPT ERROR"),),
        )
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        content = path.read_text()
        assert "SCRIPT ERROR" in content
        assert "fail_on_stdout" in content

    def test_roundtrip_string_gate(self, tmp_path: Path) -> None:
        cfg = msgspec.structs.replace(
            default_config(tmp_path),
            quality_gates=(Gate(command="uv run pytest"), Gate(command="uv run ruff check")),
        )
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        loaded = load_config(path, include_global=False)
        assert loaded.quality_gates == (
            Gate(command="uv run pytest"),
            Gate(command="uv run ruff check"),
        )

    def test_roundtrip_gate_with_fail_on_stdout(self, tmp_path: Path) -> None:
        gate = Gate(command="godot --headless --run-tests", fail_on_stdout="SCRIPT ERROR|FAILED")
        cfg = msgspec.structs.replace(default_config(tmp_path), quality_gates=(gate,))
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        loaded = load_config(path, include_global=False)
        assert loaded.quality_gates == (gate,)

    def test_none_gates_not_written_to_toml(self, tmp_path: Path) -> None:
        """quality_gates=None (unconfigured) must not emit a key — absence is the signal."""
        cfg = msgspec.structs.replace(default_config(tmp_path), quality_gates=None)
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        loaded = load_config(path, include_global=False)
        assert loaded.quality_gates is None

    def test_empty_tuple_gates_written_as_empty_list(self, tmp_path: Path) -> None:
        """quality_gates=() (explicit skip) round-trips back as empty tuple."""
        cfg = msgspec.structs.replace(default_config(tmp_path), quality_gates=())
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        loaded = load_config(path, include_global=False)
        assert loaded.quality_gates == ()
