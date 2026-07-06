"""Tier-2 completion verifier attached by LoopAdapter.create_run.

The verifier is the harness-side authoritative re-run of the quality gates plus
a non-empty-diff check, consulted by the engine when exit-0 + promise would
otherwise complete a node. These tests pin the three-branch verdict contract:
a failing gate yields ok=False naming the command, an empty diff yields ok=False
with the no-change feedback, and passing gates plus a non-empty diff yield ok=True.
They also pin that create_run attaches exactly this closure to the RunConfig.

New in this version:
- None gates → fail-closed with actionable message (no gates configured).
- ()   gates → explicit skip → passes (spec/research flavors).
- fail_on_stdout: a gate that exits 0 but matches the pattern → fails.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from milknado.adapters.git import GitAdapter
from milknado.adapters.loop import (
    _GATE_TIMEOUT_SECONDS,
    _GIT_QUERY_TIMEOUT_SECONDS,
    _UNRESOLVABLE_BASE_FEEDBACK,
    NO_GATES_CONFIGURED_MESSAGE,
    LoopAdapter,
    _build_completion_verifier,
    _has_committed_change,
    _has_working_tree_change,
    _resolve_feature_branch,
    _run_quality_gates,
)
from milknado.domains.common.config import Gate
from milknado.loop import CompletionVerdict
from milknado.loop._run_types import DEFAULT_COMMAND_TIMEOUT


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture()
def feature_repo(tmp_path: Path) -> Path:
    """A repo on a feature branch with one seed commit; returns the repo root."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-b", "feature", cwd=repo)
    _git("config", "user.email", "t@t.t", cwd=repo)
    _git("config", "user.name", "t", cwd=repo)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-m", "seed", cwd=repo)
    return repo


@pytest.fixture()
def worktree(feature_repo: Path, tmp_path: Path) -> Path:
    """A worktree forked from the feature branch's HEAD, on its own branch."""
    wt = tmp_path / "wt"
    _git("worktree", "add", "-b", "milknado/1-x", str(wt), cwd=feature_repo)
    return wt


def _commit_change(wt: Path) -> None:
    (wt / "work.txt").write_text("work\n", encoding="utf-8")
    _git("add", "-A", cwd=wt)
    _git("commit", "-m", "work", cwd=wt)


class TestGateEnvHermeticity:
    """The completion gate must run each quality command as CI would — free of
    milknado's own injected orchestration context. A worker runs the gate as a
    subprocess, so an ambient MILKNADO_RUN_ID / MILKNADO_NODE_ID would leak the
    worker's node/run identity into the suite (#178) and flake any test that reads
    the MILKNADO_* namespace (worker-env or follow-up-parent assertions).
    """

    def test_gate_env_strips_milknado_namespace(
        self, worktree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MILKNADO_RUN_ID", "leak-run-123")
        monkeypatch.setenv("MILKNADO_NODE_ID", "77")
        # Fails (exit 1) if either injected var is visible to the gate command;
        # uses shell builtins only so the check does not itself depend on PATH.
        probe = Gate('test -z "$MILKNADO_RUN_ID" && test -z "$MILKNADO_NODE_ID"')
        assert _run_quality_gates(worktree, [probe]) is None

    def test_gate_env_preserves_non_milknado_vars(
        self, worktree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stripping MILKNADO_* must not nuke the rest of the env: PATH must survive
        # so an external command still resolves (guards against passing env={}).
        monkeypatch.setenv("MILKNADO_RUN_ID", "leak-run-123")
        assert _run_quality_gates(worktree, [Gate("env >/dev/null")]) is None


class TestFailingGate:
    def test_failing_gate_yields_not_ok_naming_command(self, worktree: Path) -> None:
        _commit_change(worktree)
        cmd = "sh -c 'echo boom-marker >&2; exit 1'"
        verifier = _build_completion_verifier(worktree, (Gate(cmd),))

        verdict = verifier()

        assert verdict.ok is False
        assert cmd in verdict.feedback

    def test_failing_gate_includes_output_tail(self, worktree: Path) -> None:
        _commit_change(worktree)
        verifier = _build_completion_verifier(
            worktree, (Gate("sh -c 'echo unique-tail-token; exit 3'"),)
        )

        verdict = verifier()

        assert verdict.ok is False
        assert "unique-tail-token" in verdict.feedback

    def test_first_failing_gate_short_circuits(self, worktree: Path) -> None:
        _commit_change(worktree)
        verifier = _build_completion_verifier(
            worktree,
            (Gate("sh -c 'exit 1'"), Gate("sh -c 'echo second-gate-ran; exit 0'")),
        )

        verdict = verifier()

        assert verdict.ok is False
        assert "second-gate-ran" not in verdict.feedback

    def test_timed_out_gate_yields_not_ok_naming_command(
        self, worktree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A gate that exceeds the cap is a failure, not a silent pass."""
        import milknado.adapters.loop as loop_mod

        monkeypatch.setattr(loop_mod, "_GATE_TIMEOUT_SECONDS", 0.2)
        _commit_change(worktree)
        verifier = _build_completion_verifier(worktree, (Gate("sh -c 'sleep 5'"),))

        verdict = verifier()

        assert verdict.ok is False
        assert "timed out" in verdict.feedback
        assert "sleep 5" in verdict.feedback

    def test_failing_gate_logged_at_warning(
        self, worktree: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A gate failure must reach the operator's logs at warning level, not
        only the agent's rejection feedback — a budget-burning reject loop on a
        non-interactive server run is otherwise invisible."""
        _commit_change(worktree)
        verifier = _build_completion_verifier(worktree, (Gate("sh -c 'exit 2'"),))

        with caplog.at_level("WARNING", logger="milknado.adapters.loop"):
            verifier()

        assert any(
            r.levelname == "WARNING" and "quality gate" in r.message and "failed" in r.message
            for r in caplog.records
        )

    def test_timed_out_gate_logged_at_warning(
        self, worktree: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A gate timeout — the exact failure mode the high finding predicts for
        an undersized cap — must also be diagnosable from the operator's logs."""
        import milknado.adapters.loop as loop_mod

        monkeypatch.setattr(loop_mod, "_GATE_TIMEOUT_SECONDS", 0.2)
        _commit_change(worktree)
        verifier = _build_completion_verifier(worktree, (Gate("sh -c 'sleep 5'"),))

        with caplog.at_level("WARNING", logger="milknado.adapters.loop"):
            verifier()

        assert any(r.levelname == "WARNING" and "timed out" in r.message for r in caplog.records)

    def test_failing_gate_beats_empty_diff_in_feedback(self, worktree: Path) -> None:
        """Gates run before the diff check: a failing gate on an empty worktree
        reports the gate failure, not the no-change message — the agent must see
        the harness reason it actually hit first."""
        verifier = _build_completion_verifier(worktree, (Gate("sh -c 'exit 1'"),))

        verdict = verifier()

        assert verdict.ok is False
        assert "quality gate" in verdict.feedback
        assert "no committed/stageable change" not in verdict.feedback


class TestEmptyDiff:
    def test_empty_diff_yields_no_change_feedback(self, worktree: Path) -> None:
        verifier = _build_completion_verifier(worktree, (Gate("true"),))

        verdict = verifier()

        assert verdict.ok is False
        assert verdict.feedback == (
            "you emitted the completion promise but produced no committed/stageable change"
        )

    def test_gates_pass_but_no_change_still_rejected(self, worktree: Path) -> None:
        """A green harness with zero produced change must not be accepted."""
        verifier = _build_completion_verifier(worktree, (Gate("true"), Gate("true")))

        assert verifier().ok is False


class TestAllGreen:
    def test_passing_gates_and_committed_change_ok(self, worktree: Path) -> None:
        _commit_change(worktree)
        verifier = _build_completion_verifier(worktree, (Gate("true"),))

        verdict = verifier()

        assert verdict.ok is True
        assert verdict.feedback == ""

    def test_passing_gates_and_uncommitted_change_ok(self, worktree: Path) -> None:
        """A stageable-but-uncommitted change counts as produced work."""
        (worktree / "dirty.txt").write_text("dirty\n", encoding="utf-8")
        verifier = _build_completion_verifier(worktree, (Gate("true"),))

        assert verifier().ok is True

    def test_no_gates_with_change_ok(self, worktree: Path) -> None:
        """Empty tuple (explicit skip) plus a real change is the all-pass path."""
        _commit_change(worktree)
        verifier = _build_completion_verifier(worktree, ())

        assert verifier().ok is True


class TestNoneGatesFailClosed:
    """quality_gates = None means unconfigured — verifier must fail closed immediately."""

    def test_none_gates_yields_not_ok(self, worktree: Path) -> None:
        """None gates never pass, even with a real committed change."""
        _commit_change(worktree)
        verifier = _build_completion_verifier(worktree, None)

        verdict = verifier()

        assert verdict.ok is False

    def test_none_gates_feedback_names_the_fix(self, worktree: Path) -> None:
        """The fail-closed message must tell the user exactly how to fix it."""
        verifier = _build_completion_verifier(worktree, None)

        verdict = verifier()

        assert verdict.feedback == NO_GATES_CONFIGURED_MESSAGE
        assert "milknado.toml" in verdict.feedback
        assert "quality_gates" in verdict.feedback

    def test_empty_tuple_is_not_fail_closed(self, worktree: Path) -> None:
        """() (explicit skip) is distinct from None — it must not trigger fail-closed."""
        _commit_change(worktree)
        verifier = _build_completion_verifier(worktree, ())

        verdict = verifier()

        assert verdict.ok is True
        assert NO_GATES_CONFIGURED_MESSAGE not in verdict.feedback

    def test_empty_tuple_logs_explicit_skip(
        self, worktree: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """() gates must emit an info log so the skip is auditable."""
        import logging

        _commit_change(worktree)
        verifier = _build_completion_verifier(worktree, ())

        with caplog.at_level(logging.INFO, logger="milknado.adapters.loop"):
            verifier()

        assert any("explicitly skipped" in r.message for r in caplog.records)


class TestFailOnStdout:
    """Gates with fail_on_stdout: exit 0 + matching output → failure."""

    def test_matching_stdout_on_exit_0_fails(self, worktree: Path) -> None:
        """A gate that exits 0 but prints a pattern match should be a failure.
        This covers Godot headless and similar harnesses that exit 0 on errors."""
        _commit_change(worktree)
        gate = Gate("sh -c 'echo SCRIPT ERROR: oops; exit 0'", fail_on_stdout="SCRIPT ERROR")
        verifier = _build_completion_verifier(worktree, (gate,))

        verdict = verifier()

        assert verdict.ok is False
        assert "SCRIPT ERROR" in verdict.feedback

    def test_non_matching_stdout_on_exit_0_passes(self, worktree: Path) -> None:
        """Output that does not match the pattern should not trip the gate."""
        _commit_change(worktree)
        gate = Gate("sh -c 'echo all good; exit 0'", fail_on_stdout="SCRIPT ERROR|FAILED")
        verifier = _build_completion_verifier(worktree, (gate,))

        verdict = verifier()

        assert verdict.ok is True

    def test_fail_on_stdout_with_nonzero_exit_also_fails(self, worktree: Path) -> None:
        """Non-zero exit still fails even if fail_on_stdout would also match."""
        _commit_change(worktree)
        gate = Gate("sh -c 'echo FAILED; exit 1'", fail_on_stdout="FAILED")
        verifier = _build_completion_verifier(worktree, (gate,))

        verdict = verifier()

        assert verdict.ok is False
        assert "failed (exit 1)" in verdict.feedback

    def test_fail_on_stdout_names_command_in_feedback(self, worktree: Path) -> None:
        _commit_change(worktree)
        cmd = "sh -c 'echo SCRIPT ERROR; exit 0'"
        gate = Gate(cmd, fail_on_stdout="SCRIPT ERROR")
        verifier = _build_completion_verifier(worktree, (gate,))

        verdict = verifier()

        assert cmd in verdict.feedback

    def test_fail_on_stdout_pattern_does_not_match_across_stream_boundary(
        self, worktree: Path
    ) -> None:
        """Pattern split across stdout/stderr boundary must not match (newline separator)."""
        _commit_change(worktree)
        # stdout ends with "SCRIPT ", stderr starts with "ERROR" — would join without separator
        gate = Gate(
            'sh -c \'printf "SCRIPT " >&1; printf "ERROR" >&2; exit 0\'',
            fail_on_stdout="SCRIPT ERROR",
        )
        verifier = _build_completion_verifier(worktree, (gate,))

        verdict = verifier()

        assert verdict.ok is True


class TestFeatureBranchResolution:
    def test_resolves_main_worktree_branch_as_fork_point(self, worktree: Path) -> None:
        """The worktree's fork point is the main worktree's branch ('feature')."""
        assert _resolve_feature_branch(worktree) == "feature"

    def test_detached_main_worktree_yields_none(self, feature_repo: Path) -> None:
        """A detached main worktree has no branch line; resolution returns None."""
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=feature_repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        _git("checkout", "--detach", head, cwd=feature_repo)

        assert _resolve_feature_branch(feature_repo) is None

    def test_non_git_dir_yields_none(self, tmp_path: Path) -> None:
        """Outside a git repo, resolution fails closed rather than raising."""
        assert _resolve_feature_branch(tmp_path) is None

    def test_resolved_base_matches_executor_rebase_target(self, tmp_path: Path) -> None:
        """Integration seam (Curd 6 verifier <-> Curd 4 merge-back): the branch the
        verifier diffs against must be the SAME ref the executor rebases/ff onto.

        The executor receives ``feature_branch`` from ``GitAdapter.current_branch()``
        at the orchestrator root (cli_run.run: ``feature_branch = git.current_branch()``),
        while the verifier independently re-derives it from ``git worktree list`` inside
        the node worktree. If those two derivations drift, the empty-diff check measures
        against the wrong base — accepting work that never lands, or rejecting work that
        would. A non-default branch name proves the equality is structural, not a
        coincidence of the literal 'feature' both sides happen to use elsewhere.
        """
        repo = tmp_path / "orch"
        repo.mkdir()
        _git("init", "-b", "release/v2", cwd=repo)
        _git("config", "user.email", "t@t.t", cwd=repo)
        _git("config", "user.name", "t", cwd=repo)
        (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
        _git("add", "-A", cwd=repo)
        _git("commit", "-m", "seed", cwd=repo)
        wt = tmp_path / "node-wt"
        _git("worktree", "add", "-b", "milknado/9-x", str(wt), cwd=repo)

        executor_target = GitAdapter(repo).current_branch()
        verifier_base = _resolve_feature_branch(wt)

        assert executor_target == "release/v2"
        assert verifier_base == executor_target


class TestGitFailureFailsClosed:
    """A failing git invocation must not raise out of the verifier; it logs and
    returns a safe degraded value (no spurious accept)."""

    def test_working_tree_check_on_non_git_dir_is_false(self, tmp_path: Path) -> None:
        assert _has_working_tree_change(tmp_path) is False

    def test_committed_check_on_non_git_dir_is_false(self, tmp_path: Path) -> None:
        assert _has_committed_change(tmp_path, "feature") is False

    def test_committed_check_fails_closed_on_diff_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An errored `git diff --quiet` (exit >=2, e.g. 128) must read as 'no
        committed change', not be mistaken for a non-empty diff (exit 1). Only
        exit 1 signals real change; treating a broken git call as change would
        let a worker that produced nothing slip through completion."""
        import milknado.adapters.loop as loop_mod

        def fake_run(cmd: list[str], *args: object, **kwargs: object) -> object:
            if "merge-base" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="base\n", stderr="")
            if "diff" in cmd:
                return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="fatal: bad object")
            raise AssertionError(f"unexpected git call: {cmd}")

        monkeypatch.setattr(loop_mod.subprocess, "run", fake_run)

        assert _has_committed_change(tmp_path, "feature") is False

    def test_clean_tree_unresolvable_base_is_rejected(self, feature_repo: Path) -> None:
        """Detached main worktree + clean tree: no fork point, so the verifier
        cannot prove produced change and rejects rather than falsely accepting.

        The rejection must name the *resolution* failure, not the empty-diff
        message — an operator (and the agent) must be able to tell a degraded
        verifier apart from a worker that genuinely idled. Conflating the two
        sends an honest-but-committed worker chasing a phantom empty diff."""
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=feature_repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        _git("checkout", "--detach", head, cwd=feature_repo)
        verifier = _build_completion_verifier(feature_repo, (Gate("true"),))

        verdict = verifier()

        assert verdict.ok is False
        assert verdict.feedback == _UNRESOLVABLE_BASE_FEEDBACK
        assert "no committed/stageable change" not in verdict.feedback


class TestGateCapSizing:
    """The high-severity defect this guards: the per-gate cap was a 60s borrow
    from DEFAULT_COMMAND_TIMEOUT (sized for short RALPH.md frontmatter snippets),
    but the gates are the repo's real quality suite — `uv run pytest` over the
    full test set runs for minutes. A 60s cap times out every honest completion,
    retries to budget, and fails loud. These tests encode WHY the cap is large:
    it must clear the run-dispatch scale a whole node loop is allotted, not the
    per-command default; only the near-instant git probes keep the short cap."""

    def test_gate_cap_clears_full_suite_scale_not_per_command_default(self) -> None:
        # A real `uv run pytest` gate runs for minutes; the cap must exceed the
        # 60s per-command default by a wide margin or every honest node fails.
        assert _GATE_TIMEOUT_SECONDS > DEFAULT_COMMAND_TIMEOUT
        # Sized at the run-dispatch ceiling (1800s default a node loop is given).
        assert _GATE_TIMEOUT_SECONDS >= 1800.0

    def test_git_probes_keep_the_short_cap(self) -> None:
        # The verifier's git queries are near-instant; a hung one is a degraded
        # environment, not slow honest work, so it keeps the short ceiling.
        assert _GIT_QUERY_TIMEOUT_SECONDS == DEFAULT_COMMAND_TIMEOUT

    def test_gate_exceeding_old_60s_default_still_passes(self, worktree: Path) -> None:
        """Behavioural proof of the fix: a gate that runs longer than the old 60s
        cap completes within the new one. Run at a compressed scale (cap shrunk,
        gate sleeps past the *old* default-equivalent) so CI stays fast while the
        relationship under test — gate duration > per-command default < gate cap —
        is the same one a full pytest suite hits in production."""
        import milknado.adapters.loop as loop_mod

        # Compressed mirror: gate runs ~1.5s, "old default" 1s, new cap 10s. A
        # 1.5s gate would have timed out at a 1s cap (the bug) but passes at 10s.
        loop_mod_gate_cap = 10.0
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(loop_mod, "_GATE_TIMEOUT_SECONDS", loop_mod_gate_cap)
            _commit_change(worktree)
            verifier = _build_completion_verifier(worktree, (Gate("sh -c 'sleep 1.5'"),))

            verdict = verifier()

        assert verdict.ok is True, "a gate longer than the per-command default must not time out"


class TestCreateRunAttachesVerifier:
    def test_create_run_sets_callable_verifier_on_config(self, worktree: Path) -> None:
        _commit_change(worktree)
        adapter = LoopAdapter()
        run = adapter.create_run(
            agent="claude",
            ralph_dir=worktree,
            ralph_file=worktree / "RALPH.md",
            commands=[],
            quality_gates=(Gate("true"),),
        )

        verifier = run.config.completion_verifier
        assert callable(verifier)
        assert isinstance(verifier(), CompletionVerdict)
        assert verifier().ok is True

    def test_create_run_with_none_gates_verifier_fails_closed(self, worktree: Path) -> None:
        """When quality_gates=None, the attached verifier must fail-closed."""
        _commit_change(worktree)
        adapter = LoopAdapter()
        run = adapter.create_run(
            agent="claude",
            ralph_dir=worktree,
            ralph_file=worktree / "RALPH.md",
            commands=[],
            quality_gates=None,
        )

        verifier = run.config.completion_verifier
        verdict = verifier()

        assert verdict.ok is False
        assert "quality_gates" in verdict.feedback
