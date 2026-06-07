# PR #137 — Loop Vendor-In Execution Record

Date: 2026-06-06 · Decision rationale: [loop-vendor-in-decision](./loop-vendor-in-decision.md).

PR #137 vendored the ralphify fork's engine closure (fork SHA `ec494875`, == fork main HEAD at vendor time) into `src/milknado/loop/` + `tests/loop/`, renamed the `RalphPort`/`RalphifyAdapter` seam to `LoopPort`/`LoopAdapter`, dropped the `git+` dep and `allow-direct-references`, and removed the doctor ralphify probe.

## Verification record (age review, all mechanically checked)

- Crust contract: `from milknado.loop import EventType, QueueEmitter, RunConfig, RunManager, RunStatus` works; `__all__` is exactly those five.
- No CLI/TUI modules vendored (`cli.py`, `_console_emitter.py`, `_keypress.py`, `_brand.py` absent).
- `pyproject.toml` and `uv.lock` have zero `ralphify`/`git+` references; `uv build --wheel` METADATA has no direct-URL deps.
- `adapters/loop.py` is a true rename of `adapters/ralphify.py` (5-line delta: import swap + class name; body unchanged, confirmed via `--find-renames`).
- Seam preserved: domains import `LoopPort`/`LoopAdapter`, never `milknado.loop` directly.
- Gate: `just check-llm` PASS (project + diff coverage ≥95%); 16 seam-hardening tests added by press (`tests/test_vendor_loop_seams.py`).

## Vendored-faithfulness audit scope

Faithfulness to fork `ec494875` was **spot-checked, not byte-diffed file-by-file**: sampled `engine.py`, `_promise.py`, `_resolver.py`, `_runner.py`. Only divergences found: ruff reformatting and the behaviour-preserving `timezone.utc` → `datetime.UTC` modernization (Python 3.11+ alias). No logic divergence in the sample; unsampled files presumed clean but unverified.

## Known deviation: the AC3 grep

The acceptance item "`grep -ri ralphify src/` matches only the attribution docstring" is **not literally met**: the vendored `_CREDIT_INSTRUCTION` string in `loop/engine.py` (`"Co-authored-by: Ralphify <noreply@ralphify.co>"`) is a fourth match. It is byte-identical functional fork behaviour (a commit-trailer instruction), protected by the verbatim mandate — an acceptance-wording mismatch, not a defect. **Do not edit vendored engine code to satisfy a literal grep**; if AC3 must read clean, tighten the acceptance wording instead.

## Review-triage notes (affinage, post-open)

- Copilot claimed a missed-wakeup race in `RunManager.wait_for_any`/`wait_for_all` (Condition not tied to the predicate lock). **False**: the predicate re-check and `wait()` both run inside one `with self._done:` block, and the worker's `notify_all()` acquires the same condition lock — `Condition.wait()` releases it atomically, so no window exists. Real-but-minor note: `_finished_ids` acquires `self._lock` nested inside `self._done` — safe only while no path acquires the two locks in the reverse order. Keep that ordering invariant if touching `manager.py`.
- 31 CodeQL note-level alerts on the vendored code were all false positives: deliberate documented broad-suppresses in `_agent.py` cleanup/fanout paths, `...` Protocol stub bodies (`hooks.py`, `adapters/_protocol.py`, `_events.py`), pytest dual-import monkeypatching idiom in tests, and a misfired unused-global on the `_counter_write_failure_logged` warning latch. Expect the same alerts on any future touch of these files.
- One real Copilot find: `tests/test_plugin_manifests.py` relied on PyYAML's YAML-1.1 coercion of the workflow `on:` key to boolean `True` (`wf[True]`); fixed to accept both YAML-1.1 and 1.2 shapes.

## Follow-ups on record

- **File-cap conformance split** (spec-documented follow-up PR): `_agent.py`, `engine.py`, `_events.py` exceed the 300-line cap; vendored verbatim deliberately, split deferred.
- **Archive `paulnsorensen/ralphify`** after merge (manual step) — the fork's reason to exist ended with the vendor-in.
