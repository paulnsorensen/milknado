status: ok
next: done
artifact: .cheese/press/vendor-loop-engine.md
Vendors the ralphify fork engine (fork SHA ec494875) into src/milknado/loop/ + tests/loop/, renames the RalphPort/RalphifyAdapter seam to LoopPort/LoopAdapter, drops the git dependency, and removes the doctor ralphify probe.

# Age Report — vendor-loop-engine

## Orientation
Vendor-in of the ralphify fork closure into `src/milknado/loop/` (engine, agent layer, events, run types, manager, per-CLI adapters, hooks, wind-down shim) plus its closure-scoped test suite under `tests/loop/`. Non-vendored changes: seam rename across `protocols.py`/adapters/CLI/domains, git dep + `allow-direct-references` removal, `uv.lock` regen, doctor probe deletion, the `"execution (loop)"` label, a wiki edit, and 16 press hardening tests. Verbatim vendored internals are out of scope per the user-approved decision; non-vendored changes reviewed at full strictness.

## Press findings
None unresolved. Press added `tests/test_vendor_loop_seams.py` (16 tests) covering crust `__all__`, seam rename, dep removal, and label; all green under `just check-llm`. The only press finding was 3 ruff-fixable lint nits, already fixed via `just lint-fix` before the gate.

## Low

1. **[deslop:low]** `tests/test_vendor_loop_seams.py:210,220` — `tmp_path: pytest.Path` is a wrong type annotation; `pytest` exports no `Path`. Runtime-inert because `from __future__ import annotations` (line 13) defers evaluation, which is why tests pass, but Pyright flags it and `typing.get_type_hints()` on these functions would raise `AttributeError`. The file also has no `from pathlib import Path` import. Non-vendored test code, full strictness.
   - location: module · fix-cost-now: contained · fix-cost-later: contained
   - recommendation: change both annotations to `Path` and add `from pathlib import Path`.

2. **[spec:low]** `src/milknado/loop/engine.py:67` — acceptance item AC3 (`grep -ri ralphify src/` matches only the attribution docstring) is not literally met: the vendored `_CREDIT_INSTRUCTION` string `"Co-authored-by: Ralphify <noreply@ralphify.co>"` is a 4th `ralphify` match in `src/`. Verified byte-identical to fork SHA ec494875 (`/tmp/ralphify-vendor/src/ralphify/engine.py:68`) — it is functional vendored behaviour (a commit-trailer instruction), not a rename leftover. Reporting it as a code change would contradict the spec's verbatim mandate and the non-goal "loop-control features ride along exactly as they exist on fork main." Recorded as an acceptance-wording mismatch, not an actionable defect.
   - location: contract · fix-cost-now: contained · fix-cost-later: contained
   - recommendation: no code change; if AC3 must read clean, tighten the acceptance text to "matches only the attribution docstring and the vendored credit-trailer string" — a doc edit, deferrable to the planned follow-up PR. Do not edit vendored engine code to satisfy a literal grep.

## Acceptance checklist (mechanically verified)
- AC1 ✅ `from milknado.loop import EventType, QueueEmitter, RunConfig, RunManager, RunStatus` works; `__all__` is exactly those five (verified by import).
- AC2 ✅ `loop/` contains no `cli.py`, `_console_emitter.py`, `_keypress.py`, `_brand.py` (listed dir).
- AC3 ⚠️ `grep -ri ralphify src/` matches the 3-line attribution docstring PLUS the vendored credit string at `engine.py:67` — see Finding 2. Verbatim, behaviour-preserving; covered by the verbatim decision.
- AC4 ✅ `pyproject.toml` has no `git+`/`allow-direct-references`/`ralphify`; `uv.lock` has zero `ralphify` (regen confirmed).
- AC5 ✅ `just check-llm` PASS (lint + format + tests + project & diff coverage ≥95%).
- AC6 ✅ `uv build --wheel` succeeds; wheel `Requires-Dist` has zero direct-URL refs.
- AC7 ✅ Rename complete — no `RalphifyAdapter`/`RalphPort` in production source; only in tests asserting their absence and in notes/press docs.
- AC8 ✅ `tests/test_adapters_loop.py` (renamed, imports rewritten) passes under the gate.

Verdict: 7/8 fully met; AC3 is a literal-grep wording mismatch protected by the verbatim mandate (not a code defect).

## Verified clean
- correctness — `adapters/loop.py` is a true rename of `adapters/ralphify.py` (5-line delta: import swap + class name; body unchanged, confirmed via `--find-renames`). Vendored files spot-checked against fork ec494875: `_promise.py`/`engine.py` differ only by ruff reformatting and the behaviour-preserving `timezone.utc`→`datetime.UTC` modernization (Python 3.11+ alias); no logic divergence.
- security — no new tainted-input paths, no secrets, no unsafe parsing in non-vendored changes; vendored subprocess layer is verbatim and out of scope.
- encapsulation — seam preserved: domains import `LoopPort`/`LoopAdapter`, never `milknado.loop` directly; crust exports exactly the five promised names.
- assertions — seam tests use exact `__all__` set equality, `hasattr` absence checks, `ImportError`/`PackageNotFoundError` by type, and exact label strings; the dropped `TestDoctorMissingTool` correctly removed a test of behaviour that no longer exists; no orphaned helpers (`_PKG_VERSIONS`/`PackageNotFoundError` removal left no dangling refs).
- nih — vendoring is the explicit, approved decision (PyPI rejects direct-URL deps); `pyyaml` remains a direct dep so vendored `_frontmatter` resolves.
- efficiency, telemetry — no changes of concern in the non-vendored diff.
- complexity, deslop (vendored internals) — out of scope per the verbatim decision (oversized files / private naming are spec-documented follow-ups).

## Confidence
certain — acceptance items mechanically verified (import, dir listing, regex, `just check-llm`, `uv build` METADATA, rename detection); vendored faithfulness spot-checked against the local fork clone at `/tmp/ralphify-vendor` (SHA ec494875, == fork main HEAD). Not exhaustively diffed every vendored file byte-for-byte; sampled `engine.py`, `_promise.py`, `_resolver.py`, `_runner.py` and confirmed only ruff-formatting deltas.

## Next step
Auto-fixing the recommended set via `/cure` — Finding 1 (cheap contained low) is the only actionable item; Finding 2 is a no-code-change doc note.

## Re-age pass 2 (scoped: tests/test_vendor_loop_seams.py)
Cure pass 1 applied finding 1 in commit `fd3fd95`: `from pathlib import Path` added,
both `tmp_path: pytest.Path` annotations corrected to `Path`. Diff contains exactly the
prescribed fix and nothing else; 16/16 tests pass, `just check-llm` PASS. No new findings —
no finding meets the medium+ floor. Chain clean: `next: done`.
