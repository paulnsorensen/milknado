# Architecture gates — slice guard and depth ratchet decision

The slice guard (`scripts/check_slices.py`) and the depth ratchet (`scripts/check_depth.py`) read one human-owned `slices.toml` through `scripts/slice_map.py`.
They are a port of the Python `ast` gates in paulnsorensen/skillz-that-grillz (PR 530).
Those gates port two TypeScript prototypes in paulnsorensen/sliced-bread-architecture: sliced-bread-guard (PR 69) and sliced-bread-depth-ratchet (PR 71).
`just check-llm` and `just build-ci` run both gates. Their pytest suites are `tests/test_check_slices.py` and `tests/test_check_depth.py`.

### ADR-architecture-gates-001: Port the skillz-that-grillz Python gates  [status: accepted]

- **Context:** The upstream tools support TypeScript only. skillz-that-grillz already has a Python `ast` port with the upstream rule IDs (SB-00 to SB-07), metric keys (M08, M09, M10, M12), and file formats.
- **Decision:** Copy that port into `scripts/`. Adapt it to the milknado style: 99-column lines, at most four parameters, pytest suites, and `python -m scripts.<tool>` so that the modules import `scripts.slice_map`. The metric definitions move to `scripts/depth_metrics.py`, so each file stays readable.
- **Consequences:** The `packages/*/src` roots check is gone, because milknado has one root, `src`. The import-linter contracts in `pyproject.toml` stay. They overlap SB-01 and SB-02, and the overlap is intentional.

### ADR-architecture-gates-002: Domain slices are slices; common, adapters, and entry points are spine  [status: accepted]

- **Context:** The sliced-bread-guard skill says to make `app/`, `entrypoints/`, `adapters/`, and `common/` spine entries.
- **Decision:** Each `domains/<slice>` package except `common` is a `[[slice]]` with its `__init__.py` as the crust. `loop` and `plugins` are slices too. `common`, `adapters`, `app`, `cli`, `mcp`, and `web` are spine nodes. The `root` spine node holds `milknado/__init__.py`, `domains/__init__.py`, `process_logging.py`, and `project.py`.
- **Consequences:** `may_depend_on` records the node edges at adoption, so SB-02 starts at zero. The node graph has no cycle. Tests are outside `[policy] roots`, so SB-05 has no counterpart.

### ADR-architecture-gates-003: Adopt with a baseline, then remove it with a hill climb  [status: accepted]

- **Context:** At adoption the guard found 33 SB-01 crust bypasses: 32 from `adapters` into `loop` internals and one each from `cli` into `graph`, `coordinator` into `loop`, and `github` into `wiki`.
- **Decision:** `sbguard-baseline.json` holds the 33 entries, so the gate passes. The hill-climb thread `slice-guard-baseline` drives the entry count to zero. Each fix runs `check_slices baseline --shrink` in the same change, because `check --strict` fails on a stale entry.
- **Consequences:** A new violation fails at once; a fixed one cannot pay for a new one. The depth ratchet runs in `block` mode from its adoption measurement. `M12.unused_exports` (53 at adoption) and the `batching` crust logic (7 statements) are the next depth targets.

### ADR-architecture-gates-004: SB-07 reads the baseline at the base commit  [status: accepted]

- **Decision:** `check --base auto` uses `$GATE_BASE`, else `HEAD^1` in GitHub Actions, else the merge base with `origin/main`. The CI `build` job checks out with `fetch-depth: 2`; the `web` job already fetches full history.
- **Consequences:** A clone without `origin/main` must set `GATE_BASE`, or the guard exits 2.

### ADR-architecture-gates-005: Move worker-protection adapters into loop  [status: accepted]

- **Context:** 31 of the adoption bypasses imported `loop` internals from `adapters`. The fixes were to widen the `loop` crust by about 21 names, to re-map `loop` as spine, or to move the consumers into `loop`.
- **Decision:** The owner chose to move the consumers. `loop` may depend on `graph` and `dispatch`, and the `loop` M09 cap may rise for the moved facades only. Iteration 2 moved `process.py` (now `loop/_process_adapter.py`), `_protection.py`, `_tmux_exec.py`, `_loop_lifeline.py` (now `loop/_lifeline_main.py`), `_loop_worker_evidence.py`, `_loop_worker_release.py`, and `coordinator_worker_recovery.py` (now `loop/_worker_recovery.py`). The `loop` crust gained `ProcessAdapter`, `CoordinatorWorkerRecovery`, and `loop_protection` (M09 9 to 12). `adapters/__init__.py` re-exports the two adapters, so `cli` needs no `loop` edge.
- **Consequences:** The baseline fell from 31 to 19 entries. `adapters/loop.py` cannot move while `execution` imports `RunStatus` and `CompletionVerdict` from `loop`, because `loop -> execution` would be a cycle. `coordinator_turns.py` has the same problem through `coordinator`. Those moves need another owner decision.

_Source: port of skillz-that-grillz PR 530; owner decisions 2026-10-25 · Updated: 2026-10-25_
