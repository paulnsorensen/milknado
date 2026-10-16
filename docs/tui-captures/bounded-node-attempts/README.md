# Bounded node attempts: TUI evidence

This bundle has 24 inspected PNGs. The eight before/after pairs compare presentation only. Both sides use the same synthetic snapshots, fixed 12:00:00 clock, reference-dark renderer, focus, and dimensions. Their output is nearly identical because the TUI renderer did not change.

The eight after-only projected captures use a real SQLite graph and task-attempt ledger. They also use `RunLoop.state()` and `ExecutionController._project_snapshot()`. Their settings are `max_node_attempts=3` and `dispatch_max_retries=1`. Do not compare them with the synthetic baseline. Graph navigation and totals differ.

## Paired presentation captures

| Surface | State | 120x40 before / after | 80x24 before / after |
|---|---|---|---|
| run | retry error | [before](before-run-retry-error-120x40.png) / [after](after-run-retry-error-120x40.png) | [before](before-run-retry-error-80x24.png) / [after](after-run-retry-error-80x24.png) |
| run | unknown start | [before](before-run-unknown-start-120x40.png) / [after](after-run-unknown-start-120x40.png) | [before](before-run-unknown-start-80x24.png) / [after](after-run-unknown-start-80x24.png) |
| watch | retry error | [before](before-watch-retry-error-120x40.png) / [after](after-watch-retry-error-120x40.png) | [before](before-watch-retry-error-80x24.png) / [after](after-watch-retry-error-80x24.png) |
| watch | unknown start | [before](before-watch-unknown-start-120x40.png) / [after](after-watch-unknown-start-120x40.png) | [before](before-watch-unknown-start-80x24.png) / [after](after-watch-unknown-start-80x24.png) |

## After-only producer captures

| Surface | State | 120x40 | 80x24 |
|---|---|---|---|
| run | retry error | [projected](after-projected-run-retry-error-120x40.png) | [projected](after-projected-run-retry-error-80x24.png) |
| run | unknown start | [projected](after-projected-run-unknown-start-120x40.png) | [projected](after-projected-run-unknown-start-80x24.png) |
| watch | retry error | [projected](after-projected-watch-retry-error-120x40.png) | [projected](after-projected-watch-retry-error-80x24.png) |
| watch | unknown start | [projected](after-projected-watch-unknown-start-120x40.png) | [projected](after-projected-watch-unknown-start-80x24.png) |

## Reproduce

Run these commands from the implementation worktree. The baseline commit does not contain the bundle scripts. Use this bundle's absolute fixture path for both sides.

```sh
CURRENT=$PWD
BASELINE=/absolute/path/to/checkout-at-eadb011
test "$(git -C "$BASELINE" rev-parse HEAD)" = eadb011bfda325a6dddd8e7a4bd05cc0629529d3
PY="$CURRENT/.venv/bin/python"
FIX="$CURRENT/docs/tui-captures/bounded-node-attempts/fixture.py"
APP_SOURCE="$BASELINE"  # Use "$CURRENT" for the matching after capture.
H=$(mktemp -d /tmp/issue495-agenttty.XXXXXX)
agent-tty --home "$H" doctor --json
SID=$(agent-tty --home "$H" create --json --cols 120 --rows 40 --shell /bin/zsh --idle-timeout-ms 1800000 | jq -r '.result.sessionId')
agent-tty --home "$H" run "$SID" "cd '$CURRENT' && PYTHONPATH='$APP_SOURCE/src' '$PY' '$FIX' run retry-error" --no-wait --json
agent-tty --home "$H" wait "$SID" --text 'Bounded node attempts' --timeout 20000 --json
agent-tty --home "$H" wait "$SID" --screen-stable-ms 800 --timeout 20000 --json
agent-tty --home "$H" snapshot "$SID" --format text --json
agent-tty --home "$H" screenshot "$SID" --profile reference-dark --json
agent-tty --home "$H" resize "$SID" --cols 80 --rows 24 --json
agent-tty --home "$H" wait "$SID" --screen-stable-ms 800 --timeout 20000 --json
agent-tty --home "$H" screenshot "$SID" --profile reference-dark --json
agent-tty --home "$H" destroy "$SID" --json
```

Repeat with both surfaces (`run`, `watch`) and states (`retry-error`, `unknown-start`). Copy each screenshot artifact to its linked file name. For after-only projected captures, set `APP_SOURCE="$CURRENT"` and `FIX="$CURRENT/docs/tui-captures/bounded-node-attempts/projected_fixture.py"`. Keep the same interpreter, renderer, dimensions, and input state.

## Source and hash manifest

- Baseline commit: `eadb011bfda325a6dddd8e7a4bd05cc0629529d3`.
- Baseline and current `run_tui.py` Git blob: `f054b74342c958da33701d2bce7a955ee9908171`.
- Baseline and current `watch_tui.py` Git blob: `c4baadc174f7cdc3d30dd364c5210128537995e1`.
- Current `run.py` Git blob at capture: `815f48f0bddcafaa5975bd49d5b717caec4a2760`.
- Current `_task_attempts.py` Git blob at capture: `15454341248f73268073010ba1c0aaadf481396d`.
- Current `run_loop/__init__.py` Git blob at capture: `c2ed85f8d605ef5670a75538677bfc148b7fbdcd`.
- `SHA256SUMS` covers both fixture scripts and all 24 PNGs.

## Resume verification

The resumed review opens all 24 retained PNGs and verifies all 26 manifest entries.
Five of the eight synthetic pairs have identical file bytes. All eight have matching visible layout.
A bounded producer probe confirms retry count 1, active attempt 2/3, and unknown-start count 0 with attempt 1/3.
The fixtures and renderer remain unchanged. These retained images do not prove live-worker behavior.

## Inspection and limits

All 24 PNGs were opened. No panel overlap, lost focus, missing key hints, or hidden status appeared. Header and description ellipses are intentional. The compact projected graph overview omits the attempt cap. The compact synthetic retry list shows 2/3, while its first-attempt list omits 1/3. The standard detail shows 2/3 for retry and 1/3 for unknown start. The projected retry view shows the failed first attempt and active second attempt. The projected unknown-start view remains held without spending a failure count. The paired synthetic text does not prove baseline retry behavior. No fixture launches a live worker. The projected capture does not prove retry admission after a real process failure. The renderer checks layout, not palette fidelity. The original capture environment reports unsupported product-native preview. This resume does not test preview support.
