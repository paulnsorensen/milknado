# PR 520 integration UI evidence

These captures show actual terminal and browser presentation with fixed offline data.
Baseline checkout: `e734344fdc953dbc116e9f9bdc5f83cb6974c98e`.
Current terminal presentation matches `dbddcad986674232746f668d9abd4df6f696c4ad`; later Cure changes do not alter these view files.
Current browser captures use the rebuilt `index-CSuvHvHG.js` dashboard and reviewed proposal API.
All 90 PNGs are opened and inspected before publication.
The [manifest](./manifest.json) records their exact hashes.

## Terminal comparisons

Both sides use the same goal, run, status, elapsed time, output, keys, dimensions, and `reference-dark` renderer profile.
The clock shows 12:00:00.
The fixture starts no worker and exercises no provider credentials, durable graph, or control authority.
The real `WatchApp` and `ExecutionApp` render the fixture.
Before `c` has no coordinator panel; the same key leaves the baseline main view open.

| State and size | Before | After |
| --- | --- | --- |
| watch-normal-80x24 | [before](./terminal/before/watch-normal-80x24.png) | [after](./terminal/after/watch-normal-80x24.png) |
| watch-error-80x24 | [before](./terminal/before/watch-error-80x24.png) | [after](./terminal/after/watch-error-80x24.png) |
| watch-empty-80x24 | [before](./terminal/before/watch-empty-80x24.png) | [after](./terminal/after/watch-empty-80x24.png) |
| run-normal-80x24 | [before](./terminal/before/run-normal-80x24.png) | [after](./terminal/after/run-normal-80x24.png) |
| run-error-80x24 | [before](./terminal/before/run-error-80x24.png) | [after](./terminal/after/run-error-80x24.png) |
| run-empty-80x24 | [before](./terminal/before/run-empty-80x24.png) | [after](./terminal/after/run-empty-80x24.png) |
| watch-help-80x24 | [before](./terminal/before/watch-help-80x24.png) | [after](./terminal/after/watch-help-80x24.png) |
| run-help-80x24 | [before](./terminal/before/run-help-80x24.png) | [after](./terminal/after/run-help-80x24.png) |
| watch-coordinator-80x24 | [before](./terminal/before/watch-coordinator-80x24.png) | [after](./terminal/after/watch-coordinator-80x24.png) |
| attached-watch-coordinator-80x24 | [before](./terminal/before/attached-watch-coordinator-80x24.png) | [after](./terminal/after/attached-watch-coordinator-80x24.png) |
| run-confirm-80x24 | [before](./terminal/before/run-confirm-80x24.png) | [after](./terminal/after/run-confirm-80x24.png) |
| watch-normal-120x40 | [before](./terminal/before/watch-normal-120x40.png) | [after](./terminal/after/watch-normal-120x40.png) |
| watch-error-120x40 | [before](./terminal/before/watch-error-120x40.png) | [after](./terminal/after/watch-error-120x40.png) |
| watch-empty-120x40 | [before](./terminal/before/watch-empty-120x40.png) | [after](./terminal/after/watch-empty-120x40.png) |
| run-normal-120x40 | [before](./terminal/before/run-normal-120x40.png) | [after](./terminal/after/run-normal-120x40.png) |
| run-error-120x40 | [before](./terminal/before/run-error-120x40.png) | [after](./terminal/after/run-error-120x40.png) |
| run-empty-120x40 | [before](./terminal/before/run-empty-120x40.png) | [after](./terminal/after/run-empty-120x40.png) |
| watch-help-120x40 | [before](./terminal/before/watch-help-120x40.png) | [after](./terminal/after/watch-help-120x40.png) |
| run-help-120x40 | [before](./terminal/before/run-help-120x40.png) | [after](./terminal/after/run-help-120x40.png) |
| watch-coordinator-120x40 | [before](./terminal/before/watch-coordinator-120x40.png) | [after](./terminal/after/watch-coordinator-120x40.png) |
| attached-watch-coordinator-120x40 | [before](./terminal/before/attached-watch-coordinator-120x40.png) | [after](./terminal/after/attached-watch-coordinator-120x40.png) |
| run-confirm-120x40 | [before](./terminal/before/run-confirm-120x40.png) | [after](./terminal/after/run-confirm-120x40.png) |

The 80×24 and 120×40 views have readable status, focus, key hints, error, help, and run confirmation states.
The main view documents a 60×18 minimum; 40×15 shows its resize notice.
Search and filter states do not exist in these terminal views.
The observer has no destructive confirmation; the run view supplies that comparison.
The known compact attached-input hint limitation remains a recorded low deferral.
These captures do not prove that owner-validated input succeeds.
The renderer profile does not prove palette fidelity.

## Browser comparisons and limits

Both sides use the same underlying goal and graph state, viewport, dark theme, provider, and fixed source time.
The source time is `2026-01-01T12:00:00+00:00`.
Session and command IDs are deterministic.
The real HTTP, SQLite, authentication, commands, and browser UI run locally.
A planner boundary fixture supplies one task; no live worker or provider authority runs.

The baseline lacks intake and proposal controls.
Its proposal-labelled captures show the ordinary graph for the same goal, not equivalent proposal actions or UI.
Accepted comparisons have one task child on both sides.
Stale comparisons have one concurrent task child on both sides.
Other comparisons have no child; intake and coordinator-unavailable have no goal.
These feature-absence comparisons do not claim identical interaction state where the feature does not exist.

| State and viewport | Baseline | Current |
| --- | --- | --- |
| intake · 1440x900 | [baseline graph](./browser/before-intake-1440x900.png) | [current UI](./browser/after-intake-1440x900.png) |
| intake · 390x844 | [baseline graph](./browser/before-intake-390x844.png) | [current UI](./browser/after-intake-390x844.png) |
| pending · 1440x900 | [baseline graph](./browser/before-pending-1440x900.png) | [current UI](./browser/after-pending-1440x900.png) |
| pending · 390x844 | [baseline graph](./browser/before-pending-390x844.png) | [current UI](./browser/after-pending-390x844.png) |
| accepted · 1440x900 | [baseline graph](./browser/before-accepted-1440x900.png) | [current UI](./browser/after-accepted-1440x900.png) |
| accepted · 390x844 | [baseline graph](./browser/before-accepted-390x844.png) | [current UI](./browser/after-accepted-390x844.png) |
| rejected · 1440x900 | [baseline graph](./browser/before-rejected-1440x900.png) | [current UI](./browser/after-rejected-1440x900.png) |
| rejected · 390x844 | [baseline graph](./browser/before-rejected-390x844.png) | [current UI](./browser/after-rejected-390x844.png) |
| planner-unavailable · 1440x900 | [baseline graph](./browser/before-planner-unavailable-1440x900.png) | [current UI](./browser/after-planner-unavailable-1440x900.png) |
| planner-unavailable · 390x844 | [baseline graph](./browser/before-planner-unavailable-390x844.png) | [current UI](./browser/after-planner-unavailable-390x844.png) |
| stale · 1440x900 | [baseline graph](./browser/before-stale-1440x900.png) | [current UI](./browser/after-stale-1440x900.png) |
| stale · 390x844 | [baseline graph](./browser/before-stale-390x844.png) | [current UI](./browser/after-stale-390x844.png) |
| applying · 1440x900 | [baseline graph](./browser/before-applying-1440x900.png) | [current UI](./browser/after-applying-1440x900.png) |
| applying · 390x844 | [baseline graph](./browser/before-applying-390x844.png) | [current UI](./browser/after-applying-390x844.png) |
| recovery-unavailable · 1440x900 | [baseline graph](./browser/before-recovery-unavailable-1440x900.png) | [current UI](./browser/after-recovery-unavailable-1440x900.png) |
| recovery-unavailable · 390x844 | [baseline graph](./browser/before-recovery-unavailable-390x844.png) | [current UI](./browser/after-recovery-unavailable-390x844.png) |
| coordinator-unavailable · 1440x900 | [baseline graph](./browser/before-coordinator-unavailable-1440x900.png) | [current UI](./browser/after-coordinator-unavailable-1440x900.png) |
| coordinator-unavailable · 390x844 | [baseline graph](./browser/before-coordinator-unavailable-390x844.png) | [current UI](./browser/after-coordinator-unavailable-390x844.png) |

Proposal-focused captures scroll the cockpit internally to show status and approval controls.
That scroll omits its header and part of the visible command-result JSON.
The supplemental captures reset the cockpit scroll and show the header.
They do not expose the entire scrollable cockpit at once.
Mobile supplements have a taller page; they are not matched-size comparison pairs.
The raw JSON result remains actual UI behavior; the fixture does not hide it.
HTTP 500 responses, page errors, unexpected console errors, and incorrect graph state fail the fixture.
Coordinator-unavailable asserts one URL-attributed HTTP 409 console error and the authenticated response body.
Its unavailable capability hides the cockpit; no visible notice appears.
Stale uses a real graph revision change.
Applying seeds a valid proposal transition; it does not simulate a worker crash.
Focused captures show manual recovery text; header supplements do not necessarily show that text.
Repeated Chromium PNG bytes can vary slightly; hashes identify these files, not byte-stable reruns.

- [after-accepted-1440x900-full.png](./browser/after-accepted-1440x900-full.png)
- [after-accepted-390x844-full.png](./browser/after-accepted-390x844-full.png)
- [after-pending-1440x900-full.png](./browser/after-pending-1440x900-full.png)
- [after-pending-390x844-full.png](./browser/after-pending-390x844-full.png)
- [after-rejected-1440x900-full.png](./browser/after-rejected-1440x900-full.png)
- [after-rejected-390x844-full.png](./browser/after-rejected-390x844-full.png)
- [after-applying-1440x900-full.png](./browser/after-applying-1440x900-full.png)
- [after-applying-390x844-full.png](./browser/after-applying-390x844-full.png)
- [after-stale-1440x900-full.png](./browser/after-stale-1440x900-full.png)
- [after-stale-390x844-full.png](./browser/after-stale-390x844-full.png)

## Reproduce browser captures

Use the same script from the current checkout for both source checkouts.
Set `CHECKOUT` and `SIDE` to the baseline or current checkout and `before` or `after`.
Install the project prerequisites with `just install`.

```sh
AFTER=/path/to/current-checkout
CHECKOUT=/path/to/source-checkout
SIDE=after
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CHECKOUT/src:$CHECKOUT" \
  "$AFTER/.venv/bin/python" \
  "$AFTER/docs/tui-captures/pr-520-integration/capture-browser.py" \
  --revision "$SIDE" --checkout "$CHECKOUT" --output-dir /tmp/pr520-browser
```

## Reproduce a terminal cell

Install `agent-tty` and load its bundled skill before automation.
Use an isolated home and check its renderer with `doctor --json`.
Set `VIEW` to `watch` or `run`.
Set `STATE` to `normal`, `error`, or `empty`.
Use 80×24 or 120×40 dimensions.

```sh
AFTER=/path/to/current-checkout
CHECKOUT=/path/to/source-checkout
CAPTURE_HOME=$(mktemp -d)
COLS=80
ROWS=24
VIEW=watch
STATE=normal
agent-tty --home "$CAPTURE_HOME" doctor --json
SESSION=$(agent-tty --home "$CAPTURE_HOME" create --cols "$COLS" --rows "$ROWS" \
  --cwd "$CHECKOUT" --env "PYTHONPATH=$CHECKOUT/src:$CHECKOUT" \
  --env PYTHONDONTWRITEBYTECODE=1 --json -- \
  "$AFTER/.venv/bin/python" \
  "$AFTER/docs/tui-captures/pr-520-integration/capture-terminal.py" \
  --view "$VIEW" --state "$STATE" | jq -r '.result.sessionId')
agent-tty --home "$CAPTURE_HOME" wait "$SESSION" --text Milknado --timeout 10000 --json
agent-tty --home "$CAPTURE_HOME" snapshot "$SESSION" --format text --json
agent-tty --home "$CAPTURE_HOME" screenshot "$SESSION" --profile reference-dark --hide-cursor --json
agent-tty --home "$CAPTURE_HOME" destroy "$SESSION" --json
```

The example uses `jq` to read the returned session ID.
Add `--attached` for attached watch.
Before the snapshot, use `batch` with a key and an observable wait for overlays.
Use `?` for help, `c` for coordinator status, and `q` for run confirmation.

```sh
agent-tty --home "$CAPTURE_HOME" batch "$SESSION" \
  '[{"sendKeys":["c"]},{"wait":{"screenStableMs":1000}}]' --json
```

The terminal publication script extracts two snapshot helpers from the original capture fixture.
Snapshot equality passes for all three states on both source checkouts.
The browser fixture uses the public `CoordinatorServices` import and nine source-grounded states.
Its state helper asserts the expected unavailable response without suppressing other console errors.
Both scripts satisfy the written size limits.
