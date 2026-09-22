# Execution repair captures

The archive contains 30 matched pairs: 60 PNG files, 2,715,955 bytes.
Each pair uses the same terminal dimensions, fixture data, renderer, and interaction state.
The visible wall clock and cursor blink differ between capture times.

These are archived repair captures, not a new live-provider run or a release approval.
The images remain unchanged. No pixels are generated, redacted, or recolored.
All 60 images are opened and inspected on 2026-09-22.
The visible data is synthetic and contains no observed secrets.

## Integrated verification

The [current manifest](current/manifest.json) records ten additional, inspected screenshots.
Eight fixture screenshots cover Run and Watch at 80×24 and 120×40, in main and empty-graph states.
They use controller commit `a24af68` and the TUI source committed with this evidence.
Use the reproduction commands below with this checkout to reproduce these states.
The fixture still substitutes worker I/O; these eight frames do not prove provider execution.

Two separate frames show [a real OMP run](current/live-run-active-120x40.png) and [its completed observer](current/live-watch-completed-120x40.png).
The disposable task writes exactly `milknado-live-proof\n` to `proof.txt`.
Its configured gate checks those exact bytes.
Both graph nodes finish as `done`; the run finishes as `done`, exit code 0, with successful merge-back.
The worker worktree is removed, and the observer exits without changing the completed graph.
The run uses an isolated project and credential state root, not the operator's live graph.
The worker's MCP result deposit fails because its server is disconnected.
This evidence proves execution, gate completion, merge-back, and observation, not MCP result deposition.

The current frames use `reference-dark` and contain no observed secrets.
The archived matched pairs below remain unchanged.

## Review follow-up

The [follow-up manifest](review-followup/manifest.json) records 14 matched pairs and their interaction outcomes.
All 28 PNGs use `reference-dark`, `ghostty-web`, and the same fixture and terminal size within each pair.
Clock text and header refresh timing can differ.
The baseline source is commit `0c5222b6189af5c956c0e20f36020bd2a7c8c0d5`.
The after frames use the review fixes committed with this evidence.
Later type annotations and casts do not change the captured behavior.

- Empty graph: Run and Watch retain active and terminal rows at 80×24 and 120×40.
  Earlier empty-graph captures clear those rows and cannot prove retained-run navigation.
- Permission: Run and writable Watch preserve the focused `second: Approve beta` choice during a stale refresh.
  The `attached` filenames identify writable Watch, not the read-only observer.
- Footer owner: a synthetic focused widget exposes an `f6 Owner proof` binding.
  The real mouse click reaches that widget after the fix.
  Its diagnostic banner covers the usual header in these component captures.
- Confirmation: both sizes retain the force-stop overlay and its visible key hints.
  The real modal blocks mouse access to the underlying footer before and after the fix.
  These two pairs prove layout, not modal-footer dispatch.

The manifest records outcomes immediately after the scripted interaction.
The helper then publishes a readiness snapshot before capture.
Screenshots can therefore show the next refresh, not the earlier selected-row value.
All worker I/O remains synthetic.
The original 60 archive images and ten integrated captures remain unchanged.

Use the [capture helper](review-followup/capture.py) with either source checkout:

```text
python docs/tui-captures/execution-repair/review-followup/capture.py SOURCE MODE STATE PROOF_JSON
```

Run that command inside an agent-tty session at 80×24 or 120×40.
Use `run` or `watch` for `empty-graph` and `footer-owner`.
Use `run` or `attached` for `permission`.
Use `run` for `confirmation`.
Wait for `READY`, then capture the stable screen with the commands below.
The helper writes the interaction outcome to `PROOF_JSON`.

## Provenance

- Before: the 2026-09-20 audit at `d7c675f7552e76c452d4064484a549e0f8ea7d7c`.
- The two run empty-graph before images come from the supplemental baseline capture.
  That capture records the original checkout but does not pin its exact commit.
- After: the reconstructed TUI worktree based on `ec886ba4f151937f7b5e8f39278a85e36f5d6905`.
  These images precede the final queued-navigation verification.
- Environment: macOS arm64, Python 3.13.13, Textual 8.2.8, agent-tty 0.5.0, Node 24.18.1.
- Renderer: `reference-dark`, with its default cursor rendering.
- Terminal sizes: 120×40 and 80×24.
- Pixel sizes: 1080×640 and 720×384.
- [manifest.json](manifest.json) records each PNG checksum, size, and terminal dimensions.
- [fixture.py](fixture.py) preserves the capture fixture and uses the existing adjacent fixture data.

The fixture replaces worker I/O, not Textual widgets.
It creates disposable Git data through the structured-session fixture.
It does not use the operator's live graph or a real provider.
These frames do not establish fidelity to a user's terminal palette.

## Inspection

| State | Observed result |
|---|---|
| Standard main | After uses graph-label ellipses and contextual footer hints. |
| Compact main | After wraps run controls across readable footer rows. |
| Empty and empty graph | After exposes the run selector and removes unavailable session and mutation hints. |
| Help | Text remains inside the scrollable modal at both sizes. |
| Error | Listener status remains visible. Standard mode also shows the worker error pane. |
| Confirmation | The force-stop prompt and confirmation keys remain visible at both sizes. |
| Session input | Input focus remains visible. Compact after shows more transcript text. |
| Owner unavailable | After removes the session-input hint when no owner accepts input. |
| Watch | After omits mutation and session-input controls. |
| Search | The query and Theme result remain visible. |

Residual observation: compact search and confirmation after-images still show an extra `Enter Open` footer hint.
That archived hint must not be treated as proof of correct modal action availability.
Screenshots do not prove click dispatch, queued-key preservation, or cross-run draft isolation.
Those behaviors require the separate interaction and regression checks.
The final integrated stack still needs current runtime evidence before release.

## Reproduce a pair

Use an isolated checkout for each source version.
Keep the same dependency versions, fixture, mode, size, and keys for both sides.
The commands below reproduce the states; wall-clock text is not byte-stable.

Set `SOURCE` to the checkout under test.
Set `EVIDENCE` to this directory.
Run these commands from the repository that contains this bundle.

```sh
SOURCE="$PWD"
EVIDENCE="$PWD/docs/tui-captures/execution-repair"
CAPTURE_HOME="$(mktemp -d)"
export SOURCE EVIDENCE CAPTURE_HOME
agent-tty --home "$CAPTURE_HOME" doctor --json
agent-tty --home "$CAPTURE_HOME" create --cols 120 --rows 40 --shell /bin/bash --json
```

Set `SESSION_ID` to the returned session ID.
Replace `run main omp` with the mode and fixture state from the table below.

```sh
agent-tty --home "$CAPTURE_HOME" run "$SESSION_ID" \
  "PYTHONPATH='$SOURCE/src' '$SOURCE/.venv/bin/python' '$EVIDENCE/fixture.py' run main omp" \
  --no-wait --json
agent-tty --home "$CAPTURE_HOME" wait "$SESSION_ID" --text 'Milknado isolated TUI audit' --json
agent-tty --home "$CAPTURE_HOME" wait "$SESSION_ID" --screen-stable-ms 250 --json
```

| Image state | Fixture state | Input after launch |
|---|---|---|
| main | main | None |
| help | main | `?` |
| search | main | `Ctrl+p`, then type `theme` |
| empty | empty | None |
| empty-graph | empty-graph | None |
| error | error | None |
| confirmation | main | `f`; do not confirm |
| session | main | `i` |
| owner-unavailable | owner-unavailable | None |

Use `run` or `watch` as the mode shown in the gallery.
Watch has no destructive-confirmation or writable-session case.
For compact captures, create the session at 80 columns and 24 rows.

Example input and capture:

```sh
agent-tty --home "$CAPTURE_HOME" batch "$SESSION_ID" \
  '[{"sendKeys":["?"]},{"wait":{"screenStableMs":250}}]' --json
agent-tty --home "$CAPTURE_HOME" snapshot "$SESSION_ID" --format text --json
agent-tty --home "$CAPTURE_HOME" screenshot "$SESSION_ID" --profile reference-dark --json
```

The stock screenshot command can time out after five seconds.
Use the historical 30-second RPC fallback before destroying the session:

```sh
export AGENT_TTY_ROOT="$(npm root -g)/agent-tty"
node --input-type=module - "$CAPTURE_HOME" "$SESSION_ID" <<'JS'
import {realpathSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
const base = process.env.AGENT_TTY_ROOT + '/dist/';
const {sendRpc} = await import(pathToFileURL(base + 'host/rpcClient.js'));
const {sessionDir, socketPath} = await import(pathToFileURL(base + 'storage/sessionPaths.js'));
const socket = socketPath(sessionDir(realpathSync(process.argv[2]), process.argv[3]));
const result = await sendRpc(socket, 'screenshot', {
  profile: 'reference-dark', rendererName: 'ghostty-web'
}, 30000);
console.log(JSON.stringify(result));
JS
agent-tty --home "$CAPTURE_HOME" destroy "$SESSION_ID" --json
```

Open the returned PNG before accepting the capture.
Keep failed attempts separate from accepted evidence.
Destroy each session and remove only its disposable fixture data.

## Matched gallery

These relative links resolve on the published branch.
Use branch or commit URLs in the PR description, not local filesystem paths.

| State and terminal size | Before | After |
|---|---|---|
| run · main · 120x40 | [Before](before/run-main-120x40.png) | [After](after/run-main-120x40.png) |
| run · main · 80x24 | [Before](before/run-main-80x24.png) | [After](after/run-main-80x24.png) |
| run · help · 120x40 | [Before](before/run-help-120x40.png) | [After](after/run-help-120x40.png) |
| run · help · 80x24 | [Before](before/run-help-80x24.png) | [After](after/run-help-80x24.png) |
| run · search · 120x40 | [Before](before/run-search-120x40.png) | [After](after/run-search-120x40.png) |
| run · search · 80x24 | [Before](before/run-search-80x24.png) | [After](after/run-search-80x24.png) |
| run · empty · 120x40 | [Before](before/run-empty-120x40.png) | [After](after/run-empty-120x40.png) |
| run · empty · 80x24 | [Before](before/run-empty-80x24.png) | [After](after/run-empty-80x24.png) |
| run · error · 120x40 | [Before](before/run-error-120x40.png) | [After](after/run-error-120x40.png) |
| run · error · 80x24 | [Before](before/run-error-80x24.png) | [After](after/run-error-80x24.png) |
| run · confirmation · 120x40 | [Before](before/run-confirmation-120x40.png) | [After](after/run-confirmation-120x40.png) |
| run · confirmation · 80x24 | [Before](before/run-confirmation-80x24.png) | [After](after/run-confirmation-80x24.png) |
| run · session · 120x40 | [Before](before/run-session-120x40.png) | [After](after/run-session-120x40.png) |
| run · session · 80x24 | [Before](before/run-session-80x24.png) | [After](after/run-session-80x24.png) |
| run · owner-unavailable · 120x40 | [Before](before/run-owner-unavailable-120x40.png) | [After](after/run-owner-unavailable-120x40.png) |
| run · owner-unavailable · 80x24 | [Before](before/run-owner-unavailable-80x24.png) | [After](after/run-owner-unavailable-80x24.png) |
| watch · main · 120x40 | [Before](before/watch-main-120x40.png) | [After](after/watch-main-120x40.png) |
| watch · main · 80x24 | [Before](before/watch-main-80x24.png) | [After](after/watch-main-80x24.png) |
| watch · help · 120x40 | [Before](before/watch-help-120x40.png) | [After](after/watch-help-120x40.png) |
| watch · help · 80x24 | [Before](before/watch-help-80x24.png) | [After](after/watch-help-80x24.png) |
| watch · search · 120x40 | [Before](before/watch-search-120x40.png) | [After](after/watch-search-120x40.png) |
| watch · search · 80x24 | [Before](before/watch-search-80x24.png) | [After](after/watch-search-80x24.png) |
| watch · empty · 120x40 | [Before](before/watch-empty-120x40.png) | [After](after/watch-empty-120x40.png) |
| watch · empty · 80x24 | [Before](before/watch-empty-80x24.png) | [After](after/watch-empty-80x24.png) |
| watch · error · 120x40 | [Before](before/watch-error-120x40.png) | [After](after/watch-error-120x40.png) |
| watch · error · 80x24 | [Before](before/watch-error-80x24.png) | [After](after/watch-error-80x24.png) |
| run · empty-graph · 120x40 | [Before](before/run-empty-graph-120x40.png) | [After](after/run-empty-graph-120x40.png) |
| run · empty-graph · 80x24 | [Before](before/run-empty-graph-80x24.png) | [After](after/run-empty-graph-80x24.png) |
| watch · empty-graph · 120x40 | [Before](before/watch-empty-graph-120x40.png) | [After](after/watch-empty-graph-120x40.png) |
| watch · empty-graph · 80x24 | [Before](before/watch-empty-graph-80x24.png) | [After](after/watch-empty-graph-80x24.png) |
