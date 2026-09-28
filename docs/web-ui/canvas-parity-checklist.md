# Canvas -> live dashboard parity checklist (2026-09-27)

Source: https://claude.ai/artifact/Gb9Yyqia1FrtZ3t5rxKaLg ("Milknado Web", Milknado design
system). 13 wide boards (1440x900) share one markup, differing by `mkInit()` state. 2 narrow
boards (390x844).

## Artboards

- **Main** — run, sel 8, tab Session, dialog none: base run view.
- **Changes** — run, sel 8, tab Changes: 560px sidecar, file list + diff.
- **Details** — run, sel 8, tab Details: 560px sidecar, Brief + kv groups, paginated.
- **Watch** — watch, sel 8: "Read-only" badge only, no Stop scheduling, ETA/Attempt/guidance = `unavailable`, no session input.
- **Permission** — run, sel 9: at-risk "Permission requested" block, Approve/Deny.
- **ForceStop** — dialog "force": "Force stop the run?" confirm.
- **StopScheduling** — dialog "stop": "Stop scheduling and stop 2 active runs?" confirm.
- **Errors** — sel 14 (failed), dock true, one at-risk toast: "Session input was rejected: the run is not active."; sidecar shows `errText`.
- **Help** — dialog "help": Keyboard shortcuts, 3 columns (Graph/Runs/Steering).
- **AddNode** — sel 16, dialog "add": Description, Parent, Flavor segment, Prerequisites, Files.
- **GoalReview** — sel 1, review+reviewOpen true: 560px review aside, Accept/Reject change.
- **ReadyFilter** — sel 17, filter "ready", hideDone true: toolbar filtered.
- **Light** — same as Main, `data-theme="light"`.
- **NarrowList** (390x844) — header+badge, kicker/title/StatusStrip/totals, 44px search, OutlineTree, footer "Open node".
- **NarrowDetail** (390x844) — back link, AncestorPath(max 4), title/badge, kv rows, console tabs, seg+Send, footer Cancel run/Force stop.

## Rail

Spec: `nav` width `var(--rail-width)`=216px. `rail-btn` (full-width, glyph+count): "Dispatch ready", "Add node", "Harvest done". `Milknado.AgentRoster`. Reviews: kicker+count, `rail-row` "▣ Goal review 3" / "node 1", else "No pending reviews". Project: `rail-row` Graph(23, selected)/Outline/Roadmap(0.3.0). Footer: `StatusStrip short` + `{{totals}}` caption + "goal run 7c1e · 2h 04m". `.rail-row` = grid `1fr auto`, height 32px, 13px/500.

- [ ] AGENTS/REVIEWS panels overprint — live renders unclamped free-text description instead of `AgentRow`'s fixed `sub`/`figure` line.
- [ ] Review rows show only evidence text + "Open" link, not the `rail-row` "▣ Goal review N" / "node X" layout.
- [ ] Footer "0 / 51" ignores 4 failed — spec totals string is `"2 active · 9 completed · 1 failed · 0 stopped · {ready} available"` (all 5 figures).

## Header

Spec: kicker `{{modeKicker}}` = `"(Run|Watch) · feat/run-steering"`; `h1.t-display` = running goal's title (canvas: node 1's title, single-line ellipsis, 32px/38px serif). Run mode: `StatusBadge running` "Run active" + "Stop scheduling". Watch mode (mutually exclusive): `StatusBadge pending` "Read-only", no Stop scheduling. "Keys" button always present.

- [ ] Header title is the first root, not the running goal — spec drives `h1` from the running/selected node.
- [ ] "Read-only" badge shown together with "Stop scheduling" in owner run mode — spec branches are mutually exclusive.

## Graph

Spec: `GraphToolbar` (filter/hideDone/focus/lod cards-pills-dots/jump/collapse-expand/zoom/fit) over `MikadoGraph` (levelGap 96). Card `.mk-node` width `var(--node-width)`=208px; `.mk-node-title` = `font-weight:500; -webkit-line-clamp:2; box-orient:vertical; text-wrap:balance` (2-line clamp, not 1-line ellipsis); compact variant 148-160px one-line. `GraphNodeProps` has no id field — canvas never shows a node id on the card. `Milknado.Minimap` exists in the design system (`nodes`/`selected`/`viewport`, `.mk-minimap-block` filled, `.mk-minimap-viewport` outlined) but **no wide artboard mounts it** (grep of all 13 files: zero `Minimap` uses).

- [ ] Card titles truncate ~18 chars single-line — spec is a 2-line balanced clamp, not 1-line ellipsis.
- [ ] Card titles show no node id — not a canvas gap; `GraphNodeProps`/`.mk-node` never render an id on the card in this design.
- [ ] Minimap is an empty grey grid — canvas mounts no `Minimap` component on this screen at all; either remove it or build it against the real component (not currently either).

## Sidecar

Spec: `aside` `side-360`(360px)/`side-560`(560px, Changes tab or review). Header: `AncestorPath`(max 4) + close button. `h2.t-title` (22px/28px serif) = full `{{n.title}}`, no clamp specified. Badge row: `StatusBadge`, agent/flavor chips, `"node {{id}}"`. Run `kv`: Node/Run/Status/Elapsed/ETA/Attempt/guidance (watch→`unavailable`). `canAct`→ Cancel run/Force stop. Permission block: at-risk bg, `perm_41` id, command line, Approve/Deny. Session tab: `Console` (`con-tall` 340px / `con-short` 236px when perm/error shown), error alert row (`role=alert`, `errText`), `mk-seg` Steer/Follow up/Interrupt + input + "Send to agent"; watch → read-only caption; inactive run → "no longer accepts session input." Changes tab: `mk-console-tab` tabs in `.con-well`(520px), file grid `St/Path/+/−`, diff. Details tab: Brief + Identity/Execution/Paths/Time `kv` groups, `"Detail page {{page}} of 2"`.

- [ ] Review rows show only evidence text + "Open" link — should match rail-row / kicker+data layout (see Rail).
- [ ] Sidecar description in unclamped 22px serif pushes tabs below fold — spec's `h2.t-title` has no line-clamp (fixture titles are short); adding a clamp is new scope, not a spec violation.
- [ ] "Ended: running" — spec has no such literal row; `completed_at` stays `"none"` while active, with no separate "Ended" field.
- [ ] Failed node shows no error text — spec's `hasErr`/`errText` alert (e.g. `"Error: the gate failed. 2 tests failed in tests/graph/test_reviews.py."`) must render above the input area when `run.error` is set.
- [ ] `/api/runs/<id>/changes` 404 uncaught — **not specified**; canvas is a static mock with no real network calls (only empty state is "No changes").
- [ ] Breadcrumb parent truncated — `AncestorPath` takes `max` (default 4); no per-item truncation CSS found (`.mk-ancestor*`/`.mk-crumb*` absent from `bundle.css`) — **not specified** beyond the 4-item cap.

## Dialogs & toasts

Spec: `.scrim` (`inset:0`, 72% surface-sunken backdrop), `role=dialog aria-modal=true`, width 440px (760px Help), serif title, muted body, right-aligned buttons.
- Force stop: "Force stop the run?" / "The run stops now. It does not wait for the current turn. Changes that are not committed stay in the worktree." / "Keep the run", "Force stop".
- Stop scheduling: "Stop scheduling and stop 2 active runs?" / "Milknado dispatches no more nodes. Each active run stops after its current turn. Done work stays in the graph." / "Keep running", "Stop runs".
- Help: "Keyboard shortcuts", 3 columns (Graph/Runs/Steering).
- Add node: "Add a node" — Description, Parent, Flavor segment, Prerequisites, Files / "Cancel", "+ Add node".
- Toasts: bottom-right 320px, glyph + text + "Dismiss"; Errors board seeds "Session input was rejected: the run is not active." (at-risk).

- [x] Verified against the live dashboard with the before/after captures listed in `.cheese/web-canvas-parity-findings.md` and the regression checks recorded there.

## Narrow

Spec: 390x844, 44px min touch targets. NarrowList: header+badge → kicker/title/StatusStrip/totals/search → `OutlineTree` → footer node title+id + "Open node". NarrowDetail: back-link header → AncestorPath/title/badge/kv/console/seg+Send → footer Cancel run/Force stop.

- [x] 390px view renders the header, StatusStrip, totals, search, OutlineTree, and footer bar at 390px.
- [x] 390px view renders NarrowDetail with its back link, console tabs, session controls, and owner controls.
- Evidence: [`narrow-before-list-390x844.png`](narrow-before-list-390x844.png) → [`narrow-after-list-390x844.png`](narrow-after-list-390x844.png); [`narrow-before-detail-390x844.png`](narrow-before-detail-390x844.png) → [`narrow-after-detail-390x844.png`](narrow-after-detail-390x844.png); [`wide-before-sidecar-1440x900.png`](wide-before-sidecar-1440x900.png) → [`wide-after-sidecar-1440x900.png`](wide-after-sidecar-1440x900.png).
- Captures use the owner fixture, the same graph and run state, and the same interaction sequence: open the fixture node, open its detail view, and switch to 1440px for the wide pair.
- The narrow detail pair was regenerated on 2026-09-28 with one Playwright script run twice at 390x844, bottom-scrolled: `before` against the committed build at the branch base `88f62cdf`, `after` against this branch's committed build. The before image shows the base's three separate session buttons with no Send and no sticky run footer; the after image shows the segment, Send, and the sticky Cancel run / Force stop footer.
- Deterministic interaction tape: `just test-file tests/browser/test_narrow_layout.py`; `NARROW_VIEWPORT` is 390x844, and the owner test opens the node, checks the bottom run footer, selects Interrupt, and sends through Send.
- Verification: `just check-llm` passes on the captured source and regenerated static build.

## Out of scope

Per `.cheese/web-canvas-parity-findings.md` (decision `d-ec38938f369d`): no backend exists — leave out:
- "Dispatch ready" rail button.
- "Harvest done" rail button.
- Coordinator message input ("Message the coordinator"; Events dock/toggle itself is real).
- Project rail rows "Outline" and "Roadmap" (only "Graph" is wired).
