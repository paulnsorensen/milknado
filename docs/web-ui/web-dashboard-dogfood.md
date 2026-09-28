# Web dashboard dogfood read paths

Date: 2026-09-27.

The live smoke used `milknado web --project-root /tmp/milknado-web-round2 --port 8787 --no-open` with Chromium at `1440x900`. The seeded graph contained roadmap node 43, active goal node 87, live task node 88, four failed task runs, and one pending goal review. The goal and one failed task description were edited through `milknado_edit_node` before the refresh captures.

## Reviewer-accessible captures

Wide captures use the same 1440x900 viewport and theme.
- [Before live dashboard](dogfood-before-live.png)
- [Before selected sidecar](dogfood-before-selected.png)
- [After live dashboard and MCP refresh](dogfood-round2-live.png)
- [After root switch](dogfood-round2-root-switcher.png)
- [After active-run selection](dogfood-round2-active-run.png)
- [Owner header before fix](owner-run-mode-before.png)
- [Owner header after fix](owner-run-mode-after.png)
- [After failed run: error and no changes](dogfood-round2-failed-no-changes.png)
- [After long description, collapsed](dogfood-round2-long-description.png)
- [After long description, expanded](dogfood-round2-long-description-expanded.png)

Narrow owner-mode captures use the same 390x844 viewport and fixture:
- [Owner header before fix, narrow](owner-run-mode-narrow-before.png)
- [Owner header after fix, narrow](owner-run-mode-narrow-after.png)

The wide long-description captures use the same viewport and theme as the live captures. They isolate the detail response from the graph-card title, so the three-line sidecar clamp and control visibility are directly reviewable.

## Sidecar header and empty-state evidence (2026-09-28)

The fixture pairs use the same graph, light theme, and 1440x900 viewport. Baseline captures use the `HEAD` bundle; after captures use the rebuilt bundle. The fixture has no graph-edit backend, so its visible "Graph edits are unavailable." notice is unrelated.

- Empty sidecar: [before](sidecar-before-empty.png) → [after](sidecar-after-empty.png)
- Deep-node path: [before](sidecar-before-long-path.png) → [after](sidecar-after-long-path.png)
- Pending permission on a non-owner selection: [before](permission-before-non-owner.png) → [after](permission-after-non-owner.png)

The empty-state interaction loads the dashboard before selecting a node. The path interaction selects `Child task` beneath three ancestors. The permission interaction selects node 2 while the pending request belongs to owner node 1. The after states show the caption, no sidecar buttons or textarea before selection, a four-item path, single-line path ellipsis, and no Approve/Deny controls for a non-owner selection. Full ancestor names remain available in `title`; the accessible name uses the concise summary.

## Defect table

| Defect | Found interaction and evidence | Result and regression check | Deferred |
| --- | --- | --- | --- |
| 1. AGENTS description overflow | Open the live dashboard and inspect the AGENTS rail below the header. Before: `dogfood-before-live.png`. | Agent rows use the fixed two-line sub/figure layout. `tests/browser/test_live_update.py::test_published_snapshot_updates_the_page_without_navigating` measures the row and its overflow styles. | None |
| 2. REVIEWS row format | Open a goal with a pending review and inspect the REVIEWS rail. `dogfood-round2-live.png` shows the row and its decision state. | Rows render `▣ Goal review N`, `node X`, and the decision state. `tests/browser/test_goal_review.py`. | None |
| 3. Footer totals | Open the live dashboard and read the rail footer. `dogfood-round2-live.png` shows `1 active · 0 completed · 4 failed · 0 stopped · 5 available`. | Watch and run snapshots read durable totals, while omitted wire totals remain unknown. `tests/test_graph_observer_snapshots.py::test_watch_totals_include_runs_outside_the_bounded_run_page`, `tests/test_execution_controller.py::test_controller_snapshot_uses_durable_run_totals`, `web/src/features/live-state/RunTotals.test.tsx`. | None |
| 4. Running-goal title | Start with roots 43 and 87, then inspect the header. `dogfood-round2-root-switcher.png` shows the root switch interaction; `dogfood-round2-live.png` shows the active goal. | Header title follows the running goal, with a switcher when multiple roots exist. `web/src/app/hosts/GoalTitle.test.tsx`. | None |
| 5. Run/watch controls | Start `milknado run --web` with two active runs, then inspect the header, sidecar, and Session tab. `owner-run-mode-before.png` shows the incorrect Read-only header; the narrow pair uses the same fixture at 390x844. | `host_owner` drives run/watch presentation while `owner.run_id` remains the command-routing fact. Run mode and watch mode use mutually exclusive control branches. `tests/browser/test_observer_mode.py`, `tests/web/test_capabilities.py`. | None |
| 6. Minimap placeholder | Open the canvas and inspect its top-right corner. `dogfood-round2-live.png` has no empty grey minimap grid. | The placeholder is not mounted. `tests/browser/test_graph_view.py::test_wide_canvas_does_not_mount_placeholder_minimap`. | None |
| 7. Graph title clamp | Open the live graph and inspect the cards. `dogfood-round2-live.png` shows balanced two-line titles without node IDs. | Compact cards keep vendor spacing and use the two-line title clamp. `tests/browser/test_graph_view.py::test_compact_graph_cards_do_not_intersect`. | None |
| 8. Long sidecar description | Select node 89 after the MCP edit. `dogfood-round2-long-description.png` shows the collapsed three-line heading, expand control, status, run, and tabs before the fold. | The sidecar measures real unclamped height and expands only when needed. `tests/browser/test_node_detail.py::test_long_description_shows_expand_control_only_when_clamped`, `web/src/features/node-sidecar/NodeSidecar.test.tsx`. | None |
| 9. Active Ended row | Select `Live worker task` and inspect its run details. `dogfood-round2-active-run.png` shows `Completed: none`. | Active runs show `Completed: none`; the UI has no Ended row. `tests/browser/test_node_detail.py`, `tests/browser/test_session_input.py`. | None |
| 10. Failed changes route | Select `Failed worker 0`, open Changes, and inspect the sidecar. `dogfood-round2-failed-no-changes.png` shows `worker session gone` above the session input and `No changes`. | A no-worktree run returns the same empty state as a 404, without a page error. `tests/browser/test_node_detail.py::test_failed_run_without_worktree_uses_real_no_changes_state`. | None |
| 11. Ancestor path | Select a deep node and inspect the breadcrumb. `dogfood-round2-failed-no-changes.png` shows the full short parent name; the long-path fixture covers four items. | AncestorPath caps at four items, summarizes long titles, and applies single-line ellipsis. `tests/browser/test_node_detail.py::test_ancestor_path_caps_at_four_items_with_single_line_ellipsis`; `web/src/features/node-sidecar/NodeSidecar.test.tsx`. | None |
| 12. Permission action ownership | With pending request `perm-1` owned by node 1, select node 2. Before: `permission-before-non-owner.png` shows Approve/Deny under node 2. | PermissionActions requires `canActOnSelectedRun`, so node 2 has no permission block. `web/src/features/session-input/PermissionActions.test.tsx`. | Graph task #116 tracks the pending permission indicator. |

| 13. Run selection tab panels | Press `N` to select an active run and inspect the sidecar tabs. | `selectedNodeId` gates tab bodies, Changes, and graph actions for run selections. `web/src/features/node-sidecar/NodeSidecar.test.tsx`, `web/src/features/changes/ChangesSection.test.tsx`, `web/src/features/graph-edits/NodeActions.test.tsx`. | None |

## Capture protocol

1. Seed the graph with `uv run python tests/browser/seed_dogfood.py --root /tmp/milknado-web-round2`.
2. Start `milknado web --project-root /tmp/milknado-web-round2 --port 8787 --no-open`.
3. Open the printed token URL in Chromium at `1440x900`.
4. Edit goal node 87 through `milknado_edit_node` and confirm the title and card update without navigation.
5. Edit failed node 89 through `milknado_edit_node` with the long description used by `tests/browser/capture_dogfood.py`.
6. Run `uv run python tests/browser/capture_dogfood.py <printed-auth-url>`.
7. The capture selects root 43, selects the active run, selects failed worker 1, opens Changes, and expands the long description.

## Additional state checks

- Empty database: `tests/browser/test_error_states.py::test_empty_and_unreadable_projects_render_empty_dashboard` covers the no-DB state.
- Unreadable database: the same test verifies quarantine and recovery to an empty dashboard.
- Stale build: `tests/browser/test_stale_build.py` verifies that committed static files match `web/`.
- Live MCP refresh: the capture command reports the edited goal title and the live graph renders it without navigation.

Deferred: pending permission indicator (graph task #116).
