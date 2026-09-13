# Agent steering TUI evidence

This bundle compares the shared `milknado run` and `milknado watch` workspace before and after task 38.

## Evidence sets

- [`captures/`](captures/) contains the baseline from commit `a6b2dee`.
- [`captures-after/`](captures-after/) contains the final task 41 captures.
- Each manifest records the terminal size, surface, state, theme, and fixture limits.

The paired states cover main, session, permission, error, and run confirmation views.

The final-only states cover unavailable-owner and top-level goal-review views.

Both sets include standard `120x40`, compact `80x24`, and minimum `40x15` terminals.

## Reproduce the final set

Run this command from the repository root:

```bash
uv run python docs/tui-captures/agent-steering/capture.py \
  --output docs/tui-captures/agent-steering/captures-after
```

The script uses one synthetic snapshot and a fixed `12:00:00` clock.

The script checks each SVG view box and embedded metadata record.

## Inspection result

All 39 final SVG captures were generated from the same fixture and theme.

The minimum fallback and confirmation borders remain inside the `40x15` viewport.

The standard and compact footers show no clipped or overlapping key hints.

Compact graph labels truncate at the pane edge by design.

The selected node's detail view exposes the complete description and related records.

Permission decisions remain explicit.

The operator must select both a decision and the exact permission request.

## Fixture limits

The fixture uses fake-vendor protocol sessions for Codex, Claude, and OMP.

The fixture does not start a live worker, database, agent process, or attached-watch owner.

The captures do not prove live provider compatibility.
