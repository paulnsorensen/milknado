# Meta-harness coordinator TUI captures

These pairs show the new coordinator status panel after pressing `c`.
The base revision has no `c` panel, so the same key leaves its main view open.
Both runs use the agent-steering fixture, the same theme, the same data, and fixed 12:00:00 clock.
The fixture is synthetic. It does not start a live worker or resume a provider session.

| Surface | Size | Before | After |
| --- | --- | --- | --- |
| Read-only watch | 80×24 | [before](./before/watch-coordinator-80x24.png) | [after](./after/watch-coordinator-80x24.png) |
| Read-only watch | 40×15 | [before](./before/watch-coordinator-40x15.png) | [after](./after/watch-coordinator-40x15.png) |
| Attached watch | 80×24 | [before](./before/attached-watch-coordinator-80x24.png) | [after](./after/attached-watch-coordinator-80x24.png) |
| Attached watch | 40×15 | [before](./before/attached-watch-coordinator-40x15.png) | [after](./after/attached-watch-coordinator-40x15.png) |

The read-only panel directs the user to attach to the owner. The attached panel offers owner-validated session input.
The recovery line shows an unavailable execution group. The 40×15 panel remains readable without clipping.
All eight PNG files were opened and checked for focus, key hints, status, and clipping.

## Reproduce

Use one checkout at base revision `e65267671fd4b84e1dd3e30d9be9283fd1e97e99` and one checkout at the PR revision.
Run this script from the PR checkout. Replace `BASE` and `AFTER` with checkout paths.
The script loads Milknado from `PYTHONPATH` and the matching fixture from `--source-root`.

```sh
PYTHONPATH="$BASE/src" timeout 45 .venv/bin/python docs/tui-captures/meta-harness-07/capture.py \
  --source-root "$BASE" --source-revision e65267671fd4b84e1dd3e30d9be9283fd1e97e99 \
  --output /tmp/meta-harness-before
PYTHONPATH="$AFTER/src" timeout 45 .venv/bin/python docs/tui-captures/meta-harness-07/capture.py \
  --source-root "$AFTER" --source-revision PR_HEAD \
  --output /tmp/meta-harness-after
for set in before after; do
  for svg in /tmp/meta-harness-"$set"/*-coordinator-*.svg; do
    rsvg-convert -o "${svg%.svg}.png" "$svg"
  done
done
```

The Textual test harness can hang during process teardown after it writes the images and manifest.
A timeout does not prove capture failure. Check the four SVG files and `manifest.json` in each output directory.
