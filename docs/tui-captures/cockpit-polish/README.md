# Cockpit and login polish captures

Before captures come from base commit 4af3a962. After captures come from the polish branch.
Each pair uses the same viewport, theme, and data: a throwaway project with one goal and three tasks.
The `cockpit-*` captures show the intake state before any coordinator session exists.
The `proposal-*` captures show one Claude goal with a pending plan proposal.
`fixture_planner.py` makes that proposal through the `local-planner` hook, so no model runs.
No provider session or worker runs in any capture.

| Capture | Viewport | Theme | State |
| --- | --- | --- | --- |
| `login-1440x900-dark.png` | 1440x900 | dark | Login before the token |
| `cockpit-1440x900-dark.png` | 1440x900 | dark | Goal intake |
| `cockpit-1280x720-light.png` | 1280x720 | light | Goal intake |
| `cockpit-narrow-390x844-dark.png` | 390x844 | dark | Goal intake |
| `proposal-1440x900-dark.png` | 1440x900 | dark | Pending plan approval |
| `proposal-narrow-390x844-dark.png` | 390x844 | dark | Pending plan approval |

On the base build, the pending proposal's Approve and Reject buttons fall outside the
panel's own scroll box at 1440x900. On this branch, both buttons are visible without scrolling.

## Reproduce

```bash
mkdir proj && cd proj && git init && git commit --allow-empty -m init
milknado init
milknado add-node "Ship the billing dashboard" -k goal
milknado add-node "Design schema" -p 1
milknado add-node "Build API endpoints" -p 1 --prereq 2
milknado add-node "Render charts" -p 1 --prereq 3
cp ../fixture_planner.py .
printf '[milknado]\nplanning_agent = "local-planner"\n' > milknado.toml
MILKNADO_LOCAL_PLANNER="$PWD/fixture_planner.py" \
  milknado launch --no-open --port 8766 --project-root . > launch.log 2>&1 &
python -I ../capture.py launch.log out/ --proposal
```

Run the `before` captures on a build of commit 4af3a962.
