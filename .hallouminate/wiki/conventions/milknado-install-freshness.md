# Milknado install freshness

An unchanged package manifest does not prove that a floating Git branch is current.
The dotfiles package cache must not skip floating UV updates.
A failed update must return failure rather than claim successful synchronization.

## Failure and fix

The manifest already declares Milknado from `git+https://github.com/paulnsorensen/milknado@main` with `float: true`.[^1]
The old cache-hit path exits before the UV update step.
The old floating-update path reports failure only as a warning.

The fix runs the floating-only UV pass on cache hits.[^2]
The cold path still updates each floating tool exactly once.
A missing floating tool installs from its manifest URL.
A missing `uv` command or failed floating upgrade makes synchronization fail.
Pinned and unmarked entries retain their existing behavior.

This guarantee concerns the installed Milknado package.
It does not promise that `dots sync` checks out the dotfiles repository's `main` branch.
Network or upstream failures can prevent an update; they must remain visible.

## Verification

The isolated dotfiles gate passes, including 1,440 Bats tests.[^3]
Two Opus reviews approve the fix after regression corrections.
The reviewed files are applied to the live dotfiles source without changes to unrelated files.
The fix is recorded as commit `0b711bcf` on `fix/milknado-sync-main`.
The live checkout retains its original branch and its separate local edits.

A separate UV fixture moves `main` between two commits without changing the package version.[^4]
Plain `uv tool upgrade` changes both the installed commit receipt and executable output.
No `--refresh` option is needed for this measured moving-ref case.

The live receipt and remote `main` both report `ca93d305b3008009b80134533fdb8553d76a009e` during this test.[^5]
The package version alone is not a freshness check.

[^1]: `/Users/paul/Dev/dotfiles/packages/packages.yaml`, Milknado entry.
[^2]: Dotfiles commit `0b711bcf`, `packages/sync.sh` and `tests/packages.bats`; isolated worktree `/private/tmp/dots-milknado-sync-main`.
[^3]: `just check` passes in the isolated dotfiles worktree. Focused UV tests pass 18 cases. The deployed live-source filter passes 14 cases. A full live `dots sync` is not run because it also changes unrelated machine configuration.
[^4]: `.context/uv-moving-main-test`: commit `64c2d1493cea55bf7d9d7417a01407a259bb960b` becomes `8a789548e8cf005e252f0ad970b70c8fe6fb746f`. A private UV tool directory and cache isolate the experiment from live tools.
[^5]: `uv tool upgrade milknado` prints `Nothing to upgrade`; `git ls-remote origin refs/heads/main` matches the installed `milknado-0.2.1.dist-info/direct_url.json` receipt. This is a September 2026 observation, not a permanent SHA pin.


The publication branch applies only the floating-tool patch to newer dotfiles `main` at `8b8f3ffa`.
It excludes the unrelated local Codex pin from the original repair branch.
A sync-fixture failure reproduces on both that branch and a clean control at the same `main` commit.
The fixture omits the new external tilth install command from its shared mocks.
It calls the live mise shim under the fake test home, where the shim has no installed tool.
The fixture repair mocks that external command without changes to production behavior or assertions.

The first publication gate also reports one broker readiness setup failure.
Its focused rerun passes; the exact cause of that isolated startup failure is not established.

