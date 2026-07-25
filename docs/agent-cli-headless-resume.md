# Agent CLI headless-resume spike

Date: 2026-07-25 · Issues: #246, #247 · Scope: locally installed CLIs only

## Conclusion

Claude Code and Codex both resumed a headless session with prior conversational context after
(1) an unchanged worktree reused at the same path and (2) deletion and recreation at that
same path. This proves conversation continuity, not that a prior tool call remains safe after
the filesystem changes. Resume-mode worktrees must therefore remain intact for every planned
resume round and may be merged and removed only after final approval.

Cursor's installed CLI could not start, so its headless capture/resume behavior was not tested.
The existing `resume + cursor-agent` validation refusal stays in force.

## Method

Each matrix row used an ephemeral local Git repository and a linked worktree:

```sh
root=$(mktemp -d /tmp/milknado-<cli>-resume.XXXXXX)
source="$root/source"
worktree="$root/reused-worktree"
git init -q -b main "$source"
git -C "$source" config user.email spike@example.invalid
git -C "$source" config user.name spike
git -C "$source" commit --allow-empty -qm seed
git -C "$source" worktree add -q -b spike "$worktree" main
```

The first prompt supplied a unique generated opaque token and requested an acknowledgement.
Each follow-up requested only that token. The token, session/thread IDs, credentials, and all
prompt text are redacted. A retained-context result means the follow-up output contained the
exact generated token; exit status was not used as the success criterion.

For the recreation row, the clean worktree was removed and re-added at precisely
`$root/reused-worktree`:

```sh
git -C "$source" worktree remove --force "$worktree"
git -C "$source" worktree add -q "$worktree" spike
```

## Results

| CLI | Version and authentication | Initial structured output | Same unchanged path | Deleted and recreated same path |
|---|---|---|---|---|
| Claude Code | `claude --version` → `2.1.218 (Claude Code)`, exit 0; `claude auth status` authenticated | `--output-format json` object contained nonempty `session_id` (redacted) | command exit 0; response matched the redacted token; worktree HEAD `cbd7d7d` | removal succeeded; recreated HEAD `cbd7d7d`; command exit 0; response matched the redacted token |
| Codex | `codex --version` → `codex-cli 0.145.0`, exit 0; `codex login status` authenticated | JSONL started with `{"type":"thread.started","thread_id":"<redacted>"}` | command exit 0; response matched the redacted token; worktree HEAD `c53246f` | removal succeeded; recreated HEAD `c53246f`; command exit 0; response matched the redacted token |
| cursor-agent | installed Cask artifact path identifies `2026.07.23-e383d2b` | unavailable | unavailable | unavailable |

The successful commands were:

```sh
(cd "$worktree" && claude -p --output-format json "$FIRST_PROMPT")
(cd "$worktree" && claude -p --output-format json --resume "$SESSION_ID" "$FOLLOW_UP_PROMPT")

(cd "$worktree" && codex exec --json --sandbox read-only "$FIRST_PROMPT")
(cd "$worktree" && codex exec resume --json "$THREAD_ID" "$FOLLOW_UP_PROMPT")
```

The Claude CLI documents `-p` and `--resume`; Codex documents `exec resume` and
JSONL output.[^1][^2] The Cursor documentation now describes headless mode and resume
parameters, but local vendor documentation cannot replace an authenticated execution test.[^3]

## Cursor prerequisite

`cursor-agent --version` exited 1 before printing a version. Its only relevant output began
with the installed bundle path
`/opt/homebrew/Caskroom/cursor-cli/2026.07.23-e383d2b/dist-package/index.js:414` followed by
raw JavaScript, rather than CLI output. This broken local installation prevents determining its
runtime version, authentication state, session-id output, or headless-resume behavior. No Cursor
result is inferred from this failure.

## Teardown rule

1. Capture the session/thread ID from the initial structured output.
2. Start every review and redispatch round from the original, still-existing worktree path.
3. Do not delete, recreate, or relocate that path while another resume is possible.
4. After final approval, perform the normal merge-back. Remove the worktree only when that merge
   landed; preserve it on failure or rejection.

The current execution flow already conforms: `Executor.complete` runs the review gate before
`WorktreeManager.rebase_and_merge`; rejected rounds redispatch the same `Path`; and
`rebase_and_merge` removes only after a successful landed merge.[^4]

[^1]: https://code.claude.com/docs/en/cli-reference
[^2]: https://developers.openai.com/codex/cli/reference/
[^3]: https://cursor.com/docs/cli/headless and https://cursor.com/docs/cli/reference/parameters
[^4]: src/milknado/domains/execution/executor.py:925-987, 1306-1341, 337-375
