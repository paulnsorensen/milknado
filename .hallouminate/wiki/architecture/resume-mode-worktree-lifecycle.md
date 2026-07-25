# Resume-mode worktree lifecycle

Claude Code 2.1.218 and Codex 0.145.0 both retained conversational context in local headless tests after a same-path resume and after deleting then recreating a worktree at that exact path.[^1] That only proves transcript continuity: it does not establish that earlier tool-call history is safe against replacement filesystems. Keep a resume-mode worktree intact through every review/redispatch round, then merge and remove it only after final approval.

The execution slice already follows this ordering. `Executor.complete` runs the review gate before `WorktreeManager.rebase_and_merge`; rejection redispatches the identical `Path`; landed merge is the only condition that removes it.[^2] The adapter adds the vendor-specific resume selector at the external CLI boundary, leaving execution lifecycle vendor-neutral.[^3]

Cursor remains excluded from `session_mode = resume`: the installed Cask artifact `2026.07.23-e383d2b` failed before version/authentication/session capture, so no behavioral result exists.[^1]

[^1]: docs/agent-cli-headless-resume.md
[^2]: src/milknado/domains/execution/executor.py:925-987, 1306-1341, 337-375
[^3]: src/milknado/adapters/loop.py:42-80
