# Review harness: verdict loss and node_reviews PK collision

Two related bugs in the node-review loop caused every multi-round review during the
basedpyright-zero campaign (2026-08/09, nodes 84, 87, 89, 93, 95, 96, 97, 100, 101, 102)
to end in `review blocked after round 3` even when the work was sound.

## Bug 1 — codex-path reviewer verdicts often unparseable

The codex-family reviewer frequently returns no parseable `<verdict>` tag. The harness
then stores the literal string `reviewer produced no parseable <verdict> tag` (44 bytes)
as the round's findings and records `reject` — or, in earlier rounds, stored the raw
~64KB transcript as findings. Three unparseable rounds → node blocked. The worker's
actual diff quality is never assessed.

## Bug 2 — node_reviews PK swallows cycle-≥2 verdicts

`node_reviews` has `PRIMARY KEY (node_id, round)` and `insert_node_review`
(`_persistence.py`) does a bare INSERT; the IntegrityError is swallowed in
`executor.py` `_notify_review` (~line 947). A redispatched node re-running round N
silently drops the new verdict.

Also: `MILKNADO_RUN_ID` is unset in reviewer environments.

## Operational recovery playbook (proven 8x in the campaign)

1. Read `node_reviews.findings`; if it is the 44-byte stub, the block is the parse bug,
   not the work.
2. Mechanically verify the worktree yourself — run each check as a SEPARATE command
   (chained checks have silently skipped): `grep -rn "^# pyright:" tests/` (must be
   empty), scoped gate commands, tests. `just check-llm` remains the gate; this
   separate-command warning is about ad-hoc `&&` chains during forensic verification,
   not a substitute for the gate.
3. Dispatch an independent reviewer (severity-report) over the worker diff; cure real
   findings; squash to one commit; rebase onto campaign tip; ff-merge; push.
4. Mark done via direct SQL `UPDATE nodes SET status='done' WHERE id=N` —
   `set_subtree_status` cannot do failed→done and walks dependency edges.
5. Relaunch `milknado run --strict` (it drains on any strict failure).

## Worker cheat patterns reviews must check (all observed)

- File-wide `# pyright:` disable directives hiding hundreds of diagnostics (nodes 87,
  96, 100, 101; one even landed in src via node 88's `_completion.py`).
- `cast(X, cast(object, fake))` laundering partial fakes past Protocol checks.
- Deleted/renamed tests, demoted assertions (`assert x` → `_ = x`), assertions
  re-indented into `pytest.raises`, dropped thread barriers (turns ordering tests
  into races), `Callable[..., X]` accessors erasing the arity under test.
- Cure agents can also FALSELY REPORT fixes done (node 102's cure claimed two restored
  tests that were never written) — grep for the exact restored symbols before merging.
