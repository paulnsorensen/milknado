# Meta verification

A meta test must distinguish scheduler state, worker identity, transport vocabulary, review results, and human approval.
A passing worker does not prove review approval or merge-back.

## September 2026 run

The Bordeaux workspace runs four native Milknado tasks with Luna workers and Opus reviewers.
Nodes 2–5 finish and merge their smoke evidence, MCP contracts, flavor profiles, and handback tests.
Node 2 first demonstrates rejected-review handback, preserved work, and operator-directed recovery.
Node 3 exposes progress-only reviewer output that the old parser treats as a code rejection.

The original final combined gate fails with 3,388 passed, three skipped, and two failed tests.[^1]
All 27 focused meta tests pass.
A separate clean-main control passes once.
That control does not establish the cause of the intermittent failures.
The follow-up reproduction does not require the new meta tests.



Follow-up nodes 6–8 track the coordinator's repair and record tasks.
Their Luna workers use isolated collaboration worktrees, not native Milknado run loops.
Their graph status is not additional native-run evidence.

## Worker identity and test isolation

Workers receive `MILKNADO_NODE_ID`, `MILKNADO_RUN_ID`, and `MILKNADO_PROJECT_ROOT`.[^2]
These variables intentionally bind worker operations to the coordinator project.
A test that supplies another project root can fail under this identity.

The completion verifier removes the complete `MILKNADO_*` namespace before quality gates.
A worker shell does not automatically apply that boundary.
The autouse test fixture removes this namespace from each test process.[^3]
Its regression injects identity into a real pytest subprocess.
Do not remove identity from the worker process that deposits results.

## Blocking is not human approval

Scheduler readiness excludes blocked nodes.
Direct claims accept blocked nodes and transition them to running.[^4]
Thus, `blocked` is a scheduler hold, not an enforced human-approval lock.
The tests cover both paths.
This repair preserves that authorization model.

Follow-up work normally becomes a sibling of the current worker.[^5]
This placement lets the worker finish without waiting for the follow-up.
The enclosing goal can still wait for that follow-up.

## Profile and transport checks

The meta tests cover all seven built-in flavors and the configured `triage` flavor.
They check every resolved profile field, inheritance, overrides, registry entries, and reviewer requirements.
Review defaults apply to `implement` and `spec`.
Both need a reviewer command when review remains enabled.
The `review` flavor does not require another review layer.

MCP todo input uses `in_progress`, which maps to domain `running`.
Node output uses domain status strings.
The MCP tests use the registered transport and check intentional projections, invalid mutations, hierarchy, flavors, and canonical run output.
Input and output enums are not required to be identical.

## Repairs and evidence

[Worker startup](../gotchas/macos-worker-stub-startup.md) records the controlled fixture comparisons and detached-runner import correction.
The fixes preserve real processes, existing deadlines, output checks, cancellation, and environment checks.
They do not increase production timeouts or skip flaky tests.

[Review verdicts](../gotchas/review-harness-verdict-loss.md) records durable approvals, atomic audit sequences, and reviewer-error handbacks.
Invalid reviewer output returns control without a blind worker retry.
Required audit failures block merge under both rejection policies.
A findings-file error cannot bypass that database audit and handback path.

[Install freshness](milknado-install-freshness.md) records the separate `dots sync` repair and moving-main experiment.
Successful package synchronization refreshes floating tools even when the manifest cache matches.
Update failures remain visible.

Actual Opus reviews approve the behavioral fixes.
The combined repository gate remains the final integration requirement.
Focused tests and reviewer approval do not replace `just check-llm`.



The size correction moves prompt and findings helpers into the execution slice's private `_review.py` module.
It does not widen existing source-size limits.
Opus catches a weakened test that injects its own expected diff into the prompt helper.
The required regression must inspect the prompt sent through Executor and the Git port, not only helper interpolation.

## Cross-repository fixtures

Keep foreign repositories outside a parent repository that supplies tool configuration.
A nested dotfiles worktree inherits Milknado's Ruff configuration during this test.[^6]
The same unchanged Python file passes Ruff in the dotfiles checkout.
This measured configuration contamination is separate from the worker startup failures.

## Evidence references

[^1]: `.context/meta-final-check.log`; control SHA `ca93d305b3008009b80134533fdb8553d76a009e` at `/private/tmp/milknado-meta-baseline-control`.
[^2]: `src/milknado/domains/dispatch/runner.py`; `src/milknado/app/project.py::resolve_project_root`.
[^3]: `tests/conftest.py::_isolate_worker_identity`; `tests/test_meta_mcp_contracts.py::test_worker_identity_is_cleared_before_each_test`.
[^4]: `src/milknado/domains/graph/_reads.py`; `src/milknado/domains/graph/_transitions.py`; `src/milknado/mcp/node.py`.
[^5]: `src/milknado/mcp/todo.py`; `src/milknado/mcp/todo_mutate.py`.
[^6]: `ruff check --show-settings skills/session-analytics/scripts/ingest.py` selects Milknado's configuration in the nested worktree. Both source copies have Git blob `bc02a8f2f25344ad408549c238ae746bb4869401`. The scoped Ruff command passes in `/Users/paul/Dev/dotfiles`.


## Final verification

The combined source at `b3a1a8a` passes `just check-llm`.
Gate 5 takes 68.7 seconds.
Project coverage is 97.66%; project and diff coverage both meet the 95% gate.
Actual Opus gives final approval with no new findings after the prompt-seam regression is restored.

| Repair gate | Result | Correction |
| --- | --- | --- |
| 1 | Final typecheck rejects private helper imports | Export shared test helpers with public names |
| 2 | Runner logging test patches the old import owner | Patch the application module |
| 3 | Review changes exceed source-size limits | Extract cohesive helpers without new waivers |
| 4 | Final typecheck rejects unused audit sequence returns | Explicitly discard seed results |
| 5 | PASS in 68.7 seconds | All integration checks pass |
| 6 | PASS in 73.0 seconds | Unchanged source passes the stability repeat |

Typecheck runs after tests and coverage in this gate.
A typecheck failure does not mean those earlier steps are skipped.
Raw gate logs remain in `.context/meta-repair-check-*.log`.
The final Opus report remains in `.context/meta-final-cure-opus.txt`.
The dotfiles fix is separately committed as `0b711bcf` and applied to the live source.
Neither repository's `main` branch is changed by these local commits.



Both final gates pass on unchanged source.
The second log is `.context/meta-repair-check-6.log`.
The working branch records the fixes; upstream `main` remains unchanged.

Watch the graph with:

```bash
milknado watch --project-root /Users/paul/conductor/workspaces/milknado/bordeaux
```

