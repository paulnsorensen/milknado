# Graph read port and execution snapshots

`GraphReadPort` is the supported cross-slice read boundary: traversal and dispatch consumers use node, child, and file-ownership reads, while execution receives one atomic `GraphExecutionSnapshot`. The port keeps consumers independent of `MikadoGraph` and graph storage internals.[^1]

Execution, not the graph crust, converts snapshot facts into dispatchable-node counts and the execution overview. The graph snapshot holds the lock and a SQLite savepoint across root, requested-node, ready/running, and conflict facts, preserving a cross-connection-consistent read without exposing a connection or private read module.[^2]

The import-linter contract intentionally forbids graph → execution imports. This protects the ownership direction: graph stores transactional state; execution owns scheduling policy and presentation assembly.[^3]

[^1]: src/milknado/domains/common/protocols.py:36-69; src/milknado/domains/graph/traversals.py:3-34; src/milknado/domains/dispatch/brief.py:8-172
[^2]: src/milknado/domains/graph/graph.py:350-373; src/milknado/domains/execution/executor.py:217-249; tests/test_graph_atomicity.py:251-340
[^3]: pyproject.toml:47-52; tests/test_import_contracts.py:12-85


## Coordinator snapshot claim hydration

Coordinator snapshot claim hydration restores persisted goal-claim fields after reachable-node traversal.[^coordinator-claim-hydration]
Coordinator projection builds root-first reachable IDs with `subtree_post_order` and removes duplicate IDs.
Public `graph.get_nodes` then supplies node values, including descendant GOAL `goal_run_id` fields.[^coordinator-claim-hydration]
Real SQLite tests cover a shared diamond node and a claimed descendant.[^coordinator-claim-tests]

These facts describe PR #519 corrections integrated after PR #518.
The separately approved command decoder bounds task and revision node identifiers to signed SQLite integers.[^coordinator-api-boundaries]
Graph initialization creates web receipts at schema step 47 after unchanged coordinator steps 38–46.[^coordinator-api-schema]
The integrated full gate and 121 focused tests pass. Publication remains pending.

[^coordinator-claim-hydration]: `src/milknado/domains/coordinator/projection.py:57-63` in the retained local PR #519 correction.
[^coordinator-claim-tests]: `tests/coordinator/test_projection_updates.py:20-44` in the retained local PR #519 correction.

[^coordinator-api-boundaries]: src/milknado/domains/coordinator/control_models.py:10,27-35,85-94; tests/coordinator/test_sqlite_identifier_guards.py:1-102; separately approved resume command, 2026-10-09.
[^coordinator-api-schema]: src/milknado/domains/graph/_coordinator_schema.py:47-61; tests/coordinator/test_web_receipt_schema.py:1-58.

_Source: PR #519 integration b999ed266fc3d256dd2c4132524105f4400d0434, pending publication · Updated: 2026-10-09 · Supersedes: 2026-10-08 pending guard approval and schema ordering._
