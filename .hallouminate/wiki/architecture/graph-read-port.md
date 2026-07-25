# Graph read port and execution snapshots

`GraphReadPort` is the supported cross-slice read boundary: traversal and dispatch consumers use node, child, and file-ownership reads, while execution receives one atomic `GraphExecutionSnapshot`. The port keeps consumers independent of `MikadoGraph` and graph storage internals.[^1]

Execution, not the graph crust, converts snapshot facts into dispatchable-node counts and the execution overview. The graph snapshot holds the lock and a SQLite savepoint across root, requested-node, ready/running, and conflict facts, preserving a cross-connection-consistent read without exposing a connection or private read module.[^2]

The import-linter contract intentionally forbids graph → execution imports. This protects the ownership direction: graph stores transactional state; execution owns scheduling policy and presentation assembly.[^3]

[^1]: src/milknado/domains/common/protocols.py:36-69; src/milknado/domains/graph/traversals.py:3-34; src/milknado/domains/dispatch/brief.py:8-172
[^2]: src/milknado/domains/graph/graph.py:350-373; src/milknado/domains/execution/executor.py:217-249; tests/test_graph_atomicity.py:251-340
[^3]: pyproject.toml:47-52; tests/test_import_contracts.py:12-85
