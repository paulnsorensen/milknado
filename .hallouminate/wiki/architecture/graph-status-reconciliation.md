# Graph status reconciliation

Todo mutations preflight their complete facade transition before changing a node: invalid transitions raise the stable `InvalidTransition` contract. Both single-node and subtree `DONE` transitions validate the latest verification inside the graph state producer, which accepts only a JSON object with boolean `ok: true`. The superseded `status_flow` verification facade was removed; malformed and non-object payload tests now exercise the producer. Subtree traversal remains iterative post-order, so shared descendants are changed once and cycles or deep valid graphs terminate safely.[^1]

At `milknado run` startup, pending goals with existing direct children all marked `DONE` are reconciled through the status state machine. The pass repeats upward so an already-complete nested goal can make its parent eligible. A structurally invalid goal emits a warning and its subtree is excluded from that run; unrelated roots still dispatch.[^2]

[^1]: src/milknado/domains/graph/_status.py:36-73,155-185; src/milknado/domains/graph/status_flow.py:18-68; tests/test_mcp_node.py:560-574
[^2]: src/milknado/domains/graph/_status.py:188-211; src/milknado/cli/run.py:135-155
