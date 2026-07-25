# Graph status reconciliation

Todo mutations preflight their complete facade transition before changing a node: invalid transitions raise the stable `InvalidTransition` contract and verification only accepts a JSON object with boolean `ok: true`. Subtree traversal is iterative post-order, so shared descendants are changed once and cycles or deep valid graphs terminate safely.[^1]

At `milknado run` startup, pending goals with existing direct children all marked `DONE` are reconciled through the status state machine. The pass repeats upward so an already-complete nested goal can make its parent eligible. A structurally invalid goal emits a warning and its subtree is excluded from that run; unrelated roots still dispatch.[^2]

[^1]: src/milknado/domains/graph/status_flow.py:18-91
[^2]: src/milknado/domains/graph/_status.py:187-210; src/milknado/cli/run.py:135-155
