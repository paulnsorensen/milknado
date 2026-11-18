# Goal-first web spike

Open [index.html](index.html) in a browser from your checkout. No server, build, login, or network access is necessary.

- Select **Complete A runtime**. The draft appears, but B stays blocked.
- Select **Accept A evidence**. B becomes ready. The demo does not start B.
- Reset. Request cancellation, then disconnect. The stop remains unconfirmed, and the runtime becomes unknown.
- Load the replacement fixture and replay the old completion. A-02 stays active and has no evidence.
- Answer the fixture question. Provider permission stays pending. Use T3 itself for a real permission decision.

Run the browser checks with existing project dependencies and Chromium:

```sh
UV_PYTHON=3.12 uv run pytest -m browser -q tests/spikes/test_t3_web_demo.py
```

## Proposed server boundary

The browser would read a projection from Milknado's server, not contact T3 MCP.
Milknado would own dependencies, review, and accepted evidence.
A server-side adapter would bind each attempt to runtime identifiers and reconcile observations for that attempt.
An observation from an old attempt would not change the active attempt.
A disconnect or cancellation acknowledgment would not prove that work ended.
Provider permission would stay in T3 unless an authorized integration exposes that capability separately from user questions.

The replacement control loads a separate fixture; it does not authorize retry after a disconnect.
No endpoint, credential, start, retry, cancel, or permission operation is implemented.
The trial still needs real runtime reconciliation, ownership checks, and a decision about supported operations.
