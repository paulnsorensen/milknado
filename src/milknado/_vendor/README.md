# Vendored assets

## `tiktoken-cache/`

A pre-downloaded copy of the `cl100k_base` tiktoken encoding, committed
intentionally so token-counting works in offline / sandboxed CI without a
network round-trip to `openaipublic.blob.core.windows.net`.

- The filename is the SHA-1 of the source blob URL — the exact key tiktoken
  looks up in `TIKTOKEN_CACHE_DIR`. Do not rename it.
- Source URL and key derivation live in
  `src/milknado/domains/batching/weights.py` (`TIKTOKEN_BLOB_URL`,
  `TIKTOKEN_CACHE_KEY`). `_get_encoder()` points `TIKTOKEN_CACHE_DIR` here
  when the cached blob is present.
- Dependabot and Scorecard do not track this file; refresh it manually if the
  upstream encoding ever changes (it is stable for `cl100k_base`).

Trade-off: ~1.6 MB of git history in exchange for hermetic, network-free
tokenisation. If that footprint becomes a problem, replace this with a
CI-time download step keyed on the same cache dir.
