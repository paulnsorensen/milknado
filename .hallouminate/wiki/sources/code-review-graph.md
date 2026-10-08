# code-review-graph

code-review-graph is the PyPI package whose published metadata defines its released FastMCP dependency range.
Publisher: PyPI. Source type: package metadata. Last verified: 2026-10-08.
Canonical source: [code-review-graph](https://pypi.org/pypi/code-review-graph/2.3.9/json)

## Released dependency contract

The code-review-graph 2.3.9 package metadata requires `fastmcp>=3.2.4,<4`.
A Milknado manifest requiring FastMCP 4 has no compatible intersection with this released requirement.
Milknado keeps its existing CRG adapter and selects `fastmcp>=3.4.8,<4` for PR 515.
The manifest and regenerated lock must agree before the locked CodeQL install runs.

## Evidence limits

The code-review-graph metadata proves resolver compatibility, not runtime behavior.
Milknado's locked install and `just check-llm` provide the separate local runtime evidence.
An unreleased upstream proposal does not supersede the published dependency contract.

_Source: PyPI package metadata and PR #515 local verification · Updated: 2026-10-08 · Supersedes: none_
