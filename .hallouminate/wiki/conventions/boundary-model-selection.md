---
status: reviewed
last_verified: 2026-07-26
confidence: high
sources:
  - .agents/skills/pythonic/SKILL.md
  - pyproject.toml
  - https://msgspec.dev/
---
# Boundary model selection

Milknado uses frozen `msgspec.Struct` models for untrusted external input, serialized I/O, API payloads, configuration boundaries, and LLM tool schemas. Every untrusted value must enter through a typed decoder or `msgspec.convert(..., type=..., strict=True)`; direct Struct construction is reserved for trusted internal values.[^1]

## Why msgspec

msgspec keeps wire contracts concise while providing strict typed conversion, frozen structs, annotated constraints, post-init validation, nested conversion, and unknown-field handling. It fits Milknado's preference for minimalist typed Python without requiring Milknado to ship a C or Rust extension of its own.[^2]

Normal `Struct(...)` construction does not runtime-check field annotations. The typed decode/convert rule is therefore a safety requirement rather than a style preference.[^3]

## Boundary patterns

- Normalize only documented shorthand before strict typed conversion; configuration gate strings and tool lists follow this pattern.[^4]
- Put cross-field and semantic invariants in `__post_init__`, while retaining stable public error behavior at adapters.[^5]
- Use dataclasses for trusted internal records that need named fields but no boundary parsing.
- Preserve existing YAML and TOML libraries: msgspec's YAML support uses PyYAML, and TOML writing still uses tomli-w.[^6]

Pydantic may remain transitively installed through FastMCP or MCP dependencies, but Milknado has no direct dependency or application imports.[^7]

[^1]: `.agents/skills/pythonic/SKILL.md:20-27,63-68`.
[^2]: `pyproject.toml:12-23`; `src/milknado/domains/github/models.py:9-77`; `src/milknado/domains/planning/manifest.py:39-125`.
[^3]: https://msgspec.dev/structs#type-validation.
[^4]: `src/milknado/domains/common/config.py:179-250`; `src/milknado/domains/common/flavor_codec.py:35-78,218-238`.
[^5]: `src/milknado/domains/common/config.py:92-173`; `src/milknado/adapters/gh.py:83-89`.
[^6]: https://msgspec.dev/install#optional-dependencies.
[^7]: `pyproject.toml:12-23`; repository search for `from pydantic|import pydantic` returned zero Python matches on 2026-07-26.
