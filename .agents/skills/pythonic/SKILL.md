---
name: pythonic
description: Write, edit, refactor, or review Python with concise declarative contracts, idiomatic control flow, Sliced Bread boundaries, and the Python de-slop rules. Use for Python implementation work in this repository, especially when the user asks for Pythonic, succinct, LLM-optimized, de-slopped, msgspec, dataclass, pattern-matching, or vertical-slice code.
---

# Pythonic

Produce the smallest readable Python change that satisfies the request and matches the repository.

## Work in this order

1. Read the root configuration, target exports, immediate callers, and nearby conventions.
2. Keep every changed line traceable to the request; do not invent orchestration, tests, dependency-injection rules, compatibility layers, or future flexibility.
3. Put the change in the owning Sliced Bread slice and preserve dependency direction.
4. Express boundary schemas and internal data with the appropriate declarative type.
5. Choose the clearest succinct Python construct; do not compress code until it becomes harder to read.
6. Remove Python and cross-language slop introduced or exposed by the change.
7. Run `just check-llm` as the complete project gate.

## Model data declaratively

- Use frozen `msgspec.Struct` models for untrusted external input, serialized I/O, API payloads, configuration boundaries, and LLM tool schemas.
- Validate every untrusted value with a typed decoder or `msgspec.convert(..., type=..., strict=True)`; direct Struct construction is only for trusted internal values.
- Keep input normalization narrow and explicit before typed conversion when the wire grammar supports shorthand.
- Use dataclasses for trusted internal records that need named fields but not boundary parsing.
- Use domain value types when invariants belong to the business concept rather than transport validation.
- Fully annotate function and public-method signatures. Omit obvious local annotations unless inference cannot determine the intended type.
- Do not pass structured data as raw dictionaries when a model or dataclass makes the contract clearer.

## Prefer succinct, readable Python

- Use `match` for shape-based payload, event, command, or AST dispatch. Keep a simple `if` when it communicates a binary decision better.
- Use handler mappings for stable command-to-function dispatch. Do not create a registry for one or two branches.
- Use assignment expressions only when they eliminate a repeated computation or make a loop condition clearer.
- Use `any`, `all`, comprehensions, and generator expressions for pure collection queries or transformations. Keep a loop when it carries state, side effects, or clearer early exits.
- Prefer generators when the result is consumed once.
- Use `enumerate`, direct iteration, f-strings, context managers, `pathlib`, `itertools`, and `collections` instead of manual equivalents.
- Use `contextlib.suppress(SpecificError)` only when ignoring that exact failure is intended. Never use `suppress(Exception)` or silent `except` blocks; otherwise let the error propagate or handle it meaningfully.

## Preserve Sliced Bread architecture

Organize by business capability, not technical layer:

- Keep a feature inside its owning `src/milknado/domains/<slice>/` package.
- Treat the slice `__init__.py` as its crust: consumers import public contracts from the crust, never sibling internals.
- Let domain slices depend on the standard library, their own internals, shared domain values, and other slices' crusts.
- Define external-service protocols in the domain; implement them under `src/milknado/adapters/`.
- Put orchestration spanning multiple slices in the application or entry-point seam, not inside an adapter.
- Never import adapters or application entry points from a domain, and never import application orchestration from an adapter.
- Use domain events to avoid reverse dependencies or cycles.
- Grow structure only under demonstrated pressure: do not add factories, registries, abstract bases, ports, directories, or shared utilities for a single use.

## De-slop before finishing

- Delete comments and docstrings that restate the code; keep non-obvious rationale and public API documentation.
- Propagate failures instead of returning empty defaults or swallowing exceptions.
- Remove abstractions with one concrete consumer unless the current task requires the seam.
- Name values after domain concepts, not containers or implementation types.
- Remove unused imports, dead alternatives, speculative branches, and boilerplate.
- Consolidate repetitive tests by behavior; do not add shallow input-variation tests.
- Fix the underlying lint issue instead of adding a suppression.
- Prefer one clear expression to verbose scaffolding, but split dense expressions when intermediate names explain domain intent.

## Completion check

Confirm:

- Boundary data is validated once by typed msgspec decode or strict conversion; direct Struct construction never validates untrusted annotations.
- Internal records use dataclasses or domain types where useful.
- Slice imports cross only public crusts and point toward domains.
- Control flow is succinct without being clever.
- No silent failures, speculative abstractions, raw structured dictionaries, narration comments, or unnecessary local annotations remain in the changed code.
- `just check-llm` prints its PASS line.