# Path contracts

`domains.common.paths` is the single crust for project-local path normalization: `resolve_project_path` follows filesystem resolution and confines ordinary values to the resolved project root. Absolute paths, traversal, and symlink escapes fail; `TrustedGlobalPath`, created only by `trust_global_path`, explicitly carries the global-config exception.[^1]

Resolution validates the target at the instant it runs. It is not a promise about a later open or write; security-sensitive wiki I/O uses descriptor-relative, no-symlink operations at the use point instead.[^2]

`slugify` is lexical only: blank or punctuation-only input is the empty string, and truncation is deterministic but never makes a slug unique. Naming callers supply their own fallback or collision strategy. Planning owns its separate materialized-spec identifier normalizer because its filenames preserve dots and underscores.[^3]

[^1]: src/milknado/domains/common/paths.py:35-68
[^2]: src/milknado/domains/wiki/_locate.py:89-151
[^3]: src/milknado/domains/common/paths.py:7-15; src/milknado/domains/planning/source_material.py:20-27
