---
name: harvest
description: >
  Harvest milknado execution state back into the hallouminate-wiki roadmap goal
  files. Use when the user says "harvest the roadmap", "export results to the
  wiki", "sync milknado back to the wiki", "update the goal outcomes", "write the
  outcomes back", or "/harvest". Calls milknado_roadmap_export to rewrite each
  goal's harvest block plus status/last_synced, leaving human-authored
  Intent/Acceptance byte-identical; a goal node milknado discovered mid-run gets a
  fresh goal file created for it, and `hallouminate index` is refreshed
  best-effort. Do NOT use to import a roadmap into milknado (that is
  /load-roadmap) or to edit Intent/Acceptance text.
---

# harvest

Write milknado's execution outcomes back into the wiki goal files, without
touching the human-authored intent.

## When to use

- A roadmap has been imported and run (or partially run) in milknado, and the
  user wants the wiki goal files updated with the results.
- They say "harvest", "export results", "sync back", or "update outcomes".

**Not for:** importing a roadmap (`/load-roadmap`) or editing the human-owned
`Intent`/`Acceptance` sections.

## The membrane guarantee

Export overwrites **only**:

- the bytes between `<!-- milknado:harvest:start -->` / `:end`, and
- the `status` + `last_synced` frontmatter fields.

`Intent` and `Acceptance` are byte-identical before and after — the two writers
(human, milknado) never share bytes, so there is no merge conflict by
construction.

## Steps

1. Confirm the roadmap slug.
2. Call `milknado_roadmap_export(roadmap_slug=<slug>)` (pass `project_root` if not
   the cwd). For each goal node under the roadmap it rebuilds the harvest block
   from goal status, child-task rollup (N done / M failed), `deposit_result`
   summaries, and branch links.
3. Report the result: files updated, files created (orphan goal nodes), and
   whether the search index was refreshed.

## Notes

- A goal node with no matching wiki file (discovered mid-run) gets a new
  `<goal-slug>.md` created, with `Intent` seeded from the node brief and the
  Outcome harvested — so it round-trips on the next export.
- `hallouminate index` is best-effort: export still succeeds and writes every
  file even when the hallouminate CLI/daemon is absent (search just refreshes
  late).
- The serialization is tested milknado core (`milknado roadmap export <slug>` /
  `milknado_roadmap_export`); this skill is a thin veneer over it.
