---
name: load-roadmap
description: >
  Import a hallouminate-wiki roadmap into milknado so its goals become executable
  nodes. Use when the user says "load the roadmap", "import roadmap <slug>", "pull
  the roadmap into milknado", "seed milknado from the wiki", or "/load-roadmap".
  Calls milknado_roadmap_import to seed one roadmap node plus one goal node per
  goal file (deterministic wiki_ref keys, idempotent re-import, no task nodes),
  then optionally kicks milknado_plan_batches to decompose goals into tasks. Do
  NOT use to write execution results back to the wiki (that is /harvest) or to
  author a brand-new roadmap from scratch.
---

# load-roadmap

Pull a wiki roadmap into milknado and (optionally) decompose its goals into tasks.

## When to use

- The user has a roadmap dir under `<project>/.hallouminate/wiki/roadmaps/<slug>/`
  and wants milknado to execute it.
- They say "load", "import", or "seed" a roadmap into milknado.

**Not for:** harvesting results back to the wiki (`/harvest`), or authoring a new
roadmap from scratch.

## The model in 30 seconds

The wiki owns roadmap + goal **intent**; milknado owns task **execution state**.
The goal file is the membrane: import reads `Intent`/`Acceptance`, never writes
them back. Each goal's durable identity is a content-addressed `wiki_ref =
uuid5(roadmap_slug/goal_slug@created)` — identical on every machine, so the same
roadmap imported into two dbs yields the same keys and re-import is idempotent.

## Steps

1. Confirm the roadmap slug (the directory name under `roadmaps/`).
2. Call `milknado_roadmap_import(roadmap_slug=<slug>)` (pass `project_root` if not
   the cwd). This seeds the roadmap node + one goal node per `<goal>.md`, wires
   prereqs from frontmatter, and stamps `created` into any goal file missing it.
   It seeds the **goal layer only** — no task nodes.
3. Report the result: created / reused counts and the goal node ids.
4. If the user wants execution to start, decompose a goal into tasks with
   `milknado_plan_batches` (a separate step — import never creates task nodes).

## Notes

- Re-running import is safe: the `UNIQUE(wiki_ref)` index makes it find-or-create,
  not insert-always.
- Fails fast with a clear message if the roadmap dir or `.hallouminate/wiki/` is
  absent — no partial graph is written.
- The serialization is tested milknado core (`milknado roadmap import <slug>` /
  `milknado_roadmap_import`); this skill is a thin veneer over it.
