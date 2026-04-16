---
name: milknado-plan
description: Run milknado plan with a markdown spec (Mikado decomposition)
---

# milknado plan

Run from the project root where `milknado.toml` lives.

```bash
uv run milknado plan --spec specs/feature.md --project-root .
```

Replace `specs/feature.md` with your spec path. Ensure `milknado init` and `milknado index` have been run if this is a new tree.
