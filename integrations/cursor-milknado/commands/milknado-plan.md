---
name: milknado-plan
description: Run milknado plan with a goal (Mikado decomposition)
---

# milknado plan

Run from the project root where `milknado.toml` lives:

```bash
uv run milknado plan "<goal>" --project-root .
```

Replace `<goal>` with the user’s decomposition target. Ensure `milknado init` and `milknado index` have been run if this is a new tree.
