# OMP planning decision

Implement first-class OMP planning (#305); do not add raw agent-command passthrough (#283). The executable allowlist deliberately requires an exact bare command token, and every execution entrypoint validates it before spawning. Allowing an arbitrary command from project configuration would turn a repository-local `milknado.toml` into a direct host-process launch surface.[^1]

OMP already has a concrete worker adapter: it forces JSON event mode, preserves OMP's positional prompt convention, and parses tool events. Planning should add the equivalent explicit adapter branch: normalize OMP to read-only `read,grep,glob,lsp`, strip approval-escalation and code-loading flags, and keep normal planning stdin input/output semantics.[^2]

The planning subprocess environment is intentionally minimal and currently forwards no credentials. For the OpenRouter contract, forward only `OPENROUTER_API_KEY` for the OMP planning adapter; do not make an environment-variable allowlist configurable by the project. This narrows the credential exposure, but OMP planning must also disable extensions, plugins/config overlays, skills, rules, and PTY so untrusted project configuration cannot execute code in the credential-bearing process.[^3]

OMP execution remains explicit: a project supplies its own `execution_agent` with its chosen OMP tool set. Do not translate the existing Claude/Gemini worker-tool schema to OMP or silently fall back to unrestricted OMP tools. Resume support must parse OMP's camel-case `sessionId` from its JSON header/event output and use `omp --resume <id>`.[^4]

[^1]: src/milknado/domains/common/agent_argv.py:83-108; src/milknado/domains/dispatch/runner.py:45-62; src/milknado/loop/engine.py:207-216
[^2]: src/milknado/loop/adapters/omp.py:12-60; src/milknado/domains/common/agent_argv.py:202-257; https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/DEVELOPMENT.md
[^3]: src/milknado/domains/common/agent_argv.py:260-263; https://github.com/can1357/oh-my-pi/blob/main/docs/environment-variables.md
[^4]: src/milknado/domains/common/agent_argv.py:118-162,277-319; https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/DEVELOPMENT.md
