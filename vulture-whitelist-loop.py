"""Vulture whitelist for the vendored ``loop/`` engine — NOT dead code we own.

``src/milknado/loop/`` is vendored wholesale from ralphify (PR #137, ``49ac978``);
``loop/__init__.py`` pins the source as fork ``paulnsorensen/ralphify`` at
``ec494875d0309c273c03f5f52a2cc3fabc96fa16`` and declares its own boundary:
"Public API — import only these six names". Milknado drives a create / start /
observe / stop lifecycle through ``adapters/loop.py`` and uses nothing beyond it,
so the rest of the upstream surface reads as unused.

It is exempt rather than deleted because it is not ours to delete. Trimming forks
the vendored copy away from its own upstream, and the manager methods cascade:
removing ``add_listener`` orphans ``FanoutEmitter``, ``get_result`` orphans
``RunResult``, ``shutdown`` orphans ``reap``, and ``pause_run`` / ``resume_run``
orphan ``RunState.request_pause`` / ``request_resume`` and ``RunStatus.PAUSED``.

``src/milknado/loop/`` deliberately stays inside ``[tool.vulture] paths``: only the
names enumerated here are exempt, so genuinely new dead code in the engine still
fails the gate. A blanket ``--exclude`` was considered and rejected for that reason.

Do not extend this file to cover code milknado authors — that belongs deleted or
connected. Regenerate with ``uv run vulture --make-whitelist``. Delete the file
wholesale if the package is ever re-vendored or trimmed for real.

This module is parsed by vulture, never imported or executed.
"""

# --- loop/_events.py: TypedDict payload keys written by engine.py, plus Event.to_dict.
# Deleting a key does not stop the engine writing it; it only makes the schema wrong.
ralph_name  # unused variable (src/milknado/loop/_events.py:99)
duration_formatted  # unused variable (src/milknado/loop/_events.py:123)
echo_stdout  # unused variable (src/milknado/loop/_events.py:127)
echo_stderr  # unused variable (src/milknado/loop/_events.py:128)
prompt_length  # unused variable (src/milknado/loop/_events.py:143)
_.to_dict  # unused method (src/milknado/loop/_events.py:229)

# --- loop/_frontmatter.py: the RALPH.md frontmatter schema.
# Vestigial in milknado: engine.py:189 does `_, prompt = parse_frontmatter(raw)`,
# discarding the dict and keeping only the body, while RunConfig is built
# programmatically by adapters/loop.py. Only FIELD_AGENT, FIELD_ARGS, FIELD_COMMANDS,
# FIELD_RALPH, NAME_RE, and RALPH_MARKER are live.
FIELD_COMPLETION_SIGNAL  # unused variable (src/milknado/loop/_frontmatter.py:29)
FIELD_STOP_ON_COMPLETION_SIGNAL  # unused variable (src/milknado/loop/_frontmatter.py:30)
FIELD_MAX_TURNS  # unused variable (src/milknado/loop/_frontmatter.py:33)
FIELD_MAX_TURNS_GRACE  # unused variable (src/milknado/loop/_frontmatter.py:34)
FIELD_HOOKS  # unused variable (src/milknado/loop/_frontmatter.py:37)
CMD_FIELD_NAME  # unused variable (src/milknado/loop/_frontmatter.py:40)
CMD_FIELD_RUN  # unused variable (src/milknado/loop/_frontmatter.py:41)
CMD_FIELD_TIMEOUT  # unused variable (src/milknado/loop/_frontmatter.py:42)
HOOK_FIELD_EVENT  # unused variable (src/milknado/loop/_frontmatter.py:45)
HOOK_FIELD_RUN  # unused variable (src/milknado/loop/_frontmatter.py:46)
VALID_NAME_CHARS_MSG  # unused variable (src/milknado/loop/_frontmatter.py:57)
serialize_frontmatter  # unused function (src/milknado/loop/_frontmatter.py:141)

# --- loop/_output.py: sibling of the live format_duration.
format_count  # unused function (src/milknado/loop/_output.py:95)

# --- loop/adapters/: protocol flag declared on CLIAdapter and set by all six
# adapters, but read nowhere — the structured-vs-raw decision is made by the
# emitter at engine.py:370 via emit.wants_agent_output_lines().
renders_structured_peek  # unused variable (src/milknado/loop/adapters/_generic.py:27)
renders_structured_peek  # unused variable (src/milknado/loop/adapters/_protocol.py:85)
renders_structured_peek  # unused variable (src/milknado/loop/adapters/claude.py:58)
renders_structured_peek  # unused variable (src/milknado/loop/adapters/codex.py:74)
renders_structured_peek  # unused variable (src/milknado/loop/adapters/copilot.py:57)
renders_structured_peek  # unused variable (src/milknado/loop/adapters/crush.py:60)
renders_structured_peek  # unused variable (src/milknado/loop/adapters/opencode.py:67)

# --- loop/manager.py: RunManager surface the architecture does not want.
# adapters/loop.py drains the event queue itself instead of blocking; concurrency
# and retry live in domains/execution/; there is no pause/resume UI or fan-out.
_.add_listener  # unused method (src/milknado/loop/manager.py:43)
_.pause_run  # unused method (src/milknado/loop/manager.py:160)
_.resume_run  # unused method (src/milknado/loop/manager.py:164)
_.wait_for_any  # unused method (src/milknado/loop/manager.py:197)
_.wait_for_all  # unused method (src/milknado/loop/manager.py:221)
_.get_result  # unused method (src/milknado/loop/manager.py:245)
_.shutdown  # unused method (src/milknado/loop/manager.py:280)
