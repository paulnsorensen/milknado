# OMP JSON Event Streaming

Milknado launches Oh My Pi with `--mode json`: newline-delimited structured events make live execution observable without changing the one-shot positional-prompt contract.

## Decision

`OmpAdapter.build_command` replaces any user-supplied `--mode` with `--mode json` before `deliver_prompt` appends the worker brief (`src/milknado/loop/adapters/omp.py:22`). This is a deliberate clean cutover: text mode emits its final response after the run (with transient `Working...` on stderr), while JSON mode streams agent lifecycle, message, and tool events.

The loop keeps the current full-stdout completion scan because Milknado's completion promise remains part of the agent text, not a separate OMP terminal result event. `tool_execution_start` is parsed solely to preserve the existing `max_turns` tool-use cap (`src/milknado/loop/adapters/omp.py:47`).

## TUI consequence

The run controller retains bounded worker output and `ExecutionApp` renders those lines directly (`src/milknado/domains/execution/run_loop/__init__.py:158`; `src/milknado/app/run_tui.py:226`). JSON activity is emitted through the lower-level loop event emitter, but it is not currently projected into the execution TUI; captured JSON output appears in the run view after the worker iteration returns. A future renderer can normalize selected event types without another subprocess-protocol migration.

## Why not RPC yet

OMP RPC is bidirectional: the host must wait for `ready`, send a JSON `prompt` command through stdin, and handle command correlation. `rpc-ui` additionally requires host responses to interactive UI frames. Neither capability is needed for the lower-level event emitter, but wiring those events into the TUI is a separate presentation change, so the positional-prompt worker contract stays intact.

## Sources

- [OMP print/JSON mode implementation](https://github.com/can1357/oh-my-pi/blob/7504d4c24da76b41e62f014c96bf2da64ad8dc50/packages/coding-agent/src/modes/print-mode.ts)
- [OMP RPC protocol](https://github.com/can1357/oh-my-pi/blob/main/docs/rpc.md)
- [OMP RPC event definitions](https://github.com/can1357/oh-my-pi/blob/7504d4c24da76b41e62f014c96bf2da64ad8dc50/python/omp-rpc/src/omp_rpc/protocol.py)
