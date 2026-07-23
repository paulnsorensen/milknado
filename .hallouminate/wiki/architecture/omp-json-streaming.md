# OMP JSON Event Streaming

Milknado launches Oh My Pi with `--mode json`: newline-delimited structured events make live execution observable without changing the one-shot positional-prompt contract.

## Decision

`OmpAdapter.build_command` replaces any user-supplied `--mode` with `--mode json` before `deliver_prompt` appends the worker brief (`src/milknado/loop/adapters/omp.py:22`). This is a deliberate clean cutover: text mode yields only transient `Working...` status while JSON mode streams agent lifecycle, message, and tool events.

The loop keeps the current full-stdout completion scan because Milknado's completion promise remains part of the agent text, not a separate OMP terminal result event. `tool_execution_start` is parsed solely to preserve the existing `max_turns` tool-use cap (`src/milknado/loop/adapters/omp.py:47`).

## TUI consequence

The run controller already retains bounded worker output and `ExecutionApp` renders those lines directly (`src/milknado/domains/execution/run_loop/__init__.py:158`; `src/milknado/app/run_tui.py:226`). JSON mode therefore makes live structured frames available immediately. A future renderer can normalize selected event types without another subprocess-protocol migration.

## Why not RPC yet

OMP RPC is bidirectional: the host must wait for `ready`, send a JSON `prompt` command through stdin, and handle command correlation. `rpc-ui` additionally requires host responses to interactive UI frames. Neither capability is needed for passive TUI observability, so the positional-prompt worker contract stays intact.

## Sources

- [OMP print/JSON mode implementation](https://github.com/can1357/oh-my-pi/blob/7504d4c24da76b41e62f014c96bf2da64ad8dc50/packages/coding-agent/src/modes/print-mode.ts)
- [OMP RPC protocol](https://github.com/can1357/oh-my-pi/blob/main/docs/rpc.md)
- [OMP RPC event definitions](https://github.com/can1357/oh-my-pi/blob/7504d4c24da76b41e62f014c96bf2da64ad8dc50/python/omp-rpc/src/omp_rpc/protocol.py)
