# OMP JSON Event Streaming

Native OMP worker sessions use RPC. One-shot execution keeps the JSON event adapter.
The two paths have different input and completion contracts.

## One-shot JSON adapter

`OmpAdapter.build_command` replaces a configured output mode with `--mode json`.
The one-shot path keeps positional prompt delivery (`src/milknado/loop/adapters/omp.py:26`).
Its event parser counts `tool_execution_start` events for the tool-use limit.

## Native RPC sessions

Native OMP sessions accept steering and follow-up input through a bidirectional protocol.
The session driver sends prompt commands, handles readiness and protocol negotiation, and correlates command replies
(`src/milknado/loop/sessions/_omp.py:17`; `src/milknado/loop/sessions/_omp.py:85`).
Native worker commands use the session runtime rather than the one-shot adapter's completion scan
(`src/milknado/loop/engine.py:216`).

Native completion depends on decoded result text, never raw stdout.
A replayed instruction can contain a completion tag without completing any work.
The engine keeps completion verification and quality gates authoritative (`src/milknado/loop/engine.py:235`).

## Session consumers

Session views retain decoded user, assistant, and tool events with input receipts.
Queued and submitted inputs do not prove vendor delivery.
The controller exposes the session view alongside bounded legacy output (`src/milknado/app/run.py:321`).
See [[execution]] for receipt persistence, invocation identity, and process cleanup.

## Sources

- [OMP print/JSON mode implementation](https://github.com/can1357/oh-my-pi/blob/7504d4c24da76b41e62f014c96bf2da64ad8dc50/packages/coding-agent/src/modes/print-mode.ts)
- [OMP RPC protocol](https://github.com/can1357/oh-my-pi/blob/main/docs/rpc.md)
- [OMP RPC event definitions](https://github.com/can1357/oh-my-pi/blob/7504d4c24da76b41e62f014c96bf2da64ad8dc50/python/omp-rpc/src/omp_rpc/protocol.py)

_Source: native-session implementation and real-pipe regressions · Updated: 2026-09-09 · Supersedes: raw-stdout completion for native worker sessions._
