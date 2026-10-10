# Coordinator polling session ownership

Coordinator polling effects must match their rendered session ID before capturing the current request scope.[^1]
This prevents an old effect from owning a newly selected session's snapshot request.

## Selection and effect timing

Coordinator session selection replaces the mutable request scope before React runs the next passive polling effect.[^2]
A pending effect can retain session A's render while `scope.current` already holds session B.
Without the ID check, A's effect captures B's scope.
Its cleanup then marks B inactive and aborts B's snapshot request.
The new selection cannot display its goal, although its request starts.

## Ownership check

The coordinator polling effect returns when `scope.current.id !== sessionId`.[^1]
The check runs before `const currentScope = scope.current`.
Only the effect for the selected session captures its scope.
Existing request counters, active-scope checks, abort checks, and stale-response checks remain.[^3]
The polling interval remains two seconds.
Public hook fields and command receipts remain unchanged.

## Controlled regression

The coordinator regression renders the real `CoordinatorCockpit` inside a React `Profiler`.[^4]
The profiler selects session B when discovery commits its selection button.
This schedules selection before the pending session-A polling effect starts.
The assertion requires the visible `Goal session-b` header.
The test also confirms that selection occurs.

The unchanged controlled test fails five of five runs without the ownership check.
It passes five of five runs with the check.
An independent replay passes all five file tests in each of five runs.
These results confirm this controlled effect-ordering defect, not every possible scheduling race.
The HTTP fixture replaces transport only; it does not replace the cockpit or hook.

## Related boundaries

Coordinator backend recovery separately revalidates session ownership before persisting results.
See [Execution and Dispatch](../architecture/execution.md).
The frontend check does not replace that backend boundary.
After changing dashboard source, rebuild the committed static assets with `npm --prefix web run build`.[^5]

_Source: PR #525 approved polling repair and controlled regression · Updated: 2026-10-10 · Supersedes: none_

[^1]: web/src/features/coordinator/useCoordinatorSession.ts:97-114
[^2]: web/src/features/coordinator/useCoordinatorSession.ts:54-65
[^3]: web/src/features/coordinator/useCoordinatorSession.ts:68-94
[^4]: web/src/features/coordinator/CoordinatorMissingSession.test.tsx:111-129
[^5]: AGENTS.md:75-84
