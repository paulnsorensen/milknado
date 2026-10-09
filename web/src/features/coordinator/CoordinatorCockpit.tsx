import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, selectedNodeId, subscribe } from '../../app/store';
import { EvidencePanel, ProposalPanel } from './CoordinatorPresentation';
import { useCoordinatorSession } from './useCoordinatorSession';

export function CoordinatorCockpit(): ReactElement {
  const store = useSyncExternalStore(subscribe, getState);
  const nodeId = selectedNodeId(store);
  const { available, busy, commandResult, goal, provider, sessionId, sessions, snapshot,
    selectSession, send, setGoal, setProvider, start } = useCoordinatorSession();

  if (!available) return <></>;

  return (
    <aside className="mk-coordinator-cockpit" aria-label="Coordinator cockpit">
      <nav aria-label="Coordinator sessions">
        {sessions.map((session) => <button key={session.id} aria-current={session.id === sessionId ? 'page' : undefined}
          onClick={() => selectSession(session.id)}>{session.description} · {session.provider}</button>)}
        {sessionId && <button onClick={() => selectSession('')}>New goal</button>}
      </nav>
      {!sessionId && (
        <form onSubmit={(event) => void start(event)}>
          <label>Goal <input value={goal} onChange={(event) => setGoal(event.target.value)} required /></label>
          <label>Provider <select value={provider} onChange={(event) => setProvider(event.target.value as 'claude' | 'codex')}>
            <option value="claude">Claude</option><option value="codex">Codex</option>
          </select></label>
          <button disabled={busy} type="submit">Start goal</button>
        </form>
      )}
      {snapshot && (
        <>
          <header><strong>{snapshot.goal.description}</strong> <span>{snapshot.session.provider}</span></header>
          <div className="mk-coordinator-actions">
            <button disabled={busy} onClick={() => void send('plan_goal')}>Propose plan</button>
            <button disabled={busy} onClick={() => void send('recover')}>Recover</button>
          </div>
          <p>Session {snapshot.session.id}</p>
          <p>Graph: {snapshot.nodes.length} nodes · {snapshot.runs.length} runs</p>
          {commandResult && <p role="status">{commandResult}</p>}
          <p>Recovery: {snapshot.recovery.at(-1)?.status ?? 'not recorded'}</p>
          <ProposalPanel proposals={snapshot.proposals} busy={busy} send={send} />
          <EvidencePanel snapshot={snapshot} nodeId={nodeId} busy={busy} send={send} />
        </>
      )}
    </aside>
  );
}
