import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, selectedNodeId, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { EvidencePanel, ProposalPanel } from './CoordinatorPresentation';
import { useCoordinatorSession } from './useCoordinatorSession';

export function CoordinatorCockpit(): ReactElement {
  const store = useSyncExternalStore(subscribe, getState);
  const nodeId = selectedNodeId(store);
  const { Button } = Milknado;
  const { available, busy, commandResult, goal, provider, sessionId, sessions, snapshot,
    selectSession, send, setGoal, setProvider, start } = useCoordinatorSession();

  if (!available) return <></>;

  return (
    <aside className="mk-coordinator-cockpit" aria-label="Coordinator cockpit">
      <header className="mk-coordinator-head">
        <h2 className="mk-coordinator-title">Coordinator</h2>
        <p className="mk-coordinator-lede">
          Starting a goal creates a goal node. A Claude or Codex coordinator session plans it for your approval.
        </p>
      </header>
      {(sessions.length > 0 || sessionId) && (
        <nav className="mk-coordinator-sessions" aria-label="Coordinator sessions">
          {sessions.map((session) => <button key={session.id} type="button"
            className={session.id === sessionId ? 'mk-btn mk-btn-secondary mk-btn-sm is-on' : 'mk-btn mk-btn-secondary mk-btn-sm'}
            aria-current={session.id === sessionId ? 'page' : undefined}
            onClick={() => selectSession(session.id)}>{session.description} · {session.provider}</button>)}
          {sessionId && <Button variant="ghost" glyph="+" onClick={() => selectSession('')}>New goal</Button>}
        </nav>
      )}
      {!sessionId && (
        <form className="mk-fields" onSubmit={(event) => void start(event)}>
          <div className="mk-field">
            <label>Goal <input className="mk-input" value={goal} placeholder="Describe the outcome you want"
              onChange={(event) => setGoal(event.target.value)} required /></label>
          </div>
          <div className="mk-field">
            <label>Provider <select className="mk-input" value={provider}
              onChange={(event) => setProvider(event.target.value as 'claude' | 'codex')}>
              <option value="claude">Claude</option><option value="codex">Codex</option>
            </select></label>
          </div>
          <div className="mk-coordinator-actions">
            <Button variant="primary" disabled={busy} type="submit">Start goal</Button>
          </div>
        </form>
      )}
      {snapshot && (
        <>
          <header className="mk-coordinator-goal"><strong>{snapshot.goal.description}</strong> <span className="mk-kicker">{snapshot.session.provider}</span></header>
          <div className="mk-coordinator-actions">
            <Button variant="primary" disabled={busy} onClick={() => void send('plan_goal')}>Propose plan</Button>
            <Button disabled={busy} onClick={() => void send('recover')}>Recover</Button>
          </div>
          {commandResult && <p role="status">{commandResult}</p>}
          <ProposalPanel proposals={snapshot.proposals} busy={busy} send={send} />
          <p className="mk-coordinator-meta">Session {snapshot.session.id}</p>
          <p className="mk-coordinator-meta">Graph: {snapshot.nodes.length} nodes · {snapshot.runs.length} runs</p>
          <p className="mk-coordinator-meta">Recovery: {snapshot.recovery.at(-1)?.status ?? 'not recorded'}</p>
          <EvidencePanel snapshot={snapshot} nodeId={nodeId} busy={busy} send={send} />
        </>
      )}
    </aside>
  );
}
