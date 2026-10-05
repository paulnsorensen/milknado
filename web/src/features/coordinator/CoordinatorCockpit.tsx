import type { FormEvent, ReactElement } from 'react';
import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { get, post } from '../../app/api';
import { getState, pushNotice, selectedNodeId, setCoordinatorGraph, subscribe } from '../../app/store';
import type { WireGraphSnapshot, WireNodeKind, WireNodeStatus } from '../../app/wire';
import { setActiveTab } from '../../shared/node-detail';

interface Node {
  id: number;
  description: string;
  status: WireNodeStatus;
  parent_id: number | null;
  kind?: WireNodeKind;
  flavor?: string | null;
  artifact_path?: string | null;
}
interface Record {
  seq: number;
  kind: string;
  text: string;
  entity_kind: string;
  entity_id: string;
  status: string;
  tool_name?: string;
  duration_ms?: number | null;
}
interface Review {
  review_id: number;
  goal_id: number;
  decision: string;
  evidence: string;
  proposed_change: string;
}
interface Run {
  run_id: string;
  node_id: number;
  status: string;
  detail?: string | null;
  error?: string | null;
  verification_status?: 'accepted' | 'rejected' | null;
  verified_at?: string | null;
}
interface PlanProposal {
  id: string;
  status: string;
  manifest: {
    goal_summary: string;
    changes: Array<{ id: string; path: string; description: string; depends_on?: string[] }>;
    new_relationships?: Array<{
      source_change_id: string;
      dependant_change_id: string;
      reason: string;
    }>;
  };
}
interface Snapshot {
  session: { id: string; goal_id: number; provider: string };
  goal: Node;
  nodes: Node[];
  edges: Array<{ parent_id: number; child_id: number }>;
  runs: Run[];
  reviews: Review[];
  proposals?: PlanProposal[];
  recovery: Record[];
  provider_turns: Array<{ provider_session_id: string; status: string }>;
  provider_bindings: Array<{ provider_session_id: string; scope_kind: string; scope_id: string }>;
  capability_floor: { [key: string]: string };
  native_actions: string[];
  unsupported_actions: string[];
  events: Record[];
  cursor: number;
}
interface SessionSummary {
  id: string;
  goal_id: number;
  provider: string;
  created_at: string;
  description: string;
}

interface Receipt {
  status: string;
  result: { id?: string } | string | null;
}
const STORAGE_KEY = 'milknado.coordinator.session';

function graphFrom(snapshot: Snapshot): WireGraphSnapshot {
  return {
    nodes: snapshot.nodes.map((node) => ({
      ...node, kind: node.kind ?? 'task', flavor: node.flavor ?? null,
    })),
    edges: snapshot.edges,
    root_ids: [snapshot.goal.id],
  };
}

export function CoordinatorCockpit(): ReactElement {
  const store = useSyncExternalStore(subscribe, getState);
  const nodeId = selectedNodeId(store);
  const [sessionId, setSessionId] = useState(() => localStorage.getItem(STORAGE_KEY) ?? '');
  const session = useRef({ id: sessionId, version: 0 });
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [available, setAvailable] = useState(false);
  const [goal, setGoal] = useState('');
  const [provider, setProvider] = useState<'claude' | 'codex'>('claude');
  const [busy, setBusy] = useState(false);
  const [commandResult, setCommandResult] = useState('');

  useEffect(() => {
    let active = true;
    void fetch('/api/coordinators')
      .then(async (response) => {
        if (response.status === 409) return null;
        if (!response.ok) throw new Error('Coordinator discovery failed.');
        return await response.json() as SessionSummary[];
      })
      .then((items) => { if (active && items) { setSessions(items); setAvailable(true); } })
      .catch(() => { if (active) pushNotice('Failed to discover coordinator sessions.'); });
    return () => { active = false; };
  }, []);

  function selectSession(id: string): void {
    if (session.current.id === id) return;
    session.current = { id, version: session.current.version + 1 };
    if (id) localStorage.setItem(STORAGE_KEY, id);
    else localStorage.removeItem(STORAGE_KEY);
    setSessionId(id);
    setSnapshot(null);
    setCoordinatorGraph(null);
    setCommandResult('');
    setBusy(false);
  }

  useEffect(() => {
    if (!sessionId || !available) return;
    let active = true;
    const version = session.current.version;
    const current = () => active && session.current.id === sessionId && session.current.version === version;
    const refresh = () => get<Snapshot>(`/api/coordinators/${encodeURIComponent(sessionId)}/snapshot`)
      .then((value) => { if (current() && value) { setSnapshot(value); setCoordinatorGraph(graphFrom(value)); } })
      .catch(() => { if (current()) pushNotice('Failed to load coordinator status.'); });
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => { active = false; window.clearInterval(timer); };
  }, [sessionId, available]);

  async function send(kind: string, fields: object = {}): Promise<void> {
    if (!sessionId) return;
    const version = session.current.version;
    const current = () => session.current.id === sessionId && session.current.version === version;
    setBusy(true);
    try {
      const result = await post<Receipt>(`/api/coordinators/${encodeURIComponent(sessionId)}/commands`, {
        kind, command_id: crypto.randomUUID(), ...fields,
      });
      if (!current()) return;
      if (result && result.status !== 'accepted') pushNotice(String(result.result ?? result.status));
      if (result) setCommandResult(`${kind}: ${result.status} · ${JSON.stringify(result.result)}`);
      const latest = await get<Snapshot>(`/api/coordinators/${encodeURIComponent(sessionId)}/snapshot`);
      if (current() && latest) { setSnapshot(latest); setCoordinatorGraph(graphFrom(latest)); }
    } catch {
      if (current()) pushNotice('Coordinator command failed.');
    } finally {
      if (current()) setBusy(false);
    }
  }

  async function start(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!goal.trim()) return;
    let active = session.current;
    setBusy(true);
    try {
      const receipt = await post<Receipt>('/api/coordinators/commands', {
        kind: 'start_goal', command_id: crypto.randomUUID(), description: goal.trim(), provider,
      });
      if (session.current !== active) return;
      if (receipt?.status !== 'accepted' || !receipt.result || typeof receipt.result !== 'object' || !receipt.result.id) {
        pushNotice(String(receipt?.result ?? 'Goal intake failed.'));
        return;
      }
      selectSession(receipt.result.id);
      active = session.current;
      const current = await get<SessionSummary[]>('/api/coordinators');
      if (session.current === active && current) setSessions(current);
    } catch {
      if (session.current === active) pushNotice('Goal intake failed.');
    } finally {
      if (session.current === active) setBusy(false);
    }
  }

  const node = snapshot?.nodes.find((item) => item.id === nodeId);
  const runs = snapshot?.runs.filter((run) => run.node_id === nodeId) ?? [];
  const runIds = new Set(runs.map((run) => run.run_id));
  const evidence = snapshot?.events.filter((record) =>
    (record.entity_kind === 'node' && record.entity_id === String(nodeId)) ||
    (record.entity_kind === 'run' && runIds.has(record.entity_id))) ?? [];
  const reviews = snapshot?.reviews.filter((review) => review.goal_id === snapshot.goal.id) ?? [];
  const recovery = snapshot?.recovery ?? [];

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
          <section aria-label="Plan proposals"><h3>Plan proposals</h3>
            {snapshot.proposals?.length ? snapshot.proposals.map((proposal) => <div key={proposal.id}>
              <h4>{proposal.id}: {proposal.status}</h4>
              <p>{proposal.manifest.goal_summary}</p>
              <ul>{proposal.manifest.changes.map((change) => <li key={change.id}>
                {change.path}: {change.description}
                <p>Depends on: {change.depends_on?.join(', ') || 'none'}</p>
              </li>)}</ul>
              <p>New relationships:</p>
              <ul>{proposal.manifest.new_relationships?.length
                ? proposal.manifest.new_relationships.map((relationship) => <li key={`${relationship.source_change_id}-${relationship.dependant_change_id}`}>
                  {relationship.source_change_id} → {relationship.dependant_change_id}: {relationship.reason}
                </li>)
                : <li>none</li>}</ul>
              {proposal.status === 'pending' && <>
                <button disabled={busy} onClick={() => void send('decide_plan_proposal', { proposal_id: proposal.id, decision: 'accepted' })}>Approve {proposal.id}</button>
                <button disabled={busy} onClick={() => void send('decide_plan_proposal', { proposal_id: proposal.id, decision: 'rejected' })}>Reject {proposal.id}</button>
              </>}
              {proposal.status === 'applying' && <p>Apply incomplete. Manual recovery is required.</p>}
              {proposal.status === 'stale' && <p>Graph changed. Request a new proposal.</p>}
            </div>) : <p>No plan proposal.</p>}
          </section>
          {node && (
            <div className="mk-coordinator-evidence">
              <h3>Node {node.id}: {node.status}</h3>
              <section aria-label="Coordinator history"><h4>Coordinator history</h4>
                <button onClick={() => setActiveTab("session")}>Open transcript</button>
                {evidence.length ? evidence.map((item) => <p key={item.seq}>
                  {item.kind} {item.tool_name && `· ${item.tool_name}`} {item.status && `· ${item.status}`} {item.text}
                </p>) : <p>No coordinator output for this node.</p>}
              </section>
              <section aria-label="Approvals"><h4>Approvals</h4>
                {reviews.length ? reviews.map((review) => <div key={review.review_id}>
                  <p>{review.decision}: {review.evidence}</p><p>{review.proposed_change}</p>
                  {review.decision === 'pending' && <>
                    <button disabled={busy} onClick={() => void send('decide_goal_review', { review_id: review.review_id, decision: 'accepted' })}>Approve</button>
                    <button disabled={busy} onClick={() => void send('decide_goal_review', { review_id: review.review_id, decision: 'rejected' })}>Reject</button>
                  </>}
                </div>) : <p>No goal review.</p>}
              </section>
              <section aria-label="Run result"><h4>Run result</h4>
                {runs.length ? runs.map((run) => <p key={run.run_id}>{run.status}: {run.detail ?? run.error ?? 'No run detail.'}</p>) : <p>No run recorded.</p>}
              </section>
              <section aria-label="Completion verification"><h4>Completion verification</h4>
                {runs.length ? runs.map((run) => <p key={run.run_id}>
                  {run.verification_status ?? 'No verifier receipt'}
                  {run.verified_at && ` · ${run.verified_at}`}
                </p>) : <p>No verifier receipt.</p>}
              </section>
              <section aria-label="Artifacts"><h4>Artifacts</h4><p>{node.artifact_path ?? 'No artifact recorded.'}</p></section>
              <section aria-label="Recovery"><h4>Recovery</h4>
                {recovery.length ? recovery.map((item) => <p key={item.seq}>
                  {item.entity_kind} {item.entity_id}: {item.status}: {item.text}
                </p>) : <p>No coordinator recovery receipt.</p>}
              </section>
              <section aria-label="Provider capabilities"><h4>Provider capabilities</h4>
                {Object.entries(snapshot.capability_floor).map(([name, state]) => <p key={name}>{name}: {state}</p>)}
                <p>Native: {snapshot.native_actions.join(', ') || 'none'}</p>
                <p>Unsupported: {snapshot.unsupported_actions.join(', ') || 'none'}</p>
                {snapshot.provider_turns.map((turn) => <p key={turn.provider_session_id}>{turn.provider_session_id}: {turn.status}</p>)}
              </section>
              <button onClick={() => setActiveTab("changes")}>Open changes and diff</button>
            </div>
          )}
        </>
      )}
    </aside>
  );
}
