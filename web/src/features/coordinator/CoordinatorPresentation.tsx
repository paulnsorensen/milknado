import type { ReactElement } from 'react';
import { setActiveTab } from '../../shared/node-detail';
import type { PlanProposal, Snapshot } from './types';

type SendCommand = (kind: string, fields?: object) => Promise<void>;

export function ProposalPanel({ proposals, busy, send }: {
  proposals: PlanProposal[] | undefined;
  busy: boolean;
  send: SendCommand;
}): ReactElement {
  return <section aria-label="Plan proposals"><h3>Plan proposals</h3>
    {proposals?.length ? proposals.map((proposal) => <div key={proposal.id}>
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
  </section>;
}

export function EvidencePanel({ snapshot, nodeId, busy, send }: {
  snapshot: Snapshot;
  nodeId: number | null;
  busy: boolean;
  send: SendCommand;
}): ReactElement | null {
  const node = snapshot.nodes.find((item) => item.id === nodeId);
  if (!node) return null;
  const runs = snapshot.runs.filter((run) => run.node_id === nodeId);
  const runIds = new Set(runs.map((run) => run.run_id));
  const evidence = snapshot.events.filter((record) =>
    (record.entity_kind === 'node' && record.entity_id === String(nodeId)) ||
    (record.entity_kind === 'run' && runIds.has(record.entity_id)));
  const reviews = snapshot.reviews.filter((review) => review.goal_id === snapshot.goal.id);

  return <div className="mk-coordinator-evidence">
    <h3>Node {node.id}: {node.status}</h3>
    <section aria-label="Coordinator history"><h4>Coordinator history</h4>
      <button onClick={() => setActiveTab('session')}>Open transcript</button>
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
      {snapshot.recovery.length ? snapshot.recovery.map((item) => <p key={item.seq}>
        {item.entity_kind} {item.entity_id}: {item.status}: {item.text}
      </p>) : <p>No coordinator recovery receipt.</p>}
    </section>
    <section aria-label="Provider capabilities"><h4>Provider capabilities</h4>
      {Object.entries(snapshot.capability_floor).map(([name, state]) => <p key={name}>{name}: {state}</p>)}
      <p>Native: {snapshot.native_actions.join(', ') || 'none'}</p>
      <p>Unsupported: {snapshot.unsupported_actions.join(', ') || 'none'}</p>
      {snapshot.provider_turns.map((turn) => <p key={turn.provider_session_id}>{turn.provider_session_id}: {turn.status}</p>)}
    </section>
    <button onClick={() => setActiveTab('changes')}>Open changes and diff</button>
  </div>;
}
