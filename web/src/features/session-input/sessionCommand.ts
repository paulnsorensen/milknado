// Posts a session-input command for the owned run, minting a fresh
// `command_id` for every request per `SessionInput` in
// `src/milknado/domains/common/session.py`.
import { post } from '../../app/api';
import { getState } from '../../app/store';

export type SessionCommandAction = 'steer' | 'follow_up' | 'interrupt' | 'approve' | 'deny';

export interface SessionCommandOptions {
  text?: string;
  requestId?: string;
}

export async function sendSessionCommand(
  action: SessionCommandAction,
  { text = '', requestId = '' }: SessionCommandOptions = {},
): Promise<void> {
  const owner = getState().capabilities?.owner;
  if (!owner?.run_id) {
    return;
  }
  await post(`/api/runs/${owner.run_id}/session-input`, {
    action,
    text,
    request_id: requestId,
    command_id: crypto.randomUUID(),
    owner_incarnation: String(owner.owner_incarnation ?? ''),
    invocation_id: owner.invocation_id ?? '',
  });
}
