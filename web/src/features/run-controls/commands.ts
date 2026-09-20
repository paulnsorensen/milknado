// Command posts for the run-controls feature: one POST per confirmed action.
import { post } from '../../app/api';

export function cancelRun(runId: string): Promise<unknown> {
  return post(`/api/runs/${runId}/cancel`);
}

export function forceStopRun(runId: string): Promise<unknown> {
  return post(`/api/runs/${runId}/force-stop`);
}

export function stopScheduling(): Promise<unknown> {
  return post('/api/scheduling/stop');
}
