// GET /api/runs/{id}/diff returns `PlainTextResponse`, not JSON, so it
// cannot go through `app/api.get` (which always parses JSON). This mirrors
// `app/api.ts`'s 401 redirect only; the diff route has no 409 case.
import { pushNotice } from '../../app/store';

export async function fetchDiffText(runId: string, path: string): Promise<string> {
  const params = new URLSearchParams({ path });
  try {
    const response = await fetch(`/api/runs/${runId}/diff?${params.toString()}`);
    if (response.status === 401) {
      window.location.assign('/');
      return '';
    }
    if (!response.ok) {
      pushNotice('Failed to load the diff.');
      return '';
    }
    return await response.text();
  } catch {
    pushNotice('Failed to load the diff.');
    return '';
  }
}
