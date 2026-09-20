// Wire types for GET /api/runs/{id}/changes, bound to `ChangedFile` in
// `src/milknado/adapters/_git_changes.py`.
export interface WireChangedFile {
  path: string;
  status: string;
  added: number;
  removed: number;
  old_path: string | null;
}
