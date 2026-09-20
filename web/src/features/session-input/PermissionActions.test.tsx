import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { post } from '../../app/api';
import { resetStore, setSnapshot } from '../../app/store';
import { PermissionActions } from './PermissionActions';

vi.mock('../../app/api', () => ({ post: vi.fn().mockResolvedValue({}) }));

function snapshotWithPermissions(permissionIds: string[]) {
  return {
    goal: null,
    graph: null,
    capabilities: {
      session_input: { available: true, reason: null },
      cancel: { available: true, reason: null },
      force_stop: { available: true, reason: null },
      stop_scheduling: { available: true, reason: null },
      graph_edits: { available: true, reason: null },
      review_decision: { available: true, reason: null },
      git: { available: true, reason: null },
      owner: { available: true, run_id: 'run-1', permission_ids: permissionIds },
    },
  };
}

describe('PermissionActions', () => {
  beforeEach(() => {
    resetStore();
    vi.mocked(post).mockClear();
  });

  afterEach(cleanup);

  it('renders nothing with no pending permission request', () => {
    setSnapshot(snapshotWithPermissions([]));
    const { container } = render(<PermissionActions />);
    expect(container.children.length).toBe(0);
  });

  it('approves the oldest pending permission request', () => {
    setSnapshot(snapshotWithPermissions(['perm-1', 'perm-2']));

    render(<PermissionActions />);
    screen.getByText('Approve').click();

    expect(post).toHaveBeenCalledWith(
      '/api/runs/run-1/session-input',
      expect.objectContaining({ action: 'approve', request_id: 'perm-1' }),
    );
  });

  it('denies the oldest pending permission request', () => {
    setSnapshot(snapshotWithPermissions(['perm-1']));

    render(<PermissionActions />);
    screen.getByText('Deny').click();

    expect(post).toHaveBeenCalledWith(
      '/api/runs/run-1/session-input',
      expect.objectContaining({ action: 'deny', request_id: 'perm-1' }),
    );
  });
});
