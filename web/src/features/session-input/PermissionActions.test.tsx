import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { post } from '../../app/api';
import { resetStore, setSelection, setSnapshot } from '../../app/store';
import { PermissionActions } from './PermissionActions';

vi.mock('../../app/api', () => ({ post: vi.fn().mockResolvedValue({}) }));

function snapshotWithPermissions(
  permissionIds: string[],
  permissionCommands: Array<[string, string]> = permissionIds.map(
    (id): [string, string] => [id, 'git status'],
  ),
) {
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
      host_owner: { available: true, reason: null },
      owner: {
        available: true,
        run_id: 'run-1',
        node_id: 1,
        permission_ids: permissionIds,
        permission_commands: permissionCommands,
      },
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
    setSelection(1);
    setSnapshot(snapshotWithPermissions(['perm-1', 'perm-2']));

    render(<PermissionActions />);
    screen.getByText('Approve').click();

    expect(post).toHaveBeenCalledWith(
      '/api/runs/run-1/session-input',
      expect.objectContaining({ action: 'approve', request_id: 'perm-1' }),
    );
  });


  it('hides controls when selection is not the owner node', () => {
    setSelection(2);
    setSnapshot(snapshotWithPermissions(['perm-1']));

    const { container } = render(<PermissionActions />);

    expect(container.firstChild).toBeNull();
  });

  it('hides controls when the owner cannot provide a command line', () => {
    setSelection(1);
    setSnapshot(snapshotWithPermissions(['perm-1'], []));

    const { container } = render(<PermissionActions />);

    expect(container.firstChild).toBeNull();
  });
  it('shows the pending request id and command line from owner capabilities', () => {
    setSelection(1);
    setSnapshot(snapshotWithPermissions(['perm-1']));

    render(<PermissionActions />);

    expect(screen.getByText('perm-1')).toBeVisible();
    expect(screen.getByLabelText('Request ID perm-1')).toBeVisible();
    expect(screen.getByText('git status')).toBeVisible();
  });

  it('denies the oldest pending permission request', () => {
    setSelection(1);
    setSnapshot(snapshotWithPermissions(['perm-1']));

    render(<PermissionActions />);
    screen.getByText('Deny').click();

    expect(post).toHaveBeenCalledWith(
      '/api/runs/run-1/session-input',
      expect.objectContaining({ action: 'deny', request_id: 'perm-1' }),
    );
  });
});
