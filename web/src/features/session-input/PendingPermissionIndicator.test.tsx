import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { resetStore, setSelection, setSnapshot } from '../../app/store';
import { PendingPermissionIndicator } from './PendingPermissionIndicator';

function snapshotWithPermissions(
  permissionIds: string[],
  ownerOptions: { available?: boolean; node_id?: number } = {},
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
        available: ownerOptions.available ?? true,
        permission_ids: permissionIds,
        node_id: ownerOptions.node_id,
      },
    },
  };
}

describe('PendingPermissionIndicator', () => {
  beforeEach(resetStore);
  afterEach(cleanup);

  it('shows pending permission requests when no node is selected', () => {
    setSnapshot(snapshotWithPermissions(['perm-1', 'perm-2']));

    render(<PendingPermissionIndicator />);

    expect(screen.getByText('2 permissions requested')).toBeVisible();
  });

  it('shows the singular label for one pending permission request', () => {
    setSnapshot(snapshotWithPermissions(['perm-1']));

    render(<PendingPermissionIndicator />);

    expect(screen.getByText('Permission requested')).toBeVisible();
  });

  it('hides pending permission requests when the selected node owns the request', () => {
    setSelection(7);
    setSnapshot(snapshotWithPermissions(['perm-1'], { node_id: 7 }));

    render(<PendingPermissionIndicator />);

    expect(screen.queryByRole('status', { name: 'Pending permission requests' })).toBeNull();
  });

  it('shows pending permission requests when a different node is selected', () => {
    setSelection(3);
    setSnapshot(snapshotWithPermissions(['perm-1'], { node_id: 7 }));

    render(<PendingPermissionIndicator />);

    expect(screen.getByRole('status', { name: 'Pending permission requests' })).toHaveTextContent(
      'Permission requested',
    );
  });

  it('shows pending permission requests when a run is selected', () => {
    setSelection('run-1');
    setSnapshot(snapshotWithPermissions(['perm-1']));

    render(<PendingPermissionIndicator />);

    expect(screen.getByRole('status', { name: 'Pending permission requests' })).toHaveTextContent(
      'Permission requested',
    );
  });

  it('hides pending permission requests when no owner node is available and a node is selected', () => {
    setSelection(1);
    setSnapshot(snapshotWithPermissions(['perm-1']));

    render(<PendingPermissionIndicator />);

    expect(screen.queryByRole('status', { name: 'Pending permission requests' })).toBeNull();
  });

  it('hides pending permission requests when the owner is unavailable', () => {
    setSnapshot(snapshotWithPermissions(['perm-1'], { available: false }));

    render(<PendingPermissionIndicator />);

    expect(screen.queryByRole('status', { name: 'Pending permission requests' })).toBeNull();
  });

  it('hides when there are no pending permission requests', () => {
    setSnapshot(snapshotWithPermissions([]));

    render(<PendingPermissionIndicator />);

    expect(screen.queryByLabelText('Pending permission requests')).toBeNull();
  });
});
