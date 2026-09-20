import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { patch, post } from '../../app/api';
import { resetStore, setSelection, setSnapshot } from '../../app/store';
import { ArchiveNodeDialog } from './ArchiveNodeDialog';
import { closeDialog, openDialog, resetDialog } from './dialogState';
import { EditNodeDialog } from './EditNodeDialog';
import { MoveNodeDialog } from './MoveNodeDialog';
import { NodeActionButtons } from './NodeActionButtons';

vi.mock('../../app/api', () => ({
  post: vi.fn().mockResolvedValue({}),
  patch: vi.fn().mockResolvedValue({}),
}));

function capabilities(overrides: Record<string, unknown> = {}) {
  return {
    session_input: { available: true, reason: null },
    cancel: { available: true, reason: null },
    force_stop: { available: true, reason: null },
    stop_scheduling: { available: true, reason: null },
    graph_edits: { available: true, reason: null },
    review_decision: { available: true, reason: null },
    git: { available: true, reason: null },
    owner: { available: false },
    ...overrides,
  };
}

function seedSnapshot(): void {
  setSnapshot({
    goal: null,
    graph: {
      nodes: [
        { id: 1, description: 'Root', status: 'pending', parent_id: null, kind: 'goal', flavor: null },
        { id: 2, description: 'Child', status: 'pending', parent_id: 1, kind: 'task', flavor: null },
      ],
      edges: [],
      root_ids: [1],
    },
    capabilities: capabilities(),
  });
}

describe('NodeActionButtons', () => {
  beforeEach(() => {
    resetStore();
    resetDialog();
    vi.mocked(post).mockClear();
    vi.mocked(patch).mockClear();
  });

  afterEach(cleanup);

  it('renders nothing without a numeric selection', () => {
    seedSnapshot();
    const { container } = render(<NodeActionButtons />);
    expect(container.firstChild).toBeNull();
  });

  it('opens the edit dialog for the selected node', () => {
    seedSnapshot();
    setSelection(2);
    render(<NodeActionButtons />);

    screen.getByText('Edit node').click();
    closeDialog();
  });
});

describe('EditNodeDialog', () => {
  beforeEach(() => {
    resetStore();
    resetDialog();
    vi.mocked(patch).mockClear();
  });

  afterEach(cleanup);

  it('patches the exact EditNodeBody shape on submit', () => {
    seedSnapshot();
    openDialog('edit', 2);
    render(<EditNodeDialog />);

    screen.getByText('Save changes').click();

    expect(patch).toHaveBeenCalledWith('/api/nodes/2', { description: 'Child', flavor: null });
  });
});

describe('MoveNodeDialog', () => {
  beforeEach(() => {
    resetStore();
    resetDialog();
    vi.mocked(post).mockClear();
  });

  afterEach(cleanup);

  it('posts the new parent id on submit', () => {
    seedSnapshot();
    openDialog('move', 2);
    render(<MoveNodeDialog />);

    screen.getByText('Move node').click();

    expect(post).toHaveBeenCalledWith('/api/nodes/2/move', { new_parent_id: null });
  });
});

describe('ArchiveNodeDialog', () => {
  beforeEach(() => {
    resetStore();
    resetDialog();
    vi.mocked(post).mockClear();
  });

  afterEach(cleanup);

  it('posts an archive request on confirm', () => {
    seedSnapshot();
    openDialog('archive', 2);
    render(<ArchiveNodeDialog />);

    screen.getByText('Archive node').click();

    expect(post).toHaveBeenCalledWith('/api/nodes/2/archive');
  });
});
