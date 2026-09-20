import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { post } from '../../app/api';
import { resetStore, setSnapshot } from '../../app/store';
import { AddNodeButton } from './AddNodeButton';
import { AddNodeDialog } from './AddNodeDialog';
import { closeDialog, openDialog, resetDialog } from './dialogState';

vi.mock('../../app/api', () => ({ post: vi.fn().mockResolvedValue({}), patch: vi.fn() }));

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

describe('AddNodeButton', () => {
  beforeEach(() => {
    resetStore();
    resetDialog();
    vi.mocked(post).mockClear();
  });

  afterEach(cleanup);

  it('renders nothing without capabilities', () => {
    const { container } = render(<AddNodeButton />);
    expect(container.firstChild).toBeNull();
  });

  it('disables Add node and shows the server reason when unavailable', () => {
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: capabilities({
        graph_edits: { available: false, reason: 'Graph edits are unavailable.' },
      }),
    });
    render(<AddNodeButton />);

    expect(screen.getByText('Add node')).toBeDisabled();
    expect(screen.getByText('Graph edits are unavailable.')).toBeTruthy();
  });

  it('opens the Add node dialog', () => {
    setSnapshot({ goal: null, graph: null, capabilities: capabilities() });
    render(<AddNodeButton />);

    screen.getByText('Add node').click();

    closeDialog();
  });
});

describe('AddNodeDialog', () => {
  beforeEach(() => {
    resetStore();
    resetDialog();
    vi.mocked(post).mockClear();
  });

  afterEach(cleanup);

  it('renders nothing when the add dialog is not open', () => {
    const { container } = render(<AddNodeDialog />);
    expect(container.firstChild).toBeNull();
  });

  it('posts the exact AddNodeBody shape on submit', () => {
    setSnapshot({
      goal: null,
      graph: { nodes: [], edges: [], root_ids: [] },
      capabilities: capabilities(),
    });
    openDialog('add');
    render(<AddNodeDialog />);

    fireEvent.change(screen.getByLabelText('Description'), { target: { value: 'New node' } });
    screen.getByText('Add node').click();

    expect(post).toHaveBeenCalledWith('/api/nodes', {
      description: 'New node',
      parent_id: null,
      flavor: null,
      files: null,
      prereqs: null,
    });
  });
});
