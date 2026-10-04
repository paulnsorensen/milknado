import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from '../../app/api';
import { getState, resetStore, setSelection, setSnapshot } from '../../app/store';
import type { GraphNodeData } from '../../app/wire';
import { resetDetail, resetTab } from '../../shared/node-detail';
import { NodeSidecar } from './NodeSidecar';
import { SidecarAncestorPath } from './SidecarAncestorPath';
import { summarizePathTitle } from './pathTitle';
import { resetReviewSelection } from '../goal-review/selection';
import { detailResponse, EMPTY_CAPABILITIES } from './testFixtures';

vi.mock('../../app/api', () => ({ get: vi.fn() }));

describe('NodeSidecar ancestor path', () => {
  beforeEach(() => {
    resetStore();
    resetDetail();
    resetTab();
    resetReviewSelection();
    vi.mocked(get).mockReset();
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('summarizes path titles by sentence and length', () => {
    expect(summarizePathTitle('First sentence. Second sentence.')).toBe(
      'First sentence.',
    );
    expect(summarizePathTitle('Use e.g. milknado. Next task.')).toBe(
      'Use e.g. milknado.',
    );
    expect(
      summarizePathTitle('Use e.g. Milknado to continue. Next task.'),
    ).toBe('Use e.g. Milknado to continue.');
    expect(summarizePathTitle(`${'x'.repeat(100)}. second sentence`)).toBe(
      `${'x'.repeat(79)}…`,
    );
  });

  it('renders the summarized path title with its full accessible title', async () => {
    const fullTitle =
      'First ancestor sentence. Additional ancestor context stays in the full title.';
    vi.mocked(get).mockResolvedValue(detailResponse());
    setSnapshot({
      goal: null,
      graph: {
        nodes: [
          {
            id: 1,
            description: fullTitle,
            status: 'running',
            parent_id: null,
            kind: 'goal',
            flavor: null,
          },
          {
            id: 7,
            description: 'Bake the roadmap',
            status: 'running',
            parent_id: 1,
            kind: 'task',
            flavor: null,
          },
        ],
        edges: [{ parent_id: 1, child_id: 7 }],
        root_ids: [1],
      },
      capabilities: EMPTY_CAPABILITIES,
    });
    setSelection(7);

    render(<NodeSidecar />);

    const pathItem = await screen.findByRole('button', {
      name: summarizePathTitle(fullTitle),
    });
    expect(pathItem).toHaveTextContent('First ancestor sentence.');
    expect(pathItem).toHaveAttribute('title', fullTitle);
  });
});

describe('SidecarAncestorPath', () => {
  afterEach(() => {
    cleanup();
  });

  const nodes: GraphNodeData[] = [
    { id: 1, title: 'Ancestor one', kind: 'goal', state: 'running', parent: null },
    { id: 2, title: 'Ancestor two', kind: 'subgoal', state: 'running', parent: 1 },
    { id: 3, title: 'Ancestor three', kind: 'subgoal', state: 'running', parent: 2 },
    { id: 4, title: 'Ancestor four', kind: 'subgoal', state: 'running', parent: 3 },
    { id: 5, title: 'Ancestor five', kind: 'subgoal', state: 'running', parent: 4 },
    { id: 6, title: 'Ancestor six', kind: 'task', state: 'running', parent: 5 },
  ];

  it('elides the middle of a long path behind a single gap', () => {
    render(<SidecarAncestorPath nodes={nodes} nodeId={6} />);

    expect(screen.getAllByText('…', { selector: '.mk-path-gap' })).toHaveLength(1);
    expect(
      screen.getByRole('button', { name: 'Ancestor one' }),
    ).toBeTruthy();
    expect(
      screen.getByRole('button', { name: 'Ancestor four' }),
    ).toBeTruthy();
    expect(
      screen.getByRole('button', { name: 'Ancestor five' }),
    ).toBeTruthy();
    expect(
      screen.queryByRole('button', { name: 'Ancestor two' }),
    ).toBeNull();
    expect(
      screen.queryByRole('button', { name: 'Ancestor three' }),
    ).toBeNull();
  });

  it('marks the last item as current and selects a node on click', () => {
    render(<SidecarAncestorPath nodes={nodes} nodeId={6} />);

    const current = screen.getByRole('button', { name: 'Ancestor six' });
    expect(current).toHaveAttribute('aria-current', 'page');
    expect(current).toHaveClass('is-current');

    resetStore();
    fireEvent.click(screen.getByRole('button', { name: 'Ancestor one' }));
    expect(getState().selection).toBe(1);
  });

  it('renders no items when the parent chain cycles', () => {
    const cyclic: GraphNodeData[] = [
      { id: 1, title: 'Cyclic one', kind: 'task', state: 'running', parent: 2 },
      { id: 2, title: 'Cyclic two', kind: 'task', state: 'running', parent: 1 },
    ];

    render(<SidecarAncestorPath nodes={cyclic} nodeId={1} />);

    expect(screen.queryAllByRole('button')).toHaveLength(0);
  });
});
