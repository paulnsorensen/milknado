import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from '../../app/api';
import { resetStore, setSelection, setSnapshot } from '../../app/store';
import { resetDetail, resetTab, setActiveTab } from '../../shared/node-detail';
import { NodeSidecar } from './NodeSidecar';
import { resetReviewSelection } from '../goal-review/selection';
import { DetailsTabSection, SessionTabSection } from './TabSections';
import { detailResponse, EMPTY_CAPABILITIES } from './testFixtures';

vi.mock('../../app/api', () => ({ get: vi.fn() }));

describe('NodeSidecar tabs', () => {
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

  it('renders a visible tabpanel for a selected run', async () => {
    vi.mocked(get).mockResolvedValue(detailResponse());
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: EMPTY_CAPABILITIES,
      active_runs: [
        { run_id: 'run-1', node_id: 7, description: 'Fixture run', status: 'running' },
      ],
    });
    setSelection('run-1');

    render(
      <>
        <NodeSidecar />
        <SessionTabSection />
      </>,
    );

    expect(await screen.findByRole('tabpanel')).toBeVisible();
  });

  it('renders the details tab body when the details tab is active', async () => {
    vi.mocked(get).mockResolvedValue(detailResponse());
    setSelection(7);
    setActiveTab('details');

    render(
      <>
        <NodeSidecar />
        <DetailsTabSection />
      </>,
    );

    expect(await screen.findByText('Parent')).toBeTruthy();
  });
});
