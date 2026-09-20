import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { resetStore, setSnapshot } from '../../app/store';
import { ErrorBanner } from './ErrorBanner';

function capabilities() {
  return {
    session_input: { available: true, reason: null },
    cancel: { available: true, reason: null },
    force_stop: { available: true, reason: null },
    stop_scheduling: { available: true, reason: null },
    graph_edits: { available: true, reason: null },
    review_decision: { available: true, reason: null },
    git: { available: true, reason: null },
    owner: { available: false },
  };
}

describe('ErrorBanner', () => {
  beforeEach(resetStore);
  afterEach(cleanup);

  it('renders nothing with no listener errors', () => {
    setSnapshot({ goal: null, graph: null, capabilities: capabilities() });
    const { container } = render(<ErrorBanner />);
    expect(container.children.length).toBe(0);
  });

  it('shows each listener error', () => {
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: capabilities(),
      listener_errors: ['Live update listener failed.'],
    } as never);
    render(<ErrorBanner />);

    expect(screen.getByText('Live update listener failed.')).toBeTruthy();
  });
});
