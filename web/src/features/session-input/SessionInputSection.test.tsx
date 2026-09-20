import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { post } from '../../app/api';
import { resetStore, setSnapshot } from '../../app/store';
import { getDraft, resetDraft, setDraft } from './draft';
import { SessionInputSection } from './SessionInputSection';

vi.mock('../../app/api', () => ({ post: vi.fn().mockResolvedValue({}) }));

function capabilities(overrides: Record<string, unknown> = {}) {
  return {
    session_input: { available: true, reason: null },
    cancel: { available: true, reason: null },
    force_stop: { available: true, reason: null },
    stop_scheduling: { available: true, reason: null },
    graph_edits: { available: true, reason: null },
    review_decision: { available: true, reason: null },
    git: { available: true, reason: null },
    owner: { available: true, run_id: 'run-1' },
    ...overrides,
  };
}

describe('SessionInputSection', () => {
  beforeEach(() => {
    resetStore();
    resetDraft();
    vi.mocked(post).mockClear();
  });

  afterEach(cleanup);

  it('shows a notice instead of the input for an inactive run', () => {
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: capabilities({ session_input: { available: false, reason: 'The run has finished.' } }),
    });

    render(<SessionInputSection />);

    expect(screen.getByText('The run has finished.')).toBeTruthy();
    expect(screen.queryByLabelText('Session guidance')).toBeNull();
  });

  it('sends the draft as a steer command and clears it', () => {
    setSnapshot({ goal: null, graph: null, capabilities: capabilities() });

    render(<SessionInputSection />);

    const textarea = screen.getByLabelText('Session guidance') as HTMLTextAreaElement;
    textarea.focus();
    Object.defineProperty(textarea, 'value', { writable: true, value: 'Slow down' });
    textarea.dispatchEvent(new Event('input', { bubbles: true }));

    screen.getByText('Steer').click();

    expect(post).toHaveBeenCalledWith(
      '/api/runs/run-1/session-input',
      expect.objectContaining({ action: 'steer' }),
    );
  });

  it('disables steer and follow up with an empty draft', () => {
    setSnapshot({ goal: null, graph: null, capabilities: capabilities() });

    render(<SessionInputSection />);

    expect(screen.getByText('Steer')).toBeDisabled();
    expect(screen.getByText('Follow up')).toBeDisabled();
  });

  it('sends interrupt with an empty draft and leaves it enabled', () => {
    setSnapshot({ goal: null, graph: null, capabilities: capabilities() });

    render(<SessionInputSection />);

    expect(screen.getByText('Interrupt')).not.toBeDisabled();

    screen.getByText('Interrupt').click();

    expect(post).toHaveBeenCalledWith(
      '/api/runs/run-1/session-input',
      expect.objectContaining({ action: 'interrupt', text: '' }),
    );
  });

  it('sends interrupt without the typed draft and keeps the draft', async () => {
    setSnapshot({ goal: null, graph: null, capabilities: capabilities() });
    setDraft('Send this after the stop');

    render(<SessionInputSection />);
    screen.getByText('Interrupt').click();
    await Promise.resolve();

    expect(post).toHaveBeenCalledWith(
      '/api/runs/run-1/session-input',
      expect.objectContaining({ action: 'interrupt', text: '' }),
    );
    expect(getDraft()).toBe('Send this after the stop');
  });
});
