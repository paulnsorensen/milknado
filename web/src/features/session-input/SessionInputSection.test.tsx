import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { post } from '../../app/api';
import { resetStore, setSelection, setSnapshot } from '../../app/store';
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
    host_owner: { available: true, reason: null },
    owner: { available: true, run_id: 'run-1', node_id: 1, actions: ['steer', 'follow_up', 'interrupt'] },
    ...overrides,
  };
}

describe('SessionInputSection', () => {
  beforeEach(() => {
    resetStore();
    resetDraft();
    vi.mocked(post).mockClear();
    setSelection(1);
  });

  afterEach(cleanup);

  it('shows a notice instead of the input for an inactive run', () => {
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: capabilities({ session_input: { available: false, reason: 'The run has finished.' } }),
    });
    setSelection(1);

    render(<SessionInputSection />);

    expect(screen.getByText('The run has finished.')).toBeTruthy();
    expect(screen.queryByLabelText('Session guidance')).toBeNull();
  });

  it('omits session input entirely in watch mode', () => {
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: capabilities({
        host_owner: { available: false, reason: null },
        owner: { available: true, run_id: 'run-1', actions: ['steer', 'follow_up', 'interrupt'] },
      }),
    });

    render(<SessionInputSection />);
    expect(screen.getByText('Read-only')).toBeVisible();
    expect(screen.queryByLabelText('Session guidance')).toBeNull();
  });
  it('renders nothing without the selected owner run', () => {
    setSnapshot({ goal: null, graph: null, capabilities: capabilities() });
    setSelection(null);

    render(<SessionInputSection />);

    expect(screen.queryByLabelText('Session guidance')).toBeNull();
  });

  it('sends the draft as a steer command and clears it', () => {
  setSnapshot({ goal: null, graph: null, capabilities: capabilities() });
  setSelection(1);

  render(<SessionInputSection />);

  const textarea = screen.getByLabelText('Session guidance') as HTMLTextAreaElement;
  textarea.focus();
  Object.defineProperty(textarea, 'value', { writable: true, value: 'Slow down' });
  textarea.dispatchEvent(new Event('input', { bubbles: true }));

  screen.getByText('Send').click();

  expect(post).toHaveBeenCalledWith(
    '/api/runs/run-1/session-input',
    expect.objectContaining({ action: 'steer' }),
  );
});

  it('keeps message modes selectable with an empty draft and disables Send', () => {
    setSnapshot({ goal: null, graph: null, capabilities: capabilities() });
    setSelection(1);

    render(<SessionInputSection />);

    expect(screen.getByText('Steer')).toBeEnabled();
    expect(screen.getByText('Follow up')).toBeEnabled();
    expect(screen.getByText('Send')).toBeDisabled();
  });

  it('sends the selected action from Send', () => {
    setSnapshot({ goal: null, graph: null, capabilities: capabilities() });
    setSelection(1);
    setDraft('Send this guidance');

    render(<SessionInputSection />);
    screen.getByText('Send').click();

    expect(post).toHaveBeenCalledWith(
      '/api/runs/run-1/session-input',
      expect.objectContaining({ action: 'steer', text: 'Send this guidance' }),
    );
  });

  it('selects interrupt without dispatching immediately', () => {
    setSnapshot({ goal: null, graph: null, capabilities: capabilities() });

    render(<SessionInputSection />);

    fireEvent.click(screen.getByRole('button', { name: 'Interrupt' }));

    expect(post).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'Interrupt' })).toHaveAttribute('aria-pressed', 'true');
  });

  it('sends the selected interrupt after typing guidance', async () => {
    setSnapshot({ goal: null, graph: null, capabilities: capabilities() });
    setSelection(1);
    setDraft('Send this after the stop');

    render(<SessionInputSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Interrupt' }));
    fireEvent.click(screen.getByRole('button', { name: 'Send' }));

    await waitFor(() => {
      expect(post).toHaveBeenCalledTimes(1);
    });
    expect(post).toHaveBeenCalledWith(
      '/api/runs/run-1/session-input',
      expect.objectContaining({ action: 'interrupt', text: '' }),
    );
    expect(getDraft()).toBe('Send this after the stop');
  });

  it('returns to the first allowed action after sending an interrupt', async () => {
    setSnapshot({ goal: null, graph: null, capabilities: capabilities() });

    render(<SessionInputSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Interrupt' }));
    fireEvent.click(screen.getByRole('button', { name: 'Send' }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Steer' })).toHaveAttribute('aria-pressed', 'true');
    });

    setDraft('Deliver this next');
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Send' })).toBeEnabled();
    });
    fireEvent.click(screen.getByRole('button', { name: 'Send' }));

    await waitFor(() => {
      expect(post).toHaveBeenCalledTimes(2);
    });
    expect(post).toHaveBeenLastCalledWith(
      '/api/runs/run-1/session-input',
      expect.objectContaining({ action: 'steer', text: 'Deliver this next' }),
    );
  });

  it('selects the first allowed action when steer is unavailable', () => {
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: capabilities({
        owner: { available: true, run_id: 'run-1', node_id: 1, actions: ['follow_up', 'interrupt'] },
      }),
    });
    setDraft('Continue from here');

    render(<SessionInputSection />);

    expect(screen.getByRole('button', { name: 'Follow up' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'Send' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Send' }));

    expect(post).toHaveBeenCalledWith(
      '/api/runs/run-1/session-input',
      expect.objectContaining({ action: 'follow_up', text: 'Continue from here' }),
    );
  });

  it('disables interrupt when the owner lacks the interrupt action', () => {
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: capabilities({
        owner: { available: true, run_id: 'run-1', node_id: 1, actions: ['steer', 'follow_up'] },
      }),
    });
    setSelection(1);

    render(<SessionInputSection />);

    expect(screen.getByText('Interrupt')).toBeDisabled();
  });
});
