import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ConfirmDialog } from './ConfirmDialog';
import { requestConfirm, resetConfirm } from './confirmState';

describe('ConfirmDialog', () => {
  beforeEach(resetConfirm);
  afterEach(cleanup);

  it('renders the force-stop title, body, and accessible action names', () => {
    requestConfirm({
      prompt: 'Force stop the run?',
      body: 'The run stops now. It does not wait for the current turn. Changes that are not committed stay in the worktree.',
      dismissLabel: 'Keep the run',
      confirmLabel: 'Force stop',
      action: vi.fn(),
    });

    render(<ConfirmDialog />);

    expect(screen.getByRole('alertdialog', { name: 'Force stop the run?' })).toBeVisible();
    expect(screen.getByText('The run stops now. It does not wait for the current turn. Changes that are not committed stay in the worktree.')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Keep the run' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Force stop' })).toBeVisible();
  });
});
