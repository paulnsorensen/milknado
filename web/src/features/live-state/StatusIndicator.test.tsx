import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { resetConnectionStatus, setConnectionStatus } from './connection';
import { StatusIndicator } from './StatusIndicator';

describe('StatusIndicator', () => {
  beforeEach(resetConnectionStatus);
  afterEach(cleanup);

  it('renders nothing while connected', () => {
    const { container } = render(<StatusIndicator />);
    expect(container.innerHTML).toBe('');
  });

  it('shows a status message while reconnecting', () => {
    setConnectionStatus('reconnecting');

    render(<StatusIndicator />);

    expect(screen.getByRole('status')).toBeTruthy();
  });
});
