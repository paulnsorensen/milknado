import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { pushNotice, resetStore } from '../../app/store';
import { NoticeToasts } from './NoticeToasts';

describe('NoticeToasts', () => {
  beforeEach(resetStore);
  afterEach(cleanup);

  it('renders nothing with no notices', () => {
    const { container } = render(<NoticeToasts />);
    expect(container.querySelector('p')).toBeNull();
  });

  it('shows a notice pushed for a rejected command', () => {
    pushNotice('Cancel is unavailable.');
    render(<NoticeToasts />);

    expect(screen.getByText('Cancel is unavailable.')).toBeTruthy();
  });
});
