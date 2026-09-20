import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { resetStore, setSnapshot } from '../../app/store';
import { mergeSnapshot, type RawStreamSnapshot } from '../live-state/runtimeSnapshot';
import { ConsoleDockSection } from './ConsoleDockSection';

describe('ConsoleDockSection', () => {
  beforeEach(resetStore);
  afterEach(cleanup);

  it('renders a console line for each event line', () => {
    const raw: RawStreamSnapshot = {
      goal: 'Ship it',
      graph: null,
      active_runs: [],
      event_lines: ['a fresh console line'],
    };
    setSnapshot(mergeSnapshot(raw, null));

    render(<ConsoleDockSection />);

    expect(screen.getByText('a fresh console line')).toBeTruthy();
  });

  it('hides the guidance input', () => {
    render(<ConsoleDockSection />);

    expect(screen.queryByPlaceholderText(/./)).toBeNull();
  });
});
