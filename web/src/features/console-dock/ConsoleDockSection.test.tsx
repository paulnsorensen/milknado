import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { resetStore, setSnapshot } from '../../app/store';
import { resetConnectionStatus, setConnectionStatus } from '../live-state/connection';
import { mergeSnapshot, type RawStreamSnapshot } from '../live-state/runtimeSnapshot';
import { ConsoleDockSection } from './ConsoleDockSection';

describe('ConsoleDockSection', () => {
  beforeEach(() => {
    resetStore();
    resetConnectionStatus();
  });
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

  it('marks the coordinator live while the stream connection is connected', () => {
    render(<ConsoleDockSection />);

    expect(document.querySelector('.mk-live-dot.is-live')).not.toBeNull();
  });

  it('marks the coordinator not live while the stream connection is reconnecting', () => {
    setConnectionStatus('reconnecting');

    render(<ConsoleDockSection />);

    expect(document.querySelector('.mk-live-dot.is-live')).toBeNull();
  });

  it('opens and hides the events console on the Events toggle', () => {
    render(<ConsoleDockSection />);

    expect(screen.queryByText('Events appear when a run starts.')).toBeNull();

    fireEvent.click(screen.getByText('Events'));
    expect(screen.getByText('Events appear when a run starts.')).toBeTruthy();

    fireEvent.click(screen.getByText('Hide events'));
    expect(screen.queryByText('Events appear when a run starts.')).toBeNull();
  });
});
