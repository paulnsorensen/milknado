import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { clearActions, registerAction } from '../../app/actions';
import { getActiveTab, resetTab, setActiveTab } from '../../shared/node-detail';
import { ChangesTabButton, DetailsTabButton, SessionTabButton } from './DetailTabButtons';

describe('DetailTabButtons', () => {
  afterEach(() => {
    cleanup();
    resetTab();
    clearActions();
  });

  it('sets the details tab directly on click', () => {
    render(<DetailsTabButton />);

    screen.getByText('Details').click();

    expect(getActiveTab()).toBe('details');
  });

  it('sets the session tab directly on click', () => {
    render(<SessionTabButton />);

    screen.getByText('Session').click();

    expect(getActiveTab()).toBe('session');
  });

  it('dispatches changes.open instead of setting the tab locally', () => {
    let opened = false;
    registerAction('changes.open', () => {
      opened = true;
    });
    render(<ChangesTabButton />);

    screen.getByText('Changes').click();

    expect(opened).toBe(true);
    expect(getActiveTab()).toBe('session');
  });

  it('exposes tab state and moves focus with arrow keys', () => {
    registerAction('changes.open', () => setActiveTab('changes'));
    render(
      <div role="tablist">
        <SessionTabButton />
        <ChangesTabButton />
        <DetailsTabButton />
      </div>,
    );

    const session = screen.getByRole('tab', { name: 'Session' });
    const changes = screen.getByRole('tab', { name: 'Changes' });
    const details = screen.getByRole('tab', { name: 'Details' });

    expect(session).toHaveAttribute('aria-selected', 'true');
    expect(session).toHaveAttribute('aria-controls', 'node-detail-panel-session');
    expect(changes).toHaveAttribute('aria-selected', 'false');
    expect(changes).toHaveAttribute('tabindex', '-1');

    fireEvent.keyDown(session, { key: 'ArrowRight' });
    expect(changes).toHaveFocus();
    expect(getActiveTab()).toBe('changes');
    expect(changes).toHaveAttribute('aria-selected', 'true');

    fireEvent.keyDown(changes, { key: 'ArrowRight' });
    expect(details).toHaveFocus();
    expect(getActiveTab()).toBe('details');

    fireEvent.keyDown(details, { key: 'ArrowLeft' });
    expect(changes).toHaveFocus();
    expect(getActiveTab()).toBe('changes');
  });
});
