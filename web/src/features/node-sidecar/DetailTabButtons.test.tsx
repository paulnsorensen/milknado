import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { clearActions, registerAction } from '../../app/actions';
import { getActiveTab, resetTab } from '../../shared/node-detail';
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
});
