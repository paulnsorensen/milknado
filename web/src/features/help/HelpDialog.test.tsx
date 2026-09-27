import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { HelpDialog } from './HelpDialog';
import { isHelpOpen, openHelp, resetHelp } from './helpState';

describe('HelpDialog', () => {
  beforeEach(() => {
    resetHelp();
  });

  afterEach(cleanup);

  it('renders each shortcut column as a list of key rows', () => {
    openHelp();

    render(<HelpDialog />);

    const graphSection = screen.getByLabelText('Graph');
    const list = graphSection.querySelector('ul');
    expect(list).not.toBeNull();
    const rows = list?.querySelectorAll('li.mk-key-row') ?? [];
    expect(rows.length).toBeGreaterThan(0);
  });

  it('closes on Escape', () => {
    openHelp();
    render(<HelpDialog />);

    fireEvent.keyDown(document, { key: 'Escape' });

    expect(isHelpOpen()).toBe(false);
    expect(screen.queryByRole('dialog')).toBeNull();
  });
});
