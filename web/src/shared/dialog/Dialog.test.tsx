import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Dialog } from './Dialog';

describe('Dialog', () => {
  afterEach(cleanup);

  it('moves focus onto the dialog on open', () => {
    render(<Dialog title="Title" actions={<button>Close</button>} />);

    expect(screen.getByRole('dialog')).toHaveFocus();
  });

  it('runs onClose once on Escape', () => {
    const onClose = vi.fn();
    render(<Dialog title="Title" onClose={onClose} actions={<button>Close</button>} />);

    fireEvent.keyDown(document, { key: 'Escape' });

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('removes the Escape listener on unmount', () => {
    const onClose = vi.fn();
    const { unmount } = render(<Dialog title="Title" onClose={onClose} actions={<button>Close</button>} />);

    unmount();
    fireEvent.keyDown(document, { key: 'Escape' });

    expect(onClose).not.toHaveBeenCalled();
  });

  it('restores focus to the previously focused element on close', () => {
    const trigger = document.createElement('button');
    document.body.appendChild(trigger);
    trigger.focus();

    const { unmount } = render(<Dialog title="Title" actions={<button>Close</button>} />);
    unmount();

    expect(trigger).toHaveFocus();
    trigger.remove();
  });

  it('wraps Tab from the last control to the first', () => {
    render(
      <Dialog title="Title" actions={<button>Last</button>}>
        <button>First</button>
        <button>Middle</button>
      </Dialog>,
    );

    screen.getByText('Last').focus();
    fireEvent.keyDown(screen.getByText('Last'), { key: 'Tab' });

    expect(screen.getByText('First')).toHaveFocus();
  });

  it('wraps Shift+Tab from the first control to the last', () => {
    render(
      <Dialog title="Title" actions={<button>Last</button>}>
        <button>First</button>
        <button>Middle</button>
      </Dialog>,
    );

    screen.getByText('First').focus();
    fireEvent.keyDown(screen.getByText('First'), { key: 'Tab', shiftKey: true });

    expect(screen.getByText('Last')).toHaveFocus();
  });

  it('sends Shift+Tab from the freshly opened card to the last control', () => {
    render(
      <Dialog title="Title" actions={<button>Last</button>}>
        <button>First</button>
      </Dialog>,
    );

    expect(screen.getByRole('dialog')).toHaveFocus();
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Tab', shiftKey: true });

    expect(screen.getByText('Last')).toHaveFocus();
  });

  it('sends Tab from the freshly opened card to the first control', () => {
    render(
      <Dialog title="Title" actions={<button>Last</button>}>
        <button>First</button>
      </Dialog>,
    );

    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Tab' });

    expect(screen.getByText('First')).toHaveFocus();
  });

  it('keeps focus on the card when Tab is pressed with no focusable descendants', () => {
    render(<Dialog title="Title" actions={null} />);

    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Tab' });

    expect(screen.getByRole('dialog')).toHaveFocus();
  });
});
