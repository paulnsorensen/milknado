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
});
