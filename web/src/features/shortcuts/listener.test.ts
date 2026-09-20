import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ACTION_IDS, clearActions, registerAction } from '../../app/actions';
import { getState, resetStore, setSnapshot } from '../../app/store';
import type { WireCapabilities } from '../../app/wire';
import { KEY_BINDINGS } from './keyMap';
import { handleShortcutKey } from './listener';

function capabilities(overrides: Partial<WireCapabilities> = {}): WireCapabilities {
  const unavailable = { available: false, reason: null };
  return {
    session_input: unavailable,
    cancel: unavailable,
    force_stop: unavailable,
    stop_scheduling: unavailable,
    graph_edits: unavailable,
    review_decision: unavailable,
    git: unavailable,
    owner: { available: false },
    ...overrides,
  };
}

function press(key: string, target: EventTarget = document.body): KeyboardEvent {
  const event = new KeyboardEvent('keydown', { key, cancelable: true });
  Object.defineProperty(event, 'target', { value: target });
  return event;
}

describe('handleShortcutKey', () => {
  beforeEach(() => {
    resetStore();
    clearActions();
    for (const id of ACTION_IDS) {
      registerAction(id, vi.fn());
    }
  });

  afterEach(clearActions);

  it('does nothing while a text input has focus', () => {
    const input = document.createElement('input');
    const spy = vi.fn();
    registerAction('help.open', spy);

    const event = press('?', input);
    handleShortcutKey(event);

    expect(spy).not.toHaveBeenCalled();
    expect(event.defaultPrevented).toBe(false);
  });

  it('does nothing while a textarea has focus', () => {
    const textarea = document.createElement('textarea');
    const spy = vi.fn();
    registerAction('help.open', spy);

    handleShortcutKey(press('?', textarea));

    expect(spy).not.toHaveBeenCalled();
  });

  it('dispatches help.open on "?"', () => {
    const spy = vi.fn();
    registerAction('help.open', spy);

    const event = press('?');
    handleShortcutKey(event);

    expect(spy).toHaveBeenCalledTimes(1);
    expect(event.defaultPrevented).toBe(true);
  });

  it('moves the graph selection on an arrow key', () => {
    setSnapshot({
      goal: null,
      graph: {
        nodes: [
          { id: 1, description: 'root', status: 'pending', parent_id: null, kind: 'goal', flavor: null },
          { id: 2, description: 'a', status: 'pending', parent_id: null, kind: 'goal', flavor: null },
        ],
        edges: [],
        root_ids: [1, 2],
      },
      capabilities: capabilities(),
    });

    handleShortcutKey(press('ArrowDown'));

    expect(getState().selection).toBe(1);
  });

  it('gates a steering key on its capability', () => {
    const spy = vi.fn();
    registerAction('run.cancel', spy);
    setSnapshot({ goal: null, graph: null, capabilities: capabilities({ cancel: { available: false, reason: null } }) });

    handleShortcutKey(press('x'));
    expect(spy).not.toHaveBeenCalled();

    setSnapshot({
      goal: null,
      graph: null,
      capabilities: capabilities({
        cancel: { available: true, reason: null },
        owner: { available: true, run_id: 'run-1' },
      }),
    });
    handleShortcutKey(press('x'));
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('gates a session key on session_input availability', () => {
    const spy = vi.fn();
    registerAction('session.focus-input', spy);
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: capabilities({ session_input: { available: false, reason: null } }),
    });

    handleShortcutKey(press('i'));
    expect(spy).not.toHaveBeenCalled();
  });

  it('dispatches an ungated Runs action', () => {
    const spy = vi.fn();
    registerAction('events.open', spy);

    handleShortcutKey(press('e'));

    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('covers every non-graph action id exactly once', () => {
    const graphColumnActionless = KEY_BINDINGS.filter((b) => b.column === 'Graph').length;
    const nonGraphBindings = KEY_BINDINGS.length - graphColumnActionless;
    const nonGraphActionIds = ACTION_IDS.length;
    expect(nonGraphBindings).toBe(nonGraphActionIds);
  });
});
