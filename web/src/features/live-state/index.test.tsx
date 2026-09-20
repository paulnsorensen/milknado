import { afterEach, describe, expect, it } from 'vitest';
import { ACTION_IDS, clearActions, dispatchAction } from '../../app/actions';
import { clearSlots, getSlot } from '../../app/slots';
import { getState, resetStore } from '../../app/store';
import { register } from './index';

describe('live-state register', () => {
  afterEach(() => {
    clearSlots();
    clearActions();
    resetStore();
  });

  it('registers a provider and a status slot contribution', () => {
    register();

    expect(getSlot('provider')).toHaveLength(1);
    expect(getSlot('status')).toHaveLength(1);
  });

  it('registers the declared action ids', () => {
    register();

    dispatchAction('events.open');

    expect(getState().activeSidecar).toBe('events');
    expect(ACTION_IDS).toContain('run.select-next');
    expect(ACTION_IDS).toContain('run.select-previous');
  });
});
