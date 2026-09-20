import { afterEach, describe, expect, it } from 'vitest';
import { ACTION_IDS, clearActions, dispatchAction } from '../../app/actions';
import { clearSlots, getSlot } from '../../app/slots';
import { closeHelp, isHelpOpen } from '../help/helpState';
import { register } from './index';

describe('shortcuts register', () => {
  afterEach(() => {
    clearSlots();
    clearActions();
    closeHelp();
  });

  it('registers a provider slot contribution', () => {
    register();

    expect(getSlot('provider')).toHaveLength(1);
  });

  it('registers help.open so it opens the Help dialog', () => {
    register();

    expect(ACTION_IDS).toContain('help.open');
    dispatchAction('help.open');

    expect(isHelpOpen()).toBe(true);
  });
});
