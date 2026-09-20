import { afterEach, describe, expect, it } from 'vitest';
import { clearSlots, getSlot } from '../../app/slots';
import { resetStore } from '../../app/store';
import { register } from './index';

describe('narrow register', () => {
  afterEach(() => {
    clearSlots();
    resetStore();
  });

  it('registers the layout slot contribution', () => {
    register();

    expect(getSlot('layout')).toHaveLength(1);
  });
});
