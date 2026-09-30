import { afterEach, describe, expect, it } from 'vitest';
import { clearSlots, getSlot } from '../../app/slots';
import { resetStore } from '../../app/store';
import { register } from './index';

describe('graph-view register', () => {
  afterEach(() => {
    clearSlots();
    resetStore();
  });

  it('registers only the graph toolbar contribution', () => {
    register();

    expect(getSlot('toolbar')).toHaveLength(1);
    expect(getSlot('canvas-overlay')).toHaveLength(0);
  });
});
