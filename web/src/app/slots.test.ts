import { afterEach, describe, expect, it } from 'vitest';
import { clearSlots, getSlot, registerSlot } from './slots';

describe('registerSlot', () => {
  afterEach(() => {
    clearSlots();
  });

  it('assigns each contribution a distinct, globally monotonic key', () => {
    registerSlot('toast', () => null);
    registerSlot('dialog', () => null);
    registerSlot('toast', () => null);

    const keys = (id: 'toast' | 'dialog'): number[] => getSlot(id).map((entry) => entry.key);

    expect(keys('toast')).toEqual([0, 2]);
    expect(keys('dialog')).toEqual([1]);
    expect(keys('toast')).toEqual([0, 2]);
  });
});
