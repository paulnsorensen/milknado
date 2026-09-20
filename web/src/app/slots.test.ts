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

    const toastKeys = getSlot('toast').map((entry) => entry.key);

    expect(toastKeys[0]).not.toBe(toastKeys[1]);
    expect(toastKeys[1]).not.toBe(1);
  });
});
