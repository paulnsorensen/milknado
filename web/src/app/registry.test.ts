import { afterEach, describe, expect, it } from 'vitest';
import { registerFeatures, registerFeaturesFrom, resetFeatureRegistration } from './registry';
import { clearSlots, getSlot } from './slots';

describe('registerFeaturesFrom', () => {
  it('calls each module in sorted path order', () => {
    const order: string[] = [];
    const paths = registerFeaturesFrom({
      '../features/zeta/index.tsx': { register: () => order.push('zeta') },
      '../features/alpha/index.ts': { register: () => order.push('alpha') },
    });

    expect(paths).toEqual(['../features/alpha/index.ts', '../features/zeta/index.tsx']);
    expect(order).toEqual(['alpha', 'zeta']);
  });

  it('skips a module with no register export', () => {
    expect(() => registerFeaturesFrom({ '../features/plain/index.ts': {} })).not.toThrow();
  });
});

describe('registerFeatures', () => {
  afterEach(() => {
    clearSlots();
    resetFeatureRegistration();
  });

  it('registers each contribution once even when called twice', () => {
    registerFeatures();
    expect(getSlot('layout')).toHaveLength(1);

    registerFeatures();

    expect(getSlot('layout')).toHaveLength(1);
  });
});
