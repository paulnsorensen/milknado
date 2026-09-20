import { describe, expect, it } from 'vitest';
import { Milknado } from './index';

describe('design-system loader', () => {
  it('shares one React instance with window', () => {
    expect(Milknado.React).toBe(window.React);
  });

  it('exposes the MikadoGraph component', () => {
    expect(typeof Milknado.MikadoGraph).toBe('function');
  });
});
