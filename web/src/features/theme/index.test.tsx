import { afterEach, describe, expect, it } from 'vitest';
import { clearSlots, getSlot } from '../../app/slots';
import { initializeTheme } from './theme';
import { register } from './index';

afterEach(() => {
  clearSlots();
  window.localStorage.clear();
  document.documentElement.removeAttribute('data-theme');
});

describe('theme register', () => {
  it('registers a header-control slot contribution', () => {
    register();

    expect(getSlot('header-control')).toHaveLength(1);
  });

  it('applies a theme to the document root on initialize', () => {
    initializeTheme();

    expect(['dark', 'light']).toContain(document.documentElement.getAttribute('data-theme'));
  });
});
