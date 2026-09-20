import { afterEach, describe, expect, it, vi } from 'vitest';
import { applyTheme, currentTheme, initializeTheme, resolveTheme, systemTheme, THEME_STORAGE_KEY } from './theme';

afterEach(() => {
  vi.unstubAllGlobals();
  window.localStorage.clear();
  document.documentElement.removeAttribute('data-theme');
});

describe('systemTheme', () => {
  it('is dark when the browser cannot report a preference', () => {
    vi.stubGlobal('matchMedia', undefined);
    expect(systemTheme()).toBe('dark');
  });

  it('is light when the OS prefers light', () => {
    vi.stubGlobal('matchMedia', () => ({ matches: true }));
    expect(systemTheme()).toBe('light');
  });

  it('is dark when the OS does not prefer light', () => {
    vi.stubGlobal('matchMedia', () => ({ matches: false }));
    expect(systemTheme()).toBe('dark');
  });
});

describe('resolveTheme', () => {
  it('honours a stored dark or light choice', () => {
    expect(resolveTheme('dark')).toBe('dark');
    expect(resolveTheme('light')).toBe('light');
  });

  it('falls back to the system theme for null or an unknown value', () => {
    vi.stubGlobal('matchMedia', () => ({ matches: true }));
    expect(resolveTheme(null)).toBe('light');
    expect(resolveTheme('sepia')).toBe('light');
  });
});

describe('currentTheme', () => {
  it('reads the root data-theme attribute', () => {
    document.documentElement.setAttribute('data-theme', 'light');
    expect(currentTheme()).toBe('light');
  });

  it('defaults to dark with no attribute', () => {
    expect(currentTheme()).toBe('dark');
  });
});

describe('applyTheme', () => {
  it('sets the data-theme attribute on the given root', () => {
    const root = document.createElement('html');
    applyTheme('light', root);
    expect(root.getAttribute('data-theme')).toBe('light');
  });
});

describe('initializeTheme', () => {
  it('applies a stored choice to the document root', () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, 'light');

    const theme = initializeTheme();

    expect(theme).toBe('light');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });

  it('applies the system theme with nothing stored', () => {
    vi.stubGlobal('matchMedia', () => ({ matches: false }));

    expect(initializeTheme()).toBe('dark');
  });
});
