// Resolves and applies the light/dark palette. `initializeTheme` runs at
// module load (see index.tsx), before `main.tsx` renders the app, so the
// first paint already shows the right palette.
export type Theme = 'dark' | 'light';

export const THEME_STORAGE_KEY = 'milknado.theme';

function isTheme(value: string | null): value is Theme {
  return value === 'dark' || value === 'light';
}

/** The OS palette preference; dark when the browser cannot report one. */
export function systemTheme(): Theme {
  if (typeof window.matchMedia !== 'function') {
    return 'dark';
  }
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

/** The stored choice, or the system preference when nothing is stored. */
export function resolveTheme(stored: string | null): Theme {
  return isTheme(stored) ? stored : systemTheme();
}

/** The theme currently applied to `root` (defaults to dark, never null). */
export function currentTheme(root: HTMLElement = document.documentElement): Theme {
  const attr = root.getAttribute('data-theme');
  return isTheme(attr) ? attr : 'dark';
}

export function applyTheme(theme: Theme, root: HTMLElement = document.documentElement): void {
  root.setAttribute('data-theme', theme);
}

/** Reads the stored theme, or null when storage is unavailable or unset. */
function readStoredTheme(): string | null {
  try {
    return window.localStorage.getItem(THEME_STORAGE_KEY);
  } catch {
    return null;
  }
}

/** Applies the stored or system theme to the document root, and returns it. */
export function initializeTheme(): Theme {
  const theme = resolveTheme(readStoredTheme());
  applyTheme(theme);
  return theme;
}
