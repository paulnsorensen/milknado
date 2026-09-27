// The `header-control` slot contribution: a segmented Dark / Light control.
import type { ReactElement } from 'react';
import { useState } from 'react';
import { applyTheme, currentTheme, THEME_STORAGE_KEY, type Theme } from './theme';

const THEMES: Array<{ id: Theme; label: string }> = [
  { id: 'dark', label: 'Dark' },
  { id: 'light', label: 'Light' },
];

export function ThemeSwitch(): ReactElement {
  const [theme, setTheme] = useState<Theme>(currentTheme);

  function select(next: Theme): void {
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // Storage unavailable (private mode, quota, disabled): the palette
      // still applies for this session, just is not remembered.
    }
    applyTheme(next);
    setTheme(next);
  }

  return (
    <div className="mk mk-seg" role="group" aria-label="Theme">
      {THEMES.map((option) => (
        <button
          key={option.id}
          type="button"
          className={theme === option.id ? 'mk-seg-opt is-on' : 'mk-seg-opt'}
          aria-pressed={theme === option.id}
          onClick={() => select(option.id)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}