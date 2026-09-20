// The `header-control` slot contribution: pick the light or dark palette.
import type { ReactElement } from 'react';
import { useState } from 'react';
import { Milknado } from '../../design-system';
import { applyTheme, currentTheme, THEME_STORAGE_KEY, type Theme } from './theme';

export function ThemeSwitch(): ReactElement {
  const [theme, setTheme] = useState<Theme>(currentTheme);
  const { Button } = Milknado;

  function select(next: Theme): void {
    window.localStorage.setItem(THEME_STORAGE_KEY, next);
    applyTheme(next);
    setTheme(next);
  }

  return (
    <div role="group" aria-label="Theme">
      <Button variant={theme === 'dark' ? 'primary' : 'secondary'} onClick={() => select('dark')}>
        Dark
      </Button>
      <Button variant={theme === 'light' ? 'primary' : 'secondary'} onClick={() => select('light')}>
        Light
      </Button>
    </div>
  );
}
