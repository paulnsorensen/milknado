import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { THEME_STORAGE_KEY } from './theme';
import { ThemeSwitch } from './ThemeSwitch';

afterEach(() => {
  cleanup();
  window.localStorage.clear();
  document.documentElement.removeAttribute('data-theme');
});

describe('ThemeSwitch', () => {
  it('applies the light palette and persists the choice', () => {
    render(<ThemeSwitch />);

    screen.getByRole('button', { name: 'Light' }).click();

    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('light');
  });

  it('applies the dark palette and persists the choice', () => {
    render(<ThemeSwitch />);

    screen.getByRole('button', { name: 'Dark' }).click();

    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark');
  });
});
