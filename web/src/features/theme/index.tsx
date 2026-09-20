// The `header-control` slot contribution: a light/dark theme switch.
// Applying the stored or system theme happens here, at module load —
// `Shell.tsx` calls `registerFeatures()` (which eagerly imports every
// feature module) before `main.tsx` calls `createRoot(...).render(...)`,
// so the theme is set before the app's first paint.
import { registerSlot } from '../../app/slots';
import { initializeTheme } from './theme';
import { ThemeSwitch } from './ThemeSwitch';

initializeTheme();

export function register(): void {
  registerSlot('header-control', () => <ThemeSwitch />);
}
