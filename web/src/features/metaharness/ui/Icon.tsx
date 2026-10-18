// Inline stroke icons, so the auth-gated dashboard loads no icon font.
import type { ReactElement } from 'react';

const PATHS = {
  expand: 'M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7',
  collapse: 'M4 14h6v6M20 10h-6V4M14 10l7-7M3 21l7-7',
  caret: 'M9 6l6 6-6 6',
  diff: 'M6 3v12M18 9v12M6 21a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM18 9a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM18 9c0 6-12 3-12 9',
  focus: 'M12 2v4M12 18v4M2 12h4M18 12h4M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z',
  info: 'M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM12 16v-5M12 8h.01',
  terminal: 'M4 4h16v16H4zM8 9l3 3-3 3M13 15h3',
  back: 'M19 12H5M11 6l-6 6 6 6',
  send: 'M12 19V5M5 12l7-7 7 7',
  check: 'M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM8 12l3 3 5-6',
  close: 'M6 6l12 12M18 6L6 18',
  unblocks: 'M7 17L17 7M8 7h9v9',
  file: 'M14 3H6v18h12V7zM14 3v4h4M10 13l-2 2 2 2M14 13l2 2-2 2',
  projects: 'M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z',
  book: 'M2 5h7a3 3 0 0 1 3 3v13a2 2 0 0 0-2-2H2zM22 5h-7a3 3 0 0 0-3 3v13a2 2 0 0 1 2-2h8z',
  warning: 'M12 2l10 10-10 10L2 12zM12 8v5M12 16h.01',
  chart: 'M3 3v18h18M7 15l4-4 3 3 6-6',
  pause: 'M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM10 9v6M14 9v6',
  play: 'M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM10 8l6 4-6 4z',
  arrow: 'M5 12h14M13 6l6 6-6 6',
  classic: 'M3 4h18v16H3zM9 4v16M15 4v16',
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 16 }: { name: IconName; size?: number }): ReactElement {
  return (
    <svg className="mh-icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={PATHS[name]} />
    </svg>
  );
}
