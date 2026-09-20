// The `provider` slot contribution: the one document-level keydown listener
// for the shortcut key map.
import type { ReactElement } from 'react';
import { useEffect } from 'react';
import { handleShortcutKey } from './listener';

export function ShortcutsProvider(): ReactElement | null {
  useEffect(() => {
    document.addEventListener('keydown', handleShortcutKey);
    return () => document.removeEventListener('keydown', handleShortcutKey);
  }, []);

  return null;
}
