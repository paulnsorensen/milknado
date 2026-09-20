// The `header-control` contribution: the Keys button that opens the Help
// dialog, mirroring the `?` shortcut.
import type { ReactElement } from 'react';
import { Milknado } from '../../design-system';
import { openHelp } from './helpState';

export function HelpControl(): ReactElement {
  const { Button } = Milknado;

  return (
    <Button ariaLabel="Keyboard shortcuts" onClick={openHelp}>
      Keys
    </Button>
  );
}
