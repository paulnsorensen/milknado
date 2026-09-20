// The `dialog` slot contribution: the Graph, Runs and Steering shortcut
// columns, generated from the shortcuts feature's key map.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { Milknado } from '../../design-system';
import { KEY_BINDINGS, type ShortcutColumn } from '../shortcuts/keyMap';
import { closeHelp, isHelpOpen, subscribeHelp } from './helpState';

const COLUMNS: ShortcutColumn[] = ['Graph', 'Runs', 'Steering'];

export function HelpDialog(): ReactElement | null {
  const open = useSyncExternalStore(subscribeHelp, isHelpOpen);
  const { Button } = Milknado;

  if (!open) {
    return null;
  }

  return (
    <div role="dialog" aria-label="Keyboard shortcuts" className="mk-help-dialog">
      {COLUMNS.map((column) => (
        <section key={column} aria-label={column}>
          <h3>{column}</h3>
          <ul>
            {KEY_BINDINGS.filter((binding) => binding.column === column).map((binding) => (
              <li key={binding.key}>
                <span>{binding.label}</span>
                <span>{binding.description}</span>
              </li>
            ))}
          </ul>
        </section>
      ))}
      <Button onClick={closeHelp}>Close</Button>
    </div>
  );
}
