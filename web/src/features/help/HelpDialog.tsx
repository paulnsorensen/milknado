// The `dialog` slot contribution: the Graph, Runs and Steering shortcut
// columns, generated from the shortcuts feature's key map.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { Milknado } from '../../design-system';
import { Dialog } from '../../shared/dialog/Dialog';
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
    <Dialog title="Keyboard shortcuts" wide onClose={closeHelp} actions={<Button onClick={closeHelp}>Close</Button>}>
      <div className="mk-help-columns">
        {COLUMNS.map((column) => (
          <section key={column} aria-label={column} className="mk-section">
            <h3 className="mk-kicker">{column}</h3>
            <ul>
              {KEY_BINDINGS.filter((binding) => binding.column === column).map((binding) => (
                <li key={binding.key} className="mk-key-row">
                  <span>
                    <span className="mk-kbd">{binding.label}</span>
                  </span>
                  <span>{binding.description}</span>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </Dialog>
  );
}