// The single source for every keyboard shortcut: the global listener matches
// `event.key` against these entries, and the Help dialog renders one row per
// entry, so a listed shortcut always has a working binding.
import { dispatchAction, type ActionId } from '../../app/actions';
import { getState } from '../../app/store';
import type { WireCapabilities } from '../../app/wire';
import {
  clearSelection,
  openSelectedDetails,
  selectFirstChild,
  selectNextSibling,
  selectParent,
  selectPreviousSibling,
  toggleSelectedCollapsed,
} from './graphNav';

export type ShortcutColumn = 'Graph' | 'Runs' | 'Steering';

export interface Shortcut {
  key: string;
  label: string;
  description: string;
  column: ShortcutColumn;
  run: () => void;
}

function act(id: ActionId): () => void {
  return () => dispatchAction(id);
}

function gated(read: (capabilities: WireCapabilities) => boolean, id: ActionId): () => void {
  return () => {
    const capabilities = getState().capabilities;
    if (capabilities !== null && read(capabilities)) {
      dispatchAction(id);
    }
  };
}

export const KEY_BINDINGS: Shortcut[] = [
  {
    key: 'ArrowUp',
    label: 'Up',
    description: 'Select the previous sibling node',
    column: 'Graph',
    run: selectPreviousSibling,
  },
  {
    key: 'ArrowDown',
    label: 'Down',
    description: 'Select the next sibling node',
    column: 'Graph',
    run: selectNextSibling,
  },
  {
    key: 'ArrowLeft',
    label: 'Left',
    description: 'Select the parent node',
    column: 'Graph',
    run: selectParent,
  },
  {
    key: 'ArrowRight',
    label: 'Right',
    description: 'Select the first child node',
    column: 'Graph',
    run: selectFirstChild,
  },
  {
    key: 'Tab',
    label: 'Tab',
    description: 'Select the next sibling node',
    column: 'Graph',
    run: selectNextSibling,
  },
  {
    key: 'Enter',
    label: 'Enter',
    description: 'Open the Details tab for the selected node',
    column: 'Graph',
    run: openSelectedDetails,
  },
  {
    key: 'Escape',
    label: 'Esc',
    description: 'Clear the graph selection',
    column: 'Graph',
    run: clearSelection,
  },
  {
    key: ' ',
    label: 'Space',
    description: 'Collapse or expand the selected node',
    column: 'Graph',
    run: toggleSelectedCollapsed,
  },
  {
    key: 'n',
    label: 'N',
    description: 'Select the next active run',
    column: 'Runs',
    run: act('run.select-next'),
  },
  {
    key: 'p',
    label: 'P',
    description: 'Select the previous active run',
    column: 'Runs',
    run: act('run.select-previous'),
  },
  {
    key: 'e',
    label: 'E',
    description: 'Open the events sidecar',
    column: 'Runs',
    run: act('events.open'),
  },
  {
    key: 'c',
    label: 'C',
    description: 'Open the changes tab',
    column: 'Runs',
    run: act('changes.open'),
  },
  {
    key: ']',
    label: ']',
    description: 'Go to the next detail page',
    column: 'Runs',
    run: act('detail.page-next'),
  },
  {
    key: '[',
    label: '[',
    description: 'Go to the previous detail page',
    column: 'Runs',
    run: act('detail.page-previous'),
  },
  {
    key: '?',
    label: '?',
    description: 'Open the keyboard shortcuts help',
    column: 'Runs',
    run: act('help.open'),
  },
  {
    key: 'j',
    label: 'J',
    description: 'Go to the next session page',
    column: 'Runs',
    run: gated((c) => c.session_input.available, 'session.page-next'),
  },
  {
    key: 'k',
    label: 'K',
    description: 'Go to the previous session page',
    column: 'Runs',
    run: gated((c) => c.session_input.available, 'session.page-previous'),
  },
  {
    key: 'f',
    label: 'F',
    description: 'Follow the newest session event',
    column: 'Runs',
    run: gated((c) => c.session_input.available, 'session.follow-newest'),
  },
  {
    key: 'i',
    label: 'I',
    description: 'Focus the guidance input',
    column: 'Runs',
    run: gated((c) => c.session_input.available, 'session.focus-input'),
  },
  {
    key: 'g',
    label: 'G',
    description: 'Queue the guidance draft',
    column: 'Runs',
    run: gated((c) => c.session_input.available, 'session.queue-guidance'),
  },
  {
    key: 'x',
    label: 'X',
    description: 'Cancel the current run',
    column: 'Steering',
    run: gated((c) => c.cancel.available && Boolean(c.owner.run_id), 'run.cancel'),
  },
  {
    key: 'X',
    label: 'Shift+X',
    description: 'Force stop the current run',
    column: 'Steering',
    run: gated((c) => c.force_stop.available, 'run.force-stop'),
  },
  {
    key: 's',
    label: 'S',
    description: 'Stop scheduling new runs',
    column: 'Steering',
    run: gated((c) => c.stop_scheduling.available, 'scheduling.stop'),
  },
];
