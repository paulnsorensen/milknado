// The one global key listener and the `help.open` action id it dispatches
// through the Help feature's shared open/closed state.
import { registerAction } from '../../app/actions';
import { registerSlot } from '../../app/slots';
import { openHelp } from '../help/helpState';
import { ShortcutsProvider } from './ShortcutsProvider';

export function register(): void {
  registerSlot('provider', () => <ShortcutsProvider />);
  registerAction('help.open', openHelp);
}
