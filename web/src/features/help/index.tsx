// The `header-control` Keys button and the `dialog` slot Help overlay.
import { registerSlot } from '../../app/slots';
import { HelpControl } from './HelpControl';
import { HelpDialog } from './HelpDialog';

export function register(): void {
  registerSlot('header-control', () => <HelpControl />);
  registerSlot('dialog', () => <HelpDialog />);
}
