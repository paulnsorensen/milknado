// The `dock` slot contribution: the console, from the event lines on the
// live snapshot the `live-state` feature keeps current. Read-only: no
// guidance input on this dock.
import { registerSlot } from '../../app/slots';
import { ConsoleDockSection } from './ConsoleDockSection';

export function register(): void {
  registerSlot('dock', () => <ConsoleDockSection />);
}
