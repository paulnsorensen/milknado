// Feeds the live snapshot stream into the single client store and shows a
// reconnecting status while the stream is down (AC-3, AC-4, AC-5).
import { registerAction } from '../../app/actions';
import { registerSlot } from '../../app/slots';
import { openEvents, selectNextRun, selectPreviousRun } from './actions';
import { StatusIndicator } from './StatusIndicator';
import { StreamProvider } from './StreamProvider';

export function register(): void {
  registerSlot('provider', () => <StreamProvider />);
  registerSlot('status', () => <StatusIndicator />);
  registerAction('events.open', openEvents);
  registerAction('run.select-next', selectNextRun);
  registerAction('run.select-previous', selectPreviousRun);
}
