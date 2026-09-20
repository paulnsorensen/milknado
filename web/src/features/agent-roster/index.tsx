// The `rail-section` slot contribution: the agent roster, from the runs on
// the live snapshot the `live-state` feature keeps current.
import { registerSlot } from '../../app/slots';
import { AgentRosterSection } from './AgentRosterSection';

export function register(): void {
  registerSlot('rail-section', () => <AgentRosterSection />);
}
