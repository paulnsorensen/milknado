// The `sidecar` slot contribution: the node detail panel for the selected
// node, its three tabs in the `sidecar-tab` slot, and the Session and
// Details tab bodies in the `sidecar-section` slot.
import { registerSlot } from '../../app/slots';
import { ChangesTabButton, DetailsTabButton, SessionTabButton } from './DetailTabButtons';
import { NodeSidecar } from './NodeSidecar';
import { DetailsTabSection, SessionTabSection } from './TabSections';

export function register(): void {
  registerSlot('sidecar', () => <NodeSidecar />);
  registerSlot('sidecar-tab', () => <SessionTabButton />);
  registerSlot('sidecar-tab', () => <ChangesTabButton />);
  registerSlot('sidecar-tab', () => <DetailsTabButton />);
  registerSlot('sidecar-section', () => <SessionTabSection />);
  registerSlot('sidecar-section', () => <DetailsTabSection />);
}