// The `sidecar` slot contribution: the node detail panel for the selected
// node, plus its three tab buttons in the `sidecar-tab` slot.
import { registerSlot } from '../../app/slots';
import { ChangesTabButton, DetailsTabButton, SessionTabButton } from './DetailTabButtons';
import { NodeSidecar } from './NodeSidecar';

export function register(): void {
  registerSlot('sidecar', () => <NodeSidecar />);
  registerSlot('sidecar-tab', () => <SessionTabButton />);
  registerSlot('sidecar-tab', () => <ChangesTabButton />);
  registerSlot('sidecar-tab', () => <DetailsTabButton />);
}
