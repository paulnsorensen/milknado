// The `sidecar-section` contribution for the Changes tab, and the
// `changes.open` action `node-sidecar`'s Changes tab button dispatches.
import { registerAction } from '../../app/actions';
import { registerSlot } from '../../app/slots';
import { setActiveTab } from '../node-sidecar/detailTab';
import { ChangesSection } from './ChangesSection';

export function register(): void {
  registerSlot('sidecar-section', () => <ChangesSection />);
  registerAction('changes.open', () => setActiveTab('changes'));
}
