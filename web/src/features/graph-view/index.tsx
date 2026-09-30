// The `toolbar` slot contribution: the GraphToolbar design-system component.
import { registerSlot } from '../../app/slots';
import { GraphToolbarFeature } from './GraphToolbarFeature';

export function register(): void {
  registerSlot('toolbar', () => <GraphToolbarFeature />);
}
