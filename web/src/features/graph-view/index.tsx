// The `toolbar` and `canvas-overlay` slot contributions: the GraphToolbar
// and Minimap design-system components.
import { registerSlot } from '../../app/slots';
import { GraphToolbarFeature } from './GraphToolbarFeature';
import { MinimapFeature } from './MinimapFeature';

export function register(): void {
  registerSlot('toolbar', () => <GraphToolbarFeature />);
  registerSlot('canvas-overlay', () => <MinimapFeature />);
}
