// The `layout` slot contribution: the narrow list/detail layout at or
// below the 400px breakpoint (AC-15), the wide default layout above it.
import { registerSlot } from '../../app/slots';
import './narrow.css';
import { NarrowLayout } from './NarrowLayout';

export function register(): void {
  registerSlot('layout', () => <NarrowLayout />);
}
