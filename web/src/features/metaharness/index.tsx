// Registers the classic-layout header switch into the metaharness view.
// The view itself renders through the narrow feature's wide layout.
import { registerSlot } from '../../app/slots';
import { setViewMode } from './viewMode';

export function register(): void {
  registerSlot('header-control', () => (
    <button type="button" className="mk-btn mk-btn-sm" onClick={() => setViewMode('metaharness')}>
      Metaharness
    </button>
  ));
}
