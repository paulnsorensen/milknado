// Registers the classic-layout header switch into the metaharness view and
// its New goal dialog. The view itself renders through the narrow feature's
// wide layout.
import { registerSlot } from '../../app/slots';
import { NewGoalDialog } from './ui/NewGoalDialog';
import { setViewMode } from './viewMode';

export function register(): void {
  registerSlot('header-control', () => (
    <button type="button" className="mk-btn mk-btn-sm" onClick={() => setViewMode('metaharness')}>
      Metaharness
    </button>
  ));
  registerSlot('dialog', () => <NewGoalDialog />);
}
