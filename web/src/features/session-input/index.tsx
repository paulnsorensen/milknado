// The `sidecar-section` guidance input and the `sidecar-action` permission
// buttons, plus every action id `node-sidecar`'s paging/session buttons and
// this feature's own focus/guidance shortcuts dispatch to.
import { registerAction } from '../../app/actions';
import { registerSlot } from '../../app/slots';
import { focusSessionInput, queueGuidance } from './draft';
import {
  followNewest,
  pageNext,
  pagePrevious,
  sessionPageNext,
  sessionPagePrevious,
} from '../../shared/node-detail';
import { PermissionActions } from './PermissionActions';
import { SessionInputSection } from './SessionInputSection';

export function register(): void {
  registerSlot('sidecar-section', () => <SessionInputSection />);
  registerSlot('sidecar-action', () => <PermissionActions />);

  registerAction('detail.page-next', pageNext);
  registerAction('detail.page-previous', pagePrevious);
  registerAction('session.page-next', sessionPageNext);
  registerAction('session.page-previous', sessionPagePrevious);
  registerAction('session.follow-newest', followNewest);
  registerAction('session.focus-input', focusSessionInput);
  registerAction('session.queue-guidance', queueGuidance);
}
