// The `toast` and `banner` contributions: store notices (a 409's domain
// reason) and the snapshot's listener errors.
import { registerSlot } from '../../app/slots';
import { ErrorBanner } from './ErrorBanner';
import { NoticeToasts } from './NoticeToasts';

export function register(): void {
  registerSlot('toast', () => <NoticeToasts />);
  registerSlot('banner', () => <ErrorBanner />);
}
