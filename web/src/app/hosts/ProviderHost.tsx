import type { ReactElement } from 'react';
import { renderSlot } from './renderSlot';

/**
 * Mounts every `provider` contribution once, near the shell root, so its
 * effects run regardless of which layout renders. Providers render null.
 */
export function ProviderHost(): ReactElement {
  return <>{renderSlot('provider')}</>;
}
