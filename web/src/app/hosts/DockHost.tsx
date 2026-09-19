import type { ReactElement } from 'react';
import { renderSlot } from './renderSlot';

/** The dock region: the Console, registered on the `dock` slot. */
export function DockHost(): ReactElement {
  return <div data-region="dock">{renderSlot('dock')}</div>;
}
