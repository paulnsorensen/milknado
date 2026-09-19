import type { ReactElement } from 'react';
import { renderSlot } from './renderSlot';

/** The toast stack, anchored bottom-right, above every other region. */
export function ToastHost(): ReactElement {
  return <div data-region="toast">{renderSlot('toast')}</div>;
}
