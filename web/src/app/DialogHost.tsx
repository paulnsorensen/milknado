import type { ReactElement } from 'react';
import { renderSlot } from './hosts/renderSlot';

/** Renders every feature contribution registered on the `dialog` slot. */
export function DialogHost(): ReactElement {
  return <div data-region="dialog-host">{renderSlot('dialog')}</div>;
}
