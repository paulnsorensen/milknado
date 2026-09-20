import type { ReactElement } from 'react';
import { renderSlot } from './renderSlot';

/** The header region: header controls plus the status slot. */
export function HeaderHost(): ReactElement {
  return (
    <div data-region="header">
      {renderSlot('header-control')}
      {renderSlot('status')}
    </div>
  );
}
