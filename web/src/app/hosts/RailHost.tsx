import type { ReactElement } from 'react';
import { renderSlot } from './renderSlot';

/** The rail region: rail actions above rail sections. */
export function RailHost(): ReactElement {
  return (
    <div data-region="rail">
      {renderSlot('rail-action')}
      {renderSlot('rail-section')}
    </div>
  );
}
