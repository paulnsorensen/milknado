import type { ReactElement } from 'react';
import { renderSlot } from './renderSlot';

/**
 * The sidecar region: the node-detail panel plus its tab, section and
 * action sub-slots.
 */
export function SidecarHost(): ReactElement {
  return (
    <div data-region="sidecar">
      {renderSlot('sidecar')}
      <div data-region="sidecar-tab">{renderSlot('sidecar-tab')}</div>
      <div data-region="sidecar-section">{renderSlot('sidecar-section')}</div>
      <div data-region="sidecar-action">{renderSlot('sidecar-action')}</div>
    </div>
  );
}
