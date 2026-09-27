import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getActiveTab, subscribeTab } from '../../shared/node-detail';
import { getState, subscribe } from '../store';
import { renderSlot } from './renderSlot';

/**
 * The sidecar: the node or review panel, the owner-run actions, then the
 * detail tabs and their sections. The tab strip needs a selected node; the
 * sections and actions also carry owner-run controls, so they always render.
 * The Changes tab widens the panel to the review width. The goal-review
 * feature clears its own selection when a node is selected, so the two
 * panels never render at once.
 */
export function SidecarHost(): ReactElement {
  const store = useSyncExternalStore(subscribe, getState);
  const activeTab = useSyncExternalStore(subscribeTab, getActiveTab);
  const nodeSelected = typeof store.selection === 'number';
  const wide = nodeSelected && activeTab === 'changes';

  return (
    <aside data-region="sidecar" aria-label="Detail" className={wide ? 'is-wide' : undefined}>
      {renderSlot('sidecar')}
      <div data-region="sidecar-action">{renderSlot('sidecar-action')}</div>
      {nodeSelected && (
        <div data-region="sidecar-tab" role="tablist" aria-label="Node detail">
          {renderSlot('sidecar-tab')}
        </div>
      )}
      <div data-region="sidecar-section">{renderSlot('sidecar-section')}</div>
    </aside>
  );
}