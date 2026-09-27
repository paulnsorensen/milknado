// The narrow detail view: the node-sidecar content stacked full width
// (node header, owner-run actions, detail tabs and their sections), with a
// back control to return to the list.
import type { ReactElement } from 'react';
import { renderSlot } from '../../app/hosts/renderSlot';
import { Milknado } from '../../design-system';
import { NodeSidecar } from '../node-sidecar/NodeSidecar';

export interface NarrowDetailProps {
  onBack: () => void;
}

/** The `layout` slot's full-width detail view, shown after opening a node. */
export function NarrowDetail({ onBack }: NarrowDetailProps): ReactElement {
  const { Button } = Milknado;
  return (
    <div className="mk-narrow-root mk-narrow-detail">
      <header className="mk-narrow-bar">
        <Button variant="ghost" glyph={'‹'} className="mk-narrow-back" onClick={onBack}>
          Back to list
        </Button>
      </header>
      <div className="mk-narrow-body">
        <NodeSidecar onClose={onBack} />
        <div data-region="sidecar-action">{renderSlot('sidecar-action')}</div>
        <div role="group" aria-label="Node detail" className="mk-narrow-tabs">
          {renderSlot('sidecar-tab')}
        </div>
        <div data-region="sidecar-section">{renderSlot('sidecar-section')}</div>
      </div>
    </div>
  );
}