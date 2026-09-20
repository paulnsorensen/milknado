// The narrow detail view: a full-width node detail panel, reusing curd 3's
// node-sidecar content, with a back control to return to the list.
import type { ReactElement } from 'react';
import { NodeSidecar } from '../node-sidecar/NodeSidecar';

export interface NarrowDetailProps {
  onBack: () => void;
}

/** The `layout` slot's full-width detail view, shown after opening a node. */
export function NarrowDetail({ onBack }: NarrowDetailProps): ReactElement {
  return (
    <div className="mk-narrow-root mk-narrow-detail">
      <button type="button" className="mk-narrow-back" onClick={onBack}>
        Back to list
      </button>
      <NodeSidecar />
    </div>
  );
}
