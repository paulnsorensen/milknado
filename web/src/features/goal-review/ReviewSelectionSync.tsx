// The `provider` slot contribution: closes an open review the instant a
// node is selected, so the two sidecar panels never share a paint. The
// clearing itself runs inside the store's synchronous emit, ahead of any
// re-render; the effect only wires and tears down that subscription.
import type { ReactElement } from 'react';
import { useEffect } from 'react';
import { getState, subscribe } from '../../app/store';
import { clearReviewSelection, getSelectedReviewId } from './selection';

export function ReviewSelectionSync(): ReactElement | null {
  useEffect(() => {
    return subscribe(() => {
      if (typeof getState().selection === 'number' && getSelectedReviewId() !== null) {
        clearReviewSelection();
      }
    });
  }, []);

  return null;
}
