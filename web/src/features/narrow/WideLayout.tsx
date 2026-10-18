// The wide Main artboard, rendered above the 400px breakpoint: the classic
// region tree from `app/DefaultLayout`, or the metaharness view when the
// viewer picked it.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { DefaultLayout } from '../../app/DefaultLayout';
import { getViewMode, subscribeViewMode } from '../metaharness/viewMode';
import { MetaharnessView } from '../metaharness/ui/MetaharnessView';

export function WideLayout(): ReactElement {
  const mode = useSyncExternalStore(subscribeViewMode, getViewMode);
  return mode === 'metaharness' ? <MetaharnessView /> : <DefaultLayout />;
}
