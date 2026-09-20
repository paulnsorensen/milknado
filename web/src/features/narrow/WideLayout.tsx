// The default Main artboard region tree, rendered above the 400px
// breakpoint. Reuses `app/DefaultLayout` so the `layout` slot
// contribution falls back to the same region tree at wide viewports.
import type { ReactElement } from 'react';
import { DefaultLayout } from '../../app/DefaultLayout';

export function WideLayout(): ReactElement {
  return <DefaultLayout />;
}
