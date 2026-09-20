// The `layout` slot contribution. A registered layout replaces the whole
// Main artboard tree (see Shell.tsx), so this component renders the wide
// default layout unchanged above 400px, and the narrow list/detail layout
// at or below it (AC-15).
import type { ReactElement } from 'react';
import { useState } from 'react';
import { setSelection } from '../../app/store';
import { NarrowDetail } from './NarrowDetail';
import { NarrowList } from './NarrowList';
import { useNarrowViewport } from './useNarrowViewport';
import { WideLayout } from './WideLayout';

type NarrowView = 'list' | 'detail';

export function NarrowLayout(): ReactElement {
  const narrow = useNarrowViewport();
  const [view, setView] = useState<NarrowView>('list');

  if (!narrow) {
    return <WideLayout />;
  }

  function openNode(id: string | number): void {
    setSelection(id);
    setView('detail');
  }

  if (view === 'detail') {
    return <NarrowDetail onBack={() => setView('list')} />;
  }

  return <NarrowList onOpen={openNode} />;
}
