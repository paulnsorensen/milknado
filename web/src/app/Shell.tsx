import type { ReactElement } from 'react';
import { useEffect } from 'react';
import { get } from './api';
import { DefaultLayout } from './DefaultLayout';
import { DialogHost } from './DialogHost';
import { ProviderHost } from './hosts/ProviderHost';
import { ToastHost } from './hosts/ToastHost';
import { registerFeatures } from './registry';
import './shell.css';
import { getSlot } from './slots';
import { pushNotice, setSnapshot } from './store';
import type { WireExecutionSnapshot } from './wire';

registerFeatures();

function loadSnapshot(): void {
  get<WireExecutionSnapshot>('/api/snapshot')
    .then((snapshot) => {
      if (snapshot) {
        setSnapshot(snapshot);
      }
    })
    .catch(() => pushNotice('Failed to load the execution snapshot.'));
}

/** The Main artboard: header, rail, canvas, dock and sidecar regions. */
export function Shell(): ReactElement {
  useEffect(loadSnapshot, []);

  const layouts = getSlot('layout');

  return (
    <div className="mk-shell">
      <ProviderHost />
      {layouts.length > 0 ? layouts[0]() : <DefaultLayout />}
      <DialogHost />
      <ToastHost />
    </div>
  );
}
