import type { ReactElement } from 'react';
import { useEffect } from 'react';
import { get } from './api';
import { DefaultLayout } from './DefaultLayout';
import { DialogHost } from './DialogHost';
import { ProviderHost } from './hosts/ProviderHost';
import { ToastHost } from './hosts/ToastHost';
import './shell.css';
import { getSlot } from './slots';
import { pushNotice, setSnapshot } from './store';
import type { WireExecutionSnapshot } from './wire';

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
  if (layouts.length > 1) {
    throw new Error(`Expected at most one layout contribution, found ${layouts.length}.`);
  }

  return (
    <div className="mk-shell">
      <ProviderHost />
      {layouts.length > 0 ? layouts[0].contribution() : <DefaultLayout />}
      <DialogHost />
      <ToastHost />
    </div>
  );
}
