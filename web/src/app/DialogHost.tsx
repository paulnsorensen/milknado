import type { ReactElement } from 'react';
import { getSlot } from './slots';

/** Renders every feature contribution registered on the `dialog` slot. */
export function DialogHost(): ReactElement {
  const dialogs = getSlot('dialog');
  return (
    <div data-region="dialog-host">
      {dialogs.map((render, index) => (
        <div key={index}>{render()}</div>
      ))}
    </div>
  );
}
