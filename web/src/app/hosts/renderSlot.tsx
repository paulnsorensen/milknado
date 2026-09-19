// The shared slot-rendering step every region host uses: read a slot's
// contributions in registration order and give each a stable React key.
import { Fragment, type ReactNode } from 'react';
import { getSlot, type SlotId } from '../slots';

export function renderSlot(id: SlotId): ReactNode[] {
  return getSlot(id).map((contribution, index) => (
    <Fragment key={index}>{contribution()}</Fragment>
  ));
}
