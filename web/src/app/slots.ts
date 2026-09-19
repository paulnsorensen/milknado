// The named extension points a feature can render into. Features register
// contributions with `registerSlot`; the shell reads them with `getSlot`.
import type { ReactNode } from 'react';

export const SLOT_IDS = [
  'provider',
  'layout',
  'rail-action',
  'rail-section',
  'header-control',
  'status',
  'banner',
  'toolbar',
  'canvas-overlay',
  'dock',
  'sidecar',
  'sidecar-tab',
  'sidecar-section',
  'sidecar-action',
  'dialog',
  'toast',
] as const;

export type SlotId = (typeof SLOT_IDS)[number];

export type SlotContribution = () => ReactNode;

const slots = new Map<SlotId, SlotContribution[]>();

export function registerSlot(id: SlotId, contribution: SlotContribution): void {
  const existing = slots.get(id) ?? [];
  existing.push(contribution);
  slots.set(id, existing);
}

export function getSlot(id: SlotId): SlotContribution[] {
  return slots.get(id) ?? [];
}

export function clearSlots(): void {
  slots.clear();
}
