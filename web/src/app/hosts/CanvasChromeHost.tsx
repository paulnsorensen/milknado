import type { ReactElement } from 'react';
import { renderSlot } from './renderSlot';

/** The error banner above the canvas toolbar. */
export function CanvasBanner(): ReactElement {
  return <div data-region="banner">{renderSlot('banner')}</div>;
}

/** The toolbar above the graph: search, filters, zoom controls. */
export function CanvasToolbar(): ReactElement {
  return <div data-region="toolbar">{renderSlot('toolbar')}</div>;
}

/** The canvas overlay, anchored top-right (for example the Minimap). */
export function CanvasOverlay(): ReactElement {
  return <div data-region="canvas-overlay">{renderSlot('canvas-overlay')}</div>;
}
