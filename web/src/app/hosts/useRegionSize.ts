// Measures a region element so a canvas-sized component (MikadoGraph) can
// fill it. Without ResizeObserver (jsdom) the size stays undefined and the
// component falls back to its own sizing.
import { useEffect, useState, type RefObject } from 'react';

export interface RegionSize {
  width: number;
  height: number;
}

export function useRegionSize(ref: RefObject<HTMLElement | null>): RegionSize | undefined {
  const [size, setSize] = useState<RegionSize | undefined>(undefined);

  useEffect(() => {
    const element = ref.current;
    if (!element || typeof ResizeObserver === 'undefined') {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect && rect.width > 0 && rect.height > 0) {
        setSize({ width: Math.floor(rect.width), height: Math.floor(rect.height) });
      }
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref]);

  return size;
}
