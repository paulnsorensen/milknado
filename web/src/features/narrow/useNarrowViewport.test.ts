import { cleanup, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useNarrowViewport } from './useNarrowViewport';

function stubMatchMedia(matches: boolean): void {
  const addEventListener = vi.fn();
  const removeEventListener = vi.fn();
  vi.stubGlobal('matchMedia', vi.fn().mockReturnValue({ matches, addEventListener, removeEventListener }));
}

afterEach(() => {
  vi.unstubAllGlobals();
  cleanup();
});

describe('useNarrowViewport', () => {
  it('is false when the browser cannot report a preference', () => {
    vi.stubGlobal('matchMedia', undefined);

    const { result } = renderHook(() => useNarrowViewport());

    expect(result.current).toBe(false);
  });

  it('reflects a narrow viewport', () => {
    stubMatchMedia(true);

    const { result } = renderHook(() => useNarrowViewport());

    expect(result.current).toBe(true);
  });

  it('reflects a wide viewport', () => {
    stubMatchMedia(false);

    const { result } = renderHook(() => useNarrowViewport());

    expect(result.current).toBe(false);
  });

  it('subscribes to the media query change event and unsubscribes on unmount', () => {
    const addEventListener = vi.fn();
    const removeEventListener = vi.fn();
    vi.stubGlobal(
      'matchMedia',
      vi.fn().mockReturnValue({ matches: false, addEventListener, removeEventListener }),
    );

    const { unmount } = renderHook(() => useNarrowViewport());
    expect(addEventListener).toHaveBeenCalledWith('change', expect.any(Function));

    unmount();
    expect(removeEventListener).toHaveBeenCalledWith('change', expect.any(Function));
  });
});
