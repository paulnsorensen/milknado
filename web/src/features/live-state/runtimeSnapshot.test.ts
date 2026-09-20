import { describe, expect, it } from 'vitest';
import { DEFAULT_CAPABILITIES, mergeSnapshot, type RawStreamSnapshot } from './runtimeSnapshot';

function rawSnapshot(): RawStreamSnapshot {
  return {
    goal: 'Ship the tracer',
    graph: null,
    active_runs: [],
    event_lines: ['line one'],
  };
}

describe('mergeSnapshot', () => {
  it('keeps the raw payload capabilities when the stream sends them', () => {
    const withCapabilities = { ...rawSnapshot(), capabilities: DEFAULT_CAPABILITIES };

    const merged = mergeSnapshot(withCapabilities, null);

    expect(merged.capabilities).toBe(DEFAULT_CAPABILITIES);
  });

  it('carries over the previous capabilities when the stream omits them', () => {
    const previous = { ...DEFAULT_CAPABILITIES, cancel: { available: true, reason: null } };

    const merged = mergeSnapshot(rawSnapshot(), previous);

    expect(merged.capabilities).toBe(previous);
  });

  it('falls back to a default capabilities object with no prior snapshot', () => {
    const merged = mergeSnapshot(rawSnapshot(), null);

    expect(merged.capabilities).toEqual(DEFAULT_CAPABILITIES);
  });

  it('preserves the run and event data from the raw payload', () => {
    const raw = { ...rawSnapshot(), event_lines: ['a new line'] };

    const merged = mergeSnapshot(raw, null);

    expect(merged.event_lines).toEqual(['a new line']);
  });
});
