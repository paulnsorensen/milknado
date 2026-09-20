import { describe, expect, it } from 'vitest';
import { toBadgeState } from './badgeState';

describe('toBadgeState', () => {
  it('passes through a known state', () => {
    expect(toBadgeState('running')).toBe('running');
  });

  it('falls back to pending for an unknown state', () => {
    expect(toBadgeState('mystery')).toBe('pending');
  });
});
