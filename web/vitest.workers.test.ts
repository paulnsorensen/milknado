import { describe, expect, it } from 'vitest';
import { freeCoreWorkers } from './vitest.workers';

describe('freeCoreWorkers', () => {
  it.each([
    [{}, 0.5, 23],
    [{}, 19, 5],
    [{}, 80, 2],
    [{ TEST_WORKERS_MAX: '3' }, 0.5, 3],
    [{ TEST_WORKERS_MAX: '1' }, 80, 1],
  ])('sizes %o at load %d to %d workers', (env, load, workers) => {
    expect(freeCoreWorkers(env, 24, load)).toBe(workers);
  });

  it('rejects a cap that is not a positive integer', () => {
    expect(() => freeCoreWorkers({ TEST_WORKERS_MAX: 'many' }, 24, 0)).toThrow(
      'TEST_WORKERS_MAX must be a positive integer',
    );
  });
});
