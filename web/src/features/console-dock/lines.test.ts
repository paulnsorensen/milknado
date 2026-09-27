import { describe, expect, it } from 'vitest';
import { toConsoleLines } from './lines';

describe('toConsoleLines', () => {
  it('maps each event line to a console line', () => {
    expect(toConsoleLines(['a new line'])).toEqual([{ time: '', text: 'a new line' }]);
  });

  it('splits a leading timestamp into the time column', () => {
    expect(toConsoleLines(['14:04:19  node 8 · gate PASS'])).toEqual([
      { time: '14:04:19', text: 'node 8 · gate PASS' },
    ]);
  });

  it('maps an empty line list to no lines', () => {
    expect(toConsoleLines([])).toEqual([]);
  });
});
