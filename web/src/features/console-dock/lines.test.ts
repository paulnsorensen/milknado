import { describe, expect, it } from 'vitest';
import { toConsoleLines } from './lines';

describe('toConsoleLines', () => {
  it('maps each event line to a console line', () => {
    expect(toConsoleLines(['a new line'])).toEqual([{ time: '', text: 'a new line' }]);
  });

  it('maps an empty line list to no lines', () => {
    expect(toConsoleLines([])).toEqual([]);
  });
});
