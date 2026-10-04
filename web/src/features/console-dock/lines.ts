export interface DockLine {
  time: string;
  text: string;
}

const TIMED_LINE = /^(\d{2}:\d{2}:\d{2})\s+(.*)$/s;

/** Splits a leading HH:MM:SS stamp into the console's time column. */
export function toConsoleLines(eventLines: string[]): DockLine[] {
  return eventLines.map((line) => {
    const match = TIMED_LINE.exec(line);
    return match ? { time: match[1], text: match[2] } : { time: '', text: line };
  });
}