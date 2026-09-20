export interface DockLine {
  time: string;
  text: string;
}

export function toConsoleLines(eventLines: string[]): DockLine[] {
  return eventLines.map((text) => ({ time: '', text }));
}
