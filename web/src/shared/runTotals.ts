export interface RunTotalsSnapshot {
  active_runs?: readonly unknown[];
  completed?: number;
  failed?: number;
  stopped?: number;
  available?: number;
}

function totalLabel(value: number | undefined, label: string): string {
  return value === undefined ? `– ${label}` : `${value} ${label}`;
}

export function formatRunTotals(snapshot: RunTotalsSnapshot | null): string {
  if (!snapshot) {
    return '';
  }

  return [
    totalLabel(snapshot.active_runs?.length, 'active'),
    totalLabel(snapshot.completed, 'completed'),
    totalLabel(snapshot.failed, 'failed'),
    totalLabel(snapshot.stopped, 'stopped'),
    totalLabel(snapshot.available, 'available'),
  ].join(' · ');
}
