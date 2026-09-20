// Which sidecar tab is active for the selected node: Session, Changes or
// Details. A plain pub-sub so the `changes` and `session-input` features can
// read it without a shared parent component.
export type DetailTab = 'session' | 'changes' | 'details';

let activeTab: DetailTab = 'session';
const listeners = new Set<() => void>();

function emit(): void {
  for (const listener of listeners) {
    listener();
  }
}

export function getActiveTab(): DetailTab {
  return activeTab;
}

export function setActiveTab(tab: DetailTab): void {
  if (activeTab === tab) {
    return;
  }
  activeTab = tab;
  emit();
}

export function subscribeTab(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function resetTab(): void {
  activeTab = 'session';
  emit();
}
