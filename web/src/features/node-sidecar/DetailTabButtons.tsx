// The `sidecar-tab` contributions: Session, Changes and Details as console
// tabs (kicker type, the active one lit). Changes goes through the
// `changes.open` action so the changes feature owns that transition.
import type { KeyboardEvent, ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { dispatchAction } from '../../app/actions';
import {
  detailTabId,
  detailTabPanelId,
  getActiveTab,
  setActiveTab,
  subscribeTab,
  type DetailTab,
} from '../../shared/node-detail';

interface TabProps {
  tab: DetailTab;
  label: string;
  onSelect: () => void;
}
function moveTab(event: KeyboardEvent<HTMLButtonElement>, direction: -1 | 1): void {
  const tablist = event.currentTarget.closest('[role="tablist"]');
  const tabs = Array.from(tablist?.querySelectorAll<HTMLButtonElement>('[role="tab"]') ?? []);
  const currentIndex = tabs.indexOf(event.currentTarget);
  if (currentIndex < 0) {
    return;
  }
  const nextIndex = (currentIndex + direction + tabs.length) % tabs.length;
  const nextTab = tabs[nextIndex];
  nextTab.focus();
  nextTab.click();
}

function handleKeyDown(event: KeyboardEvent<HTMLButtonElement>): void {
  if (event.key === 'ArrowRight') {
    event.preventDefault();
    moveTab(event, 1);
  } else if (event.key === 'ArrowLeft') {
    event.preventDefault();
    moveTab(event, -1);
  }
}

function Tab({ tab, label, onSelect }: TabProps): ReactElement {
  const activeTab = useSyncExternalStore(subscribeTab, getActiveTab);
  const active = activeTab === tab;

  return (
    <button
      id={detailTabId(tab)}
      type="button"
      role="tab"
      aria-selected={active}
      aria-controls={detailTabPanelId(tab)}
      tabIndex={active ? 0 : -1}
      className={active ? 'mk-console-tab mk-kicker is-live' : 'mk-console-tab mk-kicker'}
      onClick={onSelect}
      onKeyDown={handleKeyDown}
    >
      {label}
    </button>
  );
}

export function SessionTabButton(): ReactElement {
  return <Tab tab="session" label="Session" onSelect={() => setActiveTab('session')} />;
}

export function DetailsTabButton(): ReactElement {
  return <Tab tab="details" label="Details" onSelect={() => setActiveTab('details')} />;
}

export function ChangesTabButton(): ReactElement {
  return <Tab tab="changes" label="Changes" onSelect={() => dispatchAction('changes.open')} />;
}
