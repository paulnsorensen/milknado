// The `sidecar-tab` contributions: Session, Changes and Details as console
// tabs (kicker type, the active one lit). Changes goes through the
// `changes.open` action so the changes feature owns that transition.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { dispatchAction } from '../../app/actions';
import { getActiveTab, setActiveTab, subscribeTab, type DetailTab } from '../../shared/node-detail';

interface TabProps {
  tab: DetailTab;
  label: string;
  onSelect: () => void;
}

function Tab({ tab, label, onSelect }: TabProps): ReactElement {
  const activeTab = useSyncExternalStore(subscribeTab, getActiveTab);
  const active = activeTab === tab;

  return (
    <button
      type="button"
      aria-pressed={active}
      className={active ? 'mk-console-tab mk-kicker is-live' : 'mk-console-tab mk-kicker'}
      onClick={onSelect}
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