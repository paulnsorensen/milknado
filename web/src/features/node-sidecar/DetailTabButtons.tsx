import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { dispatchAction } from '../../app/actions';
import { Milknado } from '../../design-system';
import { getActiveTab, setActiveTab, subscribeTab, type DetailTab } from './detailTab';

function TabButton({ tab, label }: { tab: DetailTab; label: string }): ReactElement {
  const activeTab = useSyncExternalStore(subscribeTab, getActiveTab);
  const { Button } = Milknado;

  return (
    <Button variant={activeTab === tab ? 'primary' : 'secondary'} onClick={() => setActiveTab(tab)}>
      {label}
    </Button>
  );
}

export function SessionTabButton(): ReactElement {
  return <TabButton tab="session" label="Session" />;
}

export function DetailsTabButton(): ReactElement {
  return <TabButton tab="details" label="Details" />;
}

export function ChangesTabButton(): ReactElement {
  const activeTab = useSyncExternalStore(subscribeTab, getActiveTab);
  const { Button } = Milknado;

  return (
    <Button
      variant={activeTab === 'changes' ? 'primary' : 'secondary'}
      onClick={() => dispatchAction('changes.open')}
    >
      Changes
    </Button>
  );
}
