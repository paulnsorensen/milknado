import type { ReactElement } from 'react';
import { dispatchAction } from '../../app/actions';
import { Milknado } from '../../design-system';
import type { WireNodeDetailSnapshot } from './detailWire';

export interface DetailsTabProps {
  detail: WireNodeDetailSnapshot;
  hasMore: boolean;
}

/** The Details tab body: the node's fields, plus detail paging. */
export function DetailsTab({ detail, hasMore }: DetailsTabProps): ReactElement {
  const { Button } = Milknado;

  return (
    <div>
      <p>{detail.description}</p>
      <dl>
        <dt>Parent</dt>
        <dd>{detail.parent?.id ?? 'none'}</dd>
        <dt>Ancestors</dt>
        <dd>{detail.ancestors.items?.length ?? 0}</dd>
        <dt>Prerequisites</dt>
        <dd>{detail.prerequisite_ids.items?.join(', ') ?? ''}</dd>
        <dt>Dependents</dt>
        <dd>{detail.dependent_ids.items?.join(', ') ?? ''}</dd>
        <dt>Owned files</dt>
        <dd>{detail.owned_files.items?.join(', ') ?? ''}</dd>
      </dl>
      <Button onClick={() => dispatchAction('detail.page-previous')}>Previous</Button>
      <Button disabled={!hasMore} onClick={() => dispatchAction('detail.page-next')}>
        Next
      </Button>
    </div>
  );
}
