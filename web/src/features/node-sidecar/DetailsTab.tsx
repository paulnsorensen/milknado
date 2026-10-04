import type { ReactElement } from 'react';
import { dispatchAction } from '../../app/actions';
import { Milknado } from '../../design-system';
import type { WireNodeDetailSnapshot } from '../../shared/node-detail';

export interface DetailsTabProps {
  detail: WireNodeDetailSnapshot;
  hasMore: boolean;
}

function joinIds(items: Array<number | string> | null): string {
  return items && items.length > 0 ? items.join(', ') : 'none';
}

/** The Details tab body: the brief, the node's relations, plus detail paging. */
export function DetailsTab({ detail, hasMore }: DetailsTabProps): ReactElement {
  const { Button } = Milknado;
  const rows: Array<[string, string]> = [
    ['Parent', detail.parent ? `${detail.parent.id} · ${detail.parent.description}` : 'none'],
    ['Ancestors', String(detail.ancestors.items?.length ?? 0)],
    ['Prerequisites', joinIds(detail.prerequisite_ids.items)],
    ['Dependents', joinIds(detail.dependent_ids.items)],
    ['Owned files', joinIds(detail.owned_files.items)],
  ];

  return (
    <div className="mk-console mk-well mk-stack">
      <section className="mk-section">
        <span className="mk-kicker">Brief</span>
        <p className="mk-text-body">{detail.description}</p>
      </section>
      <section className="mk-section">
        <div className="mk-rail-head" style={{ padding: 0 }}>
          <span className="mk-kicker">Detail data</span>
          <div className="mk-button-row">
            <Button icon className="mk-btn-ctl" ariaLabel="Previous" onClick={() => dispatchAction('detail.page-previous')}>
              {'‹'}
            </Button>
            <Button
              icon
              className="mk-btn-ctl"
              ariaLabel="Next"
              disabled={!hasMore}
              onClick={() => dispatchAction('detail.page-next')}
            >
              {'›'}
            </Button>
          </div>
        </div>
        <dl>
          {rows.map(([key, value]) => (
            <div key={key} className="mk-kv">
              <dt>{key}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      </section>
    </div>
  );
}