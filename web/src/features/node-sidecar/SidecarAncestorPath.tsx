// Renders the same breadcrumb markup as the vendored `Milknado.AncestorPath`
// (see web/vendor/milknado/components/bundle.js), but computed in React so
// the accessible label and full title never drift from the rendered DOM.
import { Fragment, useMemo, type ReactElement } from 'react';
import { setSelection } from '../../app/store';
import type { GraphNodeData } from '../../app/wire';
import { ancestorPath } from '../../shared/ancestorPath';
import { summarizePathTitle } from './pathTitle';

const PATH_ITEM_LIMIT = 4;

interface PathItem {
  id: GraphNodeData['id'];
  title: string;
  fullTitle: string;
  gap: boolean;
}

function toPathItem(node: GraphNodeData): PathItem {
  return {
    id: node.id,
    title: summarizePathTitle(node.title),
    fullTitle: node.title,
    gap: false,
  };
}

function elide(items: PathItem[]): PathItem[] {
  if (items.length <= PATH_ITEM_LIMIT) {
    return items;
  }
  const gap: PathItem = { id: '…', title: '…', fullTitle: '…', gap: true };
  return [items[0], gap, ...items.slice(-(PATH_ITEM_LIMIT - 1))];
}

export interface SidecarAncestorPathProps {
  nodes: GraphNodeData[];
  nodeId: number;
}

export function SidecarAncestorPath({
  nodes,
  nodeId,
}: SidecarAncestorPathProps): ReactElement {
  const items = useMemo(() => {
    const path = ancestorPath(
      nodes,
      nodeId,
      (node) => node.id,
      (node) => node.parent,
    );
    return elide((path ?? []).map(toPathItem));
  }, [nodes, nodeId]);

  return (
    <nav className="mk mk-path mk-path-wrapper" aria-label="Path to the goal">
      {items.map((item, index) => {
        const isCurrent = index === items.length - 1;
        return (
          <Fragment key={item.id}>
            {index > 0 && (
              <span className="mk-path-sep" aria-hidden="true">
                {'›'}
              </span>
            )}
            {item.gap ? (
              <span className="mk-path-gap">{'…'}</span>
            ) : (
              <button
                type="button"
                className={isCurrent ? 'mk-path-item is-current' : 'mk-path-item'}
                aria-current={isCurrent ? 'page' : undefined}
                title={item.fullTitle}
                aria-label={item.title}
                onClick={() => setSelection(item.id)}
              >
                {item.title}
              </button>
            )}
          </Fragment>
        );
      })}
    </nav>
  );
}
