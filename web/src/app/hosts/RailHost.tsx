import type { ReactElement } from 'react';
import { renderSlot } from './renderSlot';

/** The rail: wordmark, rail actions, then rail sections. */
export function RailHost(): ReactElement {
  return (
    <nav data-region="rail" aria-label="Milknado">
      <div className="mk-wordmark">
        <img src="/assets/milknado-mark.png" alt="" />
        <span className="mk-text-wordmark">Milknado</span>
      </div>
      <div className="mk-rail-actions">{renderSlot('rail-action')}</div>
      {renderSlot('rail-section')}
    </nav>
  );
}