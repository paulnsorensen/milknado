import type { ReactElement } from 'react';
import { Wordmark } from '../../shared/Wordmark';
import { renderSlot } from './renderSlot';

/** The rail: wordmark, rail actions, then rail sections. */
export function RailHost(): ReactElement {
  return (
    <nav data-region="rail" aria-label="Milknado">
      <Wordmark />
      <div className="mk-rail-actions">{renderSlot('rail-action')}</div>
      {renderSlot('rail-section')}
    </nav>
  );
}