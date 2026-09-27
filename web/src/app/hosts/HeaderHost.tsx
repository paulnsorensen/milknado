import type { ReactElement } from 'react';
import { GoalTitle } from './GoalTitle';
import { renderSlot } from './renderSlot';

/** The header row inside main: the mode kicker and goal title, then the controls. */
export function HeaderHost(): ReactElement {
  return (
    <header data-region="header">
      <GoalTitle />
      <div className="mk-header-controls">
        {renderSlot('status')}
        {renderSlot('header-control')}
      </div>
    </header>
  );
}