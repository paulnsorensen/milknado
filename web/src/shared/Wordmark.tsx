// The Milknado wordmark: the mark image and product name, shown in the rail
// header and the narrow layout's header bar.
import type { ReactElement } from 'react';

export function Wordmark(): ReactElement {
  return (
    <div className="mk-wordmark">
      <img src="/assets/milknado-mark.png" alt="" />
      <span className="mk-text-wordmark">Milknado</span>
    </div>
  );
}
