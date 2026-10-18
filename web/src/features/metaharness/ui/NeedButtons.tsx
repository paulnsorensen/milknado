import type { ReactElement } from 'react';
import type { Need } from '../model/needs';
import { resolveNeed } from './actions';
import { NEED_ACTIONS } from './current';
import type { MhModel } from './useMhModel';

export function NeedButtons({ need, model }: { need: Need; model: MhModel }): ReactElement {
  return (
    <div className="mh-actions">
      {NEED_ACTIONS[need.kind].map((spec) => (
        <button key={spec.action} type="button" className={`mh-btn is-${spec.tone}`}
          onClick={() => void resolveNeed(need, spec.action, model)}>{spec.label}</button>
      ))}
    </div>
  );
}
