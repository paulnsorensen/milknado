// The diff takeover: every changed file in the run's worktree, with the
// node's pending decision pinned below.
import type { ReactElement } from 'react';
import type { RunDiff } from '../data/runDiff';
import { diffTotals } from '../data/runDiff';
import { shortRunId } from '../model/thread';
import { patchMh } from '../viewState';
import type { Current } from './current';
import { Icon } from './Icon';
import { NeedButtons } from './NeedButtons';
import type { MhModel } from './useMhModel';

interface DiffViewProps {
  diff: RunDiff;
  current: Current;
  summary: string;
  model: MhModel;
}

export function DiffView({ diff, current, summary, model }: DiffViewProps): ReactElement {
  const totals = diffTotals(diff.files);
  const where = current.node.branch_name ?? current.live?.session?.context?.cwd.split('/').at(-1);
  return (
    <div className="mh-diff">
      <div className="mh-diff-top">
        <button type="button" className="mh-pill-btn" onClick={() => patchMh({ view: 'thread' })}>
          <Icon name="back" size={14} />Back to chat</button>
        <span className="mh-muted">{diff.files.length} files</span>
        <span className="mh-mono mh-add">+{totals.added}</span>
        <span className="mh-mono mh-del">−{totals.removed}</span>
        <span className="mh-mono mh-faint mh-push">{where ?? `run ${shortRunId(diff.runId ?? '')}`}</span>
      </div>
      <div className="mh-diff-files">
        {diff.loading && <p className="mh-faint">Loading the diff…</p>}
        {diff.files.map((file) => (
          <div key={file.path} className="mh-file">
            <div className="mh-file-head"><Icon name="file" size={14} /><span>{file.path}</span>
              <span className="mh-add mh-push">+{file.added}</span><span className="mh-del">−{file.removed}</span></div>
            <div className="mh-file-body">
              {file.lines.map((line, i) => (
                <div key={i} className={`mh-line is-${line.mark === '+' ? 'add' : line.mark === '-' ? 'del' : line.mark === '@' ? 'hunk' : 'ctx'}`}>
                  <span className="mh-line-mark">{line.mark === '@' || line.mark === ' ' ? '' : line.mark}</span>
                  <span className="mh-line-text">{line.text}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      {current.need && (
        <div className="mh-verdict">
          <span className="mh-verdict-note">{summary}</span>
          <NeedButtons need={current.need} model={model} />
        </div>
      )}
    </div>
  );
}
