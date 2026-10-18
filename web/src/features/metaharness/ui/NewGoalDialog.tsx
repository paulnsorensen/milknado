// The New goal dialog: a goal description and the coordinator's provider,
// sent as `start_goal`. It renders in the shell's `dialog` slot.
import type { ReactElement } from 'react';
import { useState, useSyncExternalStore } from 'react';
import { Milknado } from '../../../design-system';
import { Dialog } from '../../../shared/dialog/Dialog';
import { startGoal } from '../data/commands';
import { flash, getMh, patchMh, subscribeMh } from '../viewState';

const PROVIDERS = ['claude', 'codex'] as const;
type Provider = (typeof PROVIDERS)[number];

export function NewGoalDialog(): ReactElement | null {
  const open = useSyncExternalStore(subscribeMh, () => getMh().goalForm);
  const { Button } = Milknado;
  const [description, setDescription] = useState('');
  const [provider, setProvider] = useState<Provider>('claude');
  if (!open) {
    return null;
  }
  const close = (): void => patchMh({ goalForm: false });
  const submit = (): void => {
    const text = description.trim();
    close();
    setDescription('');
    void startGoal(text, provider).then((ok) => ok && flash('Goal started. It appears under Projects.'));
  };
  return (
    <Dialog title="Start a new goal" label="New goal" onClose={close} actions={
      <>
        <Button onClick={close}>Cancel</Button>
        <Button variant="primary" glyph="+" onClick={submit} disabled={description.trim() === ''}>Start goal</Button>
      </>
    }>
      <div className="mk-fields">
        <div className="mk-field">
          <label htmlFor="mh-goal-description">Goal</label>
          <textarea id="mh-goal-description" className="mk-input" rows={3} value={description}
            placeholder="What should be true when this goal is done?"
            onChange={(event) => setDescription(event.target.value)} />
        </div>
        <div className="mk-field">
          <div className="mk-kicker">Coordinator</div>
          <div className="mk-seg" role="group" aria-label="Coordinator">
            {PROVIDERS.map((option) => (
              <button key={option} type="button" className={provider === option ? 'mk-seg-opt is-on' : 'mk-seg-opt'}
                aria-pressed={provider === option} onClick={() => setProvider(option)}>{option}</button>
            ))}
          </div>
        </div>
      </div>
    </Dialog>
  );
}
