// The `dialog` slot contribution for adding a node: description, parent,
// flavor, prerequisites and files, posted to `POST /api/nodes`.
import type { ReactElement } from 'react';
import { useState, useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { Dialog } from '../../shared/dialog/Dialog';
import { addNode } from './commands';
import { closeDialog, getDialogState, subscribeDialog } from './dialogState';

function parseNumberList(value: string): number[] {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter((item) => item.length > 0)
    .map(Number)
    .filter((item) => !Number.isNaN(item));
}

function parseLines(value: string): string[] {
  return value
    .split('\n')
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
}

export function AddNodeDialog(): ReactElement | null {
  const dialog = useSyncExternalStore(subscribeDialog, getDialogState);
  const store = useSyncExternalStore(subscribe, getState);
  const { Button } = Milknado;
  const [description, setDescription] = useState('');
  const [parentId, setParentId] = useState('');
  const [flavor, setFlavor] = useState('');
  const [prereqs, setPrereqs] = useState('');
  const [files, setFiles] = useState('');

  if (dialog.kind !== 'add') {
    return null;
  }

  const nodes = store.snapshot?.graph?.nodes ?? [];

  function submit(): void {
    void addNode({
      description,
      parent_id: parentId === '' ? null : Number(parentId),
      flavor: flavor.trim() === '' ? null : flavor,
      files: files.trim() === '' ? null : parseLines(files),
      prereqs: prereqs.trim() === '' ? null : parseNumberList(prereqs),
    });
    setDescription('');
    setParentId('');
    setFlavor('');
    setPrereqs('');
    setFiles('');
    closeDialog();
  }

  return (
    <Dialog
      title="Add a node"
      label="Add node"
      actions={
        <>
          <Button onClick={closeDialog}>Cancel</Button>
          <Button variant="primary" glyph="+" onClick={submit} disabled={description.trim() === ''}>
            Add node
          </Button>
        </>
      }
    >
      <div className="mk-fields">
        <div className="mk-field">
          <label htmlFor="mk-add-description">Description</label>
          <textarea
            id="mk-add-description"
            className="mk-input"
            rows={3}
            value={description}
            onChange={(event) => setDescription(event.target.value)}
          />
        </div>
        <div className="mk-field">
          <label htmlFor="mk-add-parent">Parent</label>
          <select
            id="mk-add-parent"
            className="mk-input"
            value={parentId}
            onChange={(event) => setParentId(event.target.value)}
          >
            <option value="">None</option>
            {nodes.map((node) => (
              <option key={node.id} value={node.id}>
                {node.id} {'·'} {node.description}
              </option>
            ))}
          </select>
        </div>
        <div className="mk-field">
          <label htmlFor="mk-add-flavor">Flavor</label>
          <input
            id="mk-add-flavor"
            className="mk-input"
            placeholder="implement, spec, spike, prototype, research"
            value={flavor}
            onChange={(event) => setFlavor(event.target.value)}
          />
        </div>
        <div className="mk-field">
          <label htmlFor="mk-add-prereqs">Prerequisites</label>
          <input
            id="mk-add-prereqs"
            className="mk-input"
            placeholder="Node ids, comma separated"
            value={prereqs}
            onChange={(event) => setPrereqs(event.target.value)}
          />
        </div>
        <div className="mk-field">
          <label htmlFor="mk-add-files">Files</label>
          <textarea
            id="mk-add-files"
            className="mk-input"
            rows={2}
            placeholder="One path per line"
            style={{ fontFamily: 'var(--mk-font-mono)' }}
            value={files}
            onChange={(event) => setFiles(event.target.value)}
          />
        </div>
      </div>
    </Dialog>
  );
}
