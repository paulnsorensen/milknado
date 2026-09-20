// The `dialog` slot contribution for adding a node: description, parent,
// flavor, prerequisites and files, posted to `POST /api/nodes`.
import type { ReactElement } from 'react';
import { useState, useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
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
    <div role="dialog" aria-label="Add node" className="mk-add-node-dialog">
      <label>
        Description
        <input
          aria-label="Description"
          value={description}
          onChange={(event) => setDescription(event.target.value)}
        />
      </label>
      <label>
        Parent
        <select
          aria-label="Parent"
          value={parentId}
          onChange={(event) => setParentId(event.target.value)}
        >
          <option value="">None</option>
          {nodes.map((node) => (
            <option key={node.id} value={node.id}>
              {node.description}
            </option>
          ))}
        </select>
      </label>
      <label>
        Flavor
        <input aria-label="Flavor" value={flavor} onChange={(event) => setFlavor(event.target.value)} />
      </label>
      <label>
        Prerequisites
        <input
          aria-label="Prerequisites"
          value={prereqs}
          onChange={(event) => setPrereqs(event.target.value)}
        />
      </label>
      <label>
        Files
        <textarea aria-label="Files" value={files} onChange={(event) => setFiles(event.target.value)} />
      </label>
      <Button onClick={submit} disabled={description.trim() === ''}>
        Add node
      </Button>
      <Button onClick={closeDialog}>Cancel</Button>
    </div>
  );
}
