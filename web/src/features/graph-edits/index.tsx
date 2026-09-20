// The `rail-action` Add-node button, the `sidecar-action` Edit/Move/Archive
// buttons for the selected node, and their four `dialog` contributions.
import { registerSlot } from '../../app/slots';
import { AddNodeButton } from './AddNodeButton';
import { AddNodeDialog } from './AddNodeDialog';
import { ArchiveNodeDialog } from './ArchiveNodeDialog';
import { EditNodeDialog } from './EditNodeDialog';
import { MoveNodeDialog } from './MoveNodeDialog';
import { NodeActionButtons } from './NodeActionButtons';

export function register(): void {
  registerSlot('rail-action', () => <AddNodeButton />);
  registerSlot('sidecar-action', () => <NodeActionButtons />);
  registerSlot('dialog', () => <AddNodeDialog />);
  registerSlot('dialog', () => <EditNodeDialog />);
  registerSlot('dialog', () => <MoveNodeDialog />);
  registerSlot('dialog', () => <ArchiveNodeDialog />);
}
