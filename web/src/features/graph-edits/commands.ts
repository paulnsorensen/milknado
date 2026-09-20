// Command posts for the graph-edits feature, bound to the node_edits routes'
// msgspec bodies (`src/milknado/web/node_requests.py`) field for field.
import { patch, post } from '../../app/api';

export interface AddNodeInput {
  description: string;
  parent_id: number | null;
  flavor: string | null;
  files: string[] | null;
  prereqs: number[] | null;
}

export function addNode(input: AddNodeInput): Promise<unknown> {
  return post('/api/nodes', input);
}

export interface EditNodeInput {
  description?: string;
  flavor?: string | null;
}

export function editNode(nodeId: number, input: EditNodeInput): Promise<unknown> {
  return patch(`/api/nodes/${nodeId}`, input);
}

export function moveNode(nodeId: number, newParentId: number | null): Promise<unknown> {
  return post(`/api/nodes/${nodeId}/move`, { new_parent_id: newParentId });
}

export function archiveNode(nodeId: number): Promise<unknown> {
  return post(`/api/nodes/${nodeId}/archive`);
}
