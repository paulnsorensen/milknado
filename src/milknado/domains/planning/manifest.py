from __future__ import annotations

import json
import re
from dataclasses import dataclass

from milknado.domains.graph import MikadoGraph


@dataclass(frozen=True)
class PlanAtom:
    id: str
    description: str
    depends_on: list[str]
    files: list[str]


@dataclass(frozen=True)
class PlanManifest:
    manifest_version: str
    atoms: list[PlanAtom]


def parse_manifest_from_output(output: str) -> PlanManifest | None:
    """Extract and validate a planning manifest JSON from agent output."""
    if not output.strip():
        return None

    candidates: list[str] = []
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", output, flags=re.DOTALL)
    candidates.extend(fenced)
    candidates.extend(_extract_json_objects(output))

    for blob in candidates:
        try:
            raw = json.loads(blob)
        except json.JSONDecodeError:
            continue
        manifest = _coerce_manifest(raw)
        if manifest is not None:
            return manifest
    return None


def apply_manifest_to_graph(graph: MikadoGraph, manifest: PlanManifest) -> list[int]:
    """Create graph nodes/edges from a parsed manifest. Returns created node IDs.

    For each edge stored as ``(parent_id, child_id)`` in ``MikadoGraph``, **child
    must complete before parent** (see ``get_ready_nodes`` / ``get_children``).
    So if atom **A** lists ``depends_on: [B]``, **B** is the prerequisite: we call
    ``add_edge(parent_id=A, child_id=B)``. Do not reverse this: ``parent_id`` in
    the edges table names the **dependent** node, not a decomposition-tree parent.
    """
    created: dict[str, int] = {}
    for atom in manifest.atoms:
        node = graph.add_node(atom.description)
        created[atom.id] = node.id
        if atom.files:
            graph.set_file_ownership(node.id, atom.files)

    for atom in manifest.atoms:
        dependent_id = created[atom.id]
        for prereq_slug in atom.depends_on:
            prerequisite_id = created.get(prereq_slug)
            if prerequisite_id is None:
                continue
            graph.add_edge(dependent_id, prerequisite_id)
    return list(created.values())


def _coerce_manifest(raw: object) -> PlanManifest | None:
    if not isinstance(raw, dict):
        return None
    version = raw.get("manifest_version")
    atoms_raw = raw.get("atoms")
    if not isinstance(version, str) or not isinstance(atoms_raw, list):
        return None
    atoms: list[PlanAtom] = []
    for atom in atoms_raw:
        if not isinstance(atom, dict):
            return None
        atom_id = atom.get("id")
        description = atom.get("description")
        depends_on = atom.get("depends_on", [])
        files = atom.get("files", [])
        if not isinstance(atom_id, str) or not isinstance(description, str):
            return None
        if not isinstance(depends_on, list) or not all(
            isinstance(item, str) for item in depends_on
        ):
            return None
        if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
            return None
        atoms.append(
            PlanAtom(
                id=atom_id,
                description=description.strip(),
                depends_on=depends_on,
                files=files,
            )
        )
    return PlanManifest(manifest_version=version, atoms=atoms)


def _extract_json_objects(text: str) -> list[str]:
    """Best-effort extraction for top-level JSON objects from freeform text."""
    blobs: list[str] = []
    depth = 0
    start = -1
    for idx, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                blobs.append(text[start : idx + 1])
                start = -1
    return blobs
