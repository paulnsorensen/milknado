from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from easy_cheese_schemas import ArtifactRef, EvidenceKind, EvidenceRef, canonical_bytes


def prepare_artifact_directory(value: str | Path) -> Path:
    try:
        directory = Path(value)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory = directory.resolve(strict=True)
        if not directory.is_dir():
            raise ValueError("artifact directory is not a directory")
        directory.chmod(0o700)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"artifact directory is not usable: {value}") from exc
    return directory


def _persist_evidence(body: bytes, destination: Path) -> None:
    try:
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() != body:
                raise ValueError("content-addressed evidence has conflicting bytes")
            destination.chmod(0o600)
            return
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                fd = -1
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
            destination.chmod(0o600)
        finally:
            if fd >= 0:
                os.close(fd)
            temporary.unlink(missing_ok=True)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"could not persist evidence at {destination}") from exc


def node_evidence(
    node_id: int, status: str, result: str | None, artifact_directory: Path
) -> EvidenceRef:
    payload = {"node_id": node_id, "status": status, "result": result}
    body = canonical_bytes(payload)
    digest = f"sha256:{hashlib.sha256(body).hexdigest()}"
    fingerprint = digest.removeprefix("sha256:")
    destination = artifact_directory / f"{fingerprint}.json"
    evidence_id = f"milknado/node/{node_id}/outcome/{fingerprint}"
    artifact = ArtifactRef(
        artifact_id=evidence_id,
        role="node-outcome",
        uri=destination.as_uri(),
        digest=digest,
        size_bytes=len(body),
        media_type="application/json",
    )
    _persist_evidence(body, destination)
    return EvidenceRef(
        evidence_id=evidence_id,
        kind=EvidenceKind.RUNTIME,
        artifact=artifact,
    )
