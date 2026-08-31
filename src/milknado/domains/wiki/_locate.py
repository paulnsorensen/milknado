"""Roadmap location and confined filesystem primitives."""

from __future__ import annotations

import errno
import os
import stat
import uuid
from pathlib import Path

from milknado.domains.wiki._serialize import (
    compute_goal_ref,
    compute_roadmap_ref,
    load_frontmatter,
    validate_slug,
)


class RoadmapPathError(ValueError):
    """A roadmap path attempted to traverse outside its wiki root."""


def wiki_root(project_root: Path) -> Path:
    """Return the one canonical project-local wiki root."""
    return project_root / ".hallouminate" / "wiki"


def locate_roadmap_dir(root: Path, roadmap_ref: str) -> tuple[Path, str]:
    """Recover a confined roadmap directory by its deterministic wiki identity."""
    for index_path in _roadmap_indexes(root):
        slug = index_path.parent.name
        created = load_frontmatter(read_text(root, index_path)).get("created")
        if created is not None and compute_roadmap_ref(slug, str(created)) == roadmap_ref:
            return index_path.parent, slug
    raise FileNotFoundError(f"no roadmap dir under {root} matches wiki_ref {roadmap_ref}")


def goal_file_map(root: Path, roadmap_dir: Path, roadmap_slug: str) -> dict[str, Path]:
    """Map each confined goal file's deterministic wiki identity to its path."""
    mapping: dict[str, Path] = {}
    for path in roadmap_markdown_files(root, roadmap_dir):
        if path.name == "index.md":
            continue
        created = load_frontmatter(read_text(root, path)).get("created")
        if created is not None:
            mapping[compute_goal_ref(roadmap_slug, path.stem, str(created))] = path
    return mapping


def _roadmap_indexes(root: Path) -> list[Path]:
    indexes: list[Path] = []
    for container in (root / "roadmaps", root):
        try:
            directory_fd = _open_directory(root, container)
        except FileNotFoundError:
            continue
        try:
            for name in sorted(os.listdir(directory_fd)):
                if container == root and name == "roadmaps":
                    continue
                mode = os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                    continue
                candidate = container / name / "index.md"
                try:
                    _ = read_text(root, candidate)
                except FileNotFoundError:
                    continue
                indexes.append(candidate)
        finally:
            os.close(directory_fd)
    return indexes


def resolve_roadmap_dir(root: Path, slug: str) -> Path:
    """Return nested or legacy-flat roadmap directory, without following links."""
    _ = validate_slug(slug)
    candidates = (root / "roadmaps" / slug, root / slug)
    for candidate in candidates:
        try:
            _ = read_text(root, candidate / "index.md")
        except FileNotFoundError:
            continue
        return candidate
    tried = ", ".join(str(candidate / "index.md") for candidate in candidates)
    raise FileNotFoundError(f"roadmap not found; tried {tried}")


def read_text(root: Path, path: Path) -> str:
    """Read one UTF-8 file beneath root with no symlink traversal."""
    parent_fd, name = _open_parent(root, path)
    try:
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise RoadmapPathError(f"symlinked roadmap path rejected: {path}") from exc
            raise
        try:
            mode = os.fstat(fd).st_mode
            if not stat.S_ISREG(mode):
                raise RoadmapPathError(f"roadmap path is not a regular file: {path}")
            with os.fdopen(fd, encoding="utf-8") as handle:
                fd = -1
                return handle.read()
        finally:
            if fd >= 0:
                os.close(fd)
    finally:
        os.close(parent_fd)


def write_text_atomic(root: Path, path: Path, content: str) -> None:
    """Atomically replace one confined UTF-8 file without following links."""
    parent_fd, name = _open_parent(root, path)
    temp_name = f".{name}.{uuid.uuid4().hex}.tmp"
    try:
        try:
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            mode = 0o600
        else:
            if stat.S_ISLNK(current.st_mode):
                raise RoadmapPathError(f"symlinked roadmap path rejected: {path}")
            if not stat.S_ISREG(current.st_mode):
                raise RoadmapPathError(f"roadmap path is not a regular file: {path}")
            mode = stat.S_IMODE(current.st_mode)

        fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            mode,
            dir_fd=parent_fd,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                fd = -1
                _ = handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            if fd >= 0:
                os.close(fd)
    finally:
        try:
            os.unlink(temp_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)


def roadmap_markdown_files(root: Path, roadmap_dir: Path) -> list[Path]:
    """List markdown children of a confined roadmap directory."""
    directory_fd = _open_directory(root, roadmap_dir)
    try:
        return sorted(
            roadmap_dir / name for name in os.listdir(directory_fd) if name.endswith(".md")
        )
    finally:
        os.close(directory_fd)


def _open_parent(root: Path, path: Path) -> tuple[int, str]:
    parts = _relative_parts(root, path)
    if not parts:
        raise RoadmapPathError(f"roadmap path must name a file beneath {root}")
    parent_fd = _open_directory_parts(root, parts[:-1])
    return parent_fd, parts[-1]


def _open_directory(root: Path, path: Path) -> int:
    return _open_directory_parts(root, _relative_parts(root, path))


def _open_directory_parts(root: Path, parts: tuple[str, ...]) -> int:
    try:
        root_mode = os.lstat(root).st_mode
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(root_mode):
        raise RoadmapPathError(f"symlinked wiki root rejected: {root}")
    try:
        current = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise RoadmapPathError(f"symlinked wiki root rejected: {root}") from exc
        raise
    try:
        for part in parts:
            component = os.stat(part, dir_fd=current, follow_symlinks=False)
            if stat.S_ISLNK(component.st_mode):
                raise RoadmapPathError(f"symlinked roadmap directory rejected beneath {root}")
            try:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=current,
                )
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise RoadmapPathError(
                        f"symlinked roadmap directory rejected beneath {root}"
                    ) from exc
                raise
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _relative_parts(root: Path, path: Path) -> tuple[str, ...]:
    root_abs = Path(os.path.abspath(root))
    path_abs = Path(os.path.abspath(path))
    try:
        relative = path_abs.relative_to(root_abs)
    except ValueError as exc:
        raise RoadmapPathError(f"roadmap path escapes wiki root: {path}") from exc
    return relative.parts
