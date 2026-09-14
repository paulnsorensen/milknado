import subprocess
from pathlib import Path

from milknado.domains.graph import MikadoGraph


def _git(repo: Path, *args: str) -> None:
    _ = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "feature")
    _git(repo, "config", "user.email", "test@milknado.test")
    _git(repo, "config", "user.name", "Milknado Test")
    _ = (repo / "README.md").write_text("# session test\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "seed")
    return repo


def build_graph(repo: Path) -> MikadoGraph:
    db_path = repo / ".milknado" / "graph.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    graph = MikadoGraph(db_path)
    root = graph.add_node("Interactive session goal")
    _ = graph.add_node("Accept human guidance", parent_id=root.id)
    return graph
