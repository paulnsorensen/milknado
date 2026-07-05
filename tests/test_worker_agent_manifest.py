from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_AGENT = REPO_ROOT / "plugins" / "milknado" / "agents" / "milknado-worker.md"

PLUGIN_MCP_PREFIX = "mcp__plugin_milknado_milknado__"


def _worker_tools() -> list[str]:
    """Parse the `tools:` frontmatter line of the milknado-worker agent def."""
    for line in WORKER_AGENT.read_text(encoding="utf-8").splitlines():
        if line.startswith("tools:"):
            return [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()]
    raise AssertionError(f"no `tools:` frontmatter line in {WORKER_AGENT}")


def test_worker_agent_uses_installed_plugin_mcp_namespace() -> None:
    """Native ultracode worker MCP allowlist comes ONLY from this agent def —
    node-runner passes no `tools` to `agent()`. Under the installed plugin the
    server is namespaced `mcp__plugin_milknado_milknado__`, so a bare
    `mcp__milknado__` name matches nothing: the worker silently loses
    milknado_deposit_result / node_verify, breaking the deposit contract.
    """
    tools = _worker_tools()
    milknado_tools = [t for t in tools if t.startswith("mcp__") and "milknado_" in t]
    assert milknado_tools, "worker allowlist must grant milknado MCP tools"
    bad = [t for t in milknado_tools if not t.startswith(PLUGIN_MCP_PREFIX)]
    assert not bad, (
        f"milknado tools must use the installed-plugin namespace "
        f"{PLUGIN_MCP_PREFIX!r}; bare mcp__milknado__ matches nothing under the "
        f"plugin so the native worker gets no MCP tools: {bad}"
    )
    joined = ",".join(tools)
    assert f"{PLUGIN_MCP_PREFIX}milknado_deposit_result" in joined
    assert f"{PLUGIN_MCP_PREFIX}milknado_node_verify" in joined
    assert f"{PLUGIN_MCP_PREFIX}milknado_track_follow_up" in joined
