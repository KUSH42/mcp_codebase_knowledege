"""Entry point for the Codebase Knowledge MCP server.

Usage:
    python -m mcp_servers.codebase_knowledge [--store path/to/store.json] [--config path/to/config.yaml]
"""

import argparse
from pathlib import Path

import yaml

from mcp_servers.codebase_knowledge.knowledge import KnowledgeConfig, KnowledgeStore


def _load_config(path: str) -> KnowledgeConfig:
    """Load KnowledgeConfig from YAML, falling back to defaults."""
    p = Path(path)
    if not p.exists():
        return KnowledgeConfig()
    with open(p) as f:
        raw = yaml.safe_load(f) or {}
    section = raw.get("codebase_knowledge", {})
    return KnowledgeConfig(**{
        k: v for k, v in section.items()
        if k in KnowledgeConfig.__dataclass_fields__
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Codebase Knowledge MCP Server")
    parser.add_argument(
        "--store",
        default="memory/store.json",
        help="Path to MemoryStore backing file (default: memory/store.json)",
    )
    parser.add_argument(
        "--config",
        default="config/codebase_knowledge.yaml",
        help="Path to config YAML (default: config/codebase_knowledge.yaml)",
    )
    args = parser.parse_args()

    import mcp_servers.codebase_knowledge.server as srv
    srv._knowledge = KnowledgeStore(store_path=args.store, config=_load_config(args.config))
    srv.mcp.run()


if __name__ == "__main__":
    main()
