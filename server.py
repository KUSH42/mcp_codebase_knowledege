"""CodebaseKnowledgeServer — MCP server for architectural knowledge.

FastMCP-based server providing 13 tools for CRUD and search over
decisions, patterns, and conventions.

No imports from control_plane or execution_plane.
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from mcp_servers.codebase_knowledge.knowledge import KnowledgeStore

mcp = FastMCP("codebase-knowledge")
_knowledge: KnowledgeStore | None = None


def _ks() -> KnowledgeStore:
    """Return the initialized KnowledgeStore singleton."""
    assert _knowledge is not None, "KnowledgeStore not initialized"
    return _knowledge


# ---------------------------------------------------------------------------
# Decision tools
# ---------------------------------------------------------------------------

@mcp.tool()
def store_decision(
    title: str, content: str, tags: list[str] | None = None,
) -> str:
    """Store an architectural decision with title, content, and optional tags.
    Returns the assigned key (may include a numeric suffix on collision)."""
    key = _ks().store_decision(title=title, content=content, tags=tags)
    return f"Decision stored with key: {key}"


@mcp.tool()
def query_decisions(
    query: str,
    tags: list[str] | None = None,
    limit: int = 5,
    offset: int = 0,
) -> str:
    """Query stored architectural decisions using natural language.
    Returns paginated results with _key field for use with update/delete."""
    results, total = _ks().query_decisions(
        query=query, tags=tags, limit=limit, offset=offset,
    )
    return json.dumps(
        {"results": results, "total": total, "offset": offset, "limit": limit},
        default=str,
    )


@mcp.tool()
def update_decision(
    key: str,
    content: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Update a decision's content and/or tags. Title is immutable."""
    _ks().update_decision(key=key, content=content, tags=tags)
    return f"Decision '{key}' updated"


@mcp.tool()
def delete_decision(key: str) -> str:
    """Delete a decision by its key (from _key field in query results)."""
    _ks().delete_decision(key)
    return f"Decision '{key}' deleted"


# ---------------------------------------------------------------------------
# Pattern tools
# ---------------------------------------------------------------------------

@mcp.tool()
def store_pattern(
    name: str, content: str, tags: list[str] | None = None,
) -> str:
    """Store a code pattern or convention.
    Returns the assigned key (may include a numeric suffix on collision)."""
    key = _ks().store_pattern(name=name, content=content, tags=tags)
    return f"Pattern stored with key: {key}"


@mcp.tool()
def query_patterns(
    query: str,
    tags: list[str] | None = None,
    limit: int = 5,
    offset: int = 0,
) -> str:
    """Query stored code patterns using natural language.
    Returns paginated results with _key field for use with update/delete."""
    results, total = _ks().query_patterns(
        query=query, tags=tags, limit=limit, offset=offset,
    )
    return json.dumps(
        {"results": results, "total": total, "offset": offset, "limit": limit},
        default=str,
    )


@mcp.tool()
def update_pattern(
    key: str,
    content: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Update a pattern's content and/or tags. Name is immutable."""
    _ks().update_pattern(key=key, content=content, tags=tags)
    return f"Pattern '{key}' updated"


@mcp.tool()
def delete_pattern(key: str) -> str:
    """Delete a pattern by its key (from _key field in query results)."""
    _ks().delete_pattern(key)
    return f"Pattern '{key}' deleted"


# ---------------------------------------------------------------------------
# Convention tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_conventions(category: str | None = None) -> str:
    """Get project conventions, optionally filtered by category."""
    result = _ks().get_conventions(category=category)
    return json.dumps(result, default=str)


@mcp.tool()
def update_conventions(category: str, content: str) -> str:
    """Update a project convention category."""
    _ks().update_conventions(category=category, content=content)
    return f"Convention '{category}' updated"


@mcp.tool()
def delete_convention(category: str) -> str:
    """Delete a convention by its category key."""
    _ks().delete_convention(category)
    return f"Convention '{category}' deleted"


# ---------------------------------------------------------------------------
# Cross-namespace search
# ---------------------------------------------------------------------------

@mcp.tool()
def search_knowledge(
    query: str,
    tags: list[str] | None = None,
    limit: int = 5,
    offset: int = 0,
) -> str:
    """Search across all knowledge namespaces (decisions, patterns, conventions).
    Returns paginated results with _key and _namespace fields."""
    results, total = _ks().search_all(
        query=query, tags=tags, limit=limit, offset=offset,
    )
    return json.dumps(
        {"results": results, "total": total, "offset": offset, "limit": limit},
        default=str,
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@mcp.tool()
def knowledge_stats() -> str:
    """Get corpus statistics: entry counts, top tags, date range, per-namespace breakdowns."""
    stats = _ks().get_stats()
    return json.dumps(stats, default=str)
