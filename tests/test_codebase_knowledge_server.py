"""Tests for mcp_servers.codebase_knowledge.server — FastMCP tool dispatch.

Tests tool registration count, round-trip dispatch for all 13 tools,
and error paths via direct KnowledgeStore + server function calls.
"""

import json

import pytest

from memory.memory_store import MemoryStore
from mcp_servers.codebase_knowledge.knowledge import KnowledgeStore
import mcp_servers.codebase_knowledge.server as srv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_knowledge(tmp_path):
    """Wire a fresh KnowledgeStore into the server module for each test."""
    srv._knowledge = KnowledgeStore(
        store=MemoryStore(store_path=str(tmp_path / "test.json"))
    )
    yield
    srv._knowledge = None


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    @pytest.mark.asyncio
    async def test_tool_count(self):
        """Server exposes exactly 13 tools."""
        tools = await srv.mcp.list_tools()
        assert len(tools) == 13

    @pytest.mark.asyncio
    async def test_tool_names(self):
        """All expected tool names are registered."""
        tools = await srv.mcp.list_tools()
        names = {t.name for t in tools}
        expected = {
            "store_decision", "query_decisions", "update_decision", "delete_decision",
            "store_pattern", "query_patterns", "update_pattern", "delete_pattern",
            "get_conventions", "update_conventions", "delete_convention",
            "search_knowledge", "knowledge_stats",
        }
        assert names == expected


# ---------------------------------------------------------------------------
# Decision round-trips
# ---------------------------------------------------------------------------

class TestDecisionTools:
    def test_store_and_query(self):
        result = srv.store_decision("Use Python", "For backend services")
        assert "use-python" in result

        query_result = srv.query_decisions("python backend")
        parsed = json.loads(query_result)
        assert parsed["total"] >= 1
        assert any("Python" in r["title"] for r in parsed["results"])

    def test_update_round_trip(self):
        srv.store_decision("Test Decision", "original")
        result = srv.update_decision("test-decision", content="updated")
        assert "updated" in result

        query_result = srv.query_decisions("updated")
        parsed = json.loads(query_result)
        assert parsed["results"][0]["content"] == "updated"

    def test_delete_round_trip(self):
        srv.store_decision("To Delete", "content")
        result = srv.delete_decision("to-delete")
        assert "deleted" in result

        query_result = srv.query_decisions("delete")
        parsed = json.loads(query_result)
        assert parsed["total"] == 0

    def test_delete_not_found(self):
        with pytest.raises(KeyError):
            srv.delete_decision("nonexistent")


# ---------------------------------------------------------------------------
# Pattern round-trips
# ---------------------------------------------------------------------------

class TestPatternTools:
    def test_store_and_query(self):
        result = srv.store_pattern("Singleton", "Single instance pattern")
        assert "singleton" in result

        query_result = srv.query_patterns("singleton")
        parsed = json.loads(query_result)
        assert parsed["total"] >= 1

    def test_update_round_trip(self):
        srv.store_pattern("Test Pattern", "original")
        result = srv.update_pattern("test-pattern", content="updated")
        assert "updated" in result

    def test_delete_round_trip(self):
        srv.store_pattern("To Delete", "content")
        result = srv.delete_pattern("to-delete")
        assert "deleted" in result


# ---------------------------------------------------------------------------
# Convention round-trips
# ---------------------------------------------------------------------------

class TestConventionTools:
    def test_update_and_get(self):
        result = srv.update_conventions("naming", "camelCase")
        assert "updated" in result

        get_result = srv.get_conventions("naming")
        parsed = json.loads(get_result)
        assert "naming" in parsed

    def test_delete_round_trip(self):
        srv.update_conventions("naming", "camelCase")
        result = srv.delete_convention("naming")
        assert "deleted" in result

        get_result = srv.get_conventions("naming")
        parsed = json.loads(get_result)
        assert parsed == {}

    def test_delete_not_found(self):
        with pytest.raises(KeyError):
            srv.delete_convention("nonexistent")


# ---------------------------------------------------------------------------
# Cross-namespace search
# ---------------------------------------------------------------------------

class TestSearchKnowledge:
    def test_dispatch(self):
        srv.store_decision("Caching strategy", "Use Redis for caching")
        srv.store_pattern("Cache-aside", "Cache-aside pattern")
        result = srv.search_knowledge("caching")
        parsed = json.loads(result)
        assert parsed["total"] >= 2


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestKnowledgeStats:
    def test_dispatch(self):
        srv.store_decision("D1", "content")
        srv.store_pattern("P1", "content")
        result = srv.knowledge_stats()
        parsed = json.loads(result)
        assert parsed["total_entries"] == 2
