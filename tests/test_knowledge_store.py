"""Tests for mcp_servers.codebase_knowledge.knowledge — KnowledgeStore.

Covers: BM25 search, stemmer, stopwords, CRUD, validation, pagination,
cross-namespace search, tag normalization, stats, golden ranking tests.
"""

import time

import pytest

from mcp_servers.codebase_knowledge.knowledge import (
    KnowledgeConfig,
    KnowledgeStore,
    _normalize_tags,
    _slugify,
    _stem,
    _tokenize,
)
from memory.memory_store import MemoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    return KnowledgeStore(store=MemoryStore(store_path=str(tmp_path / "test.json")))


@pytest.fixture()
def custom_config_store(tmp_path):
    """Store with tight validation limits for testing."""
    config = KnowledgeConfig(
        max_content_length=100,
        max_title_length=20,
        max_tag_length=10,
        max_tags_per_entry=3,
        max_query_length=50,
        max_limit=10,
    )
    return KnowledgeStore(
        store=MemoryStore(store_path=str(tmp_path / "test.json")),
        config=config,
    )


# ---------------------------------------------------------------------------
# Step 0: Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_config(self):
        c = KnowledgeConfig()
        assert c.max_content_length == 50_000
        assert c.bm25_k1 == 1.5
        assert c.bm25_b == 0.75
        assert c.recency_window_days == 30
        assert c.recency_factor == 0.15

    def test_custom_config(self):
        c = KnowledgeConfig(max_content_length=100, bm25_k1=2.0)
        assert c.max_content_length == 100
        assert c.bm25_k1 == 2.0

    def test_config_loading_from_yaml(self, tmp_path):
        from mcp_servers.codebase_knowledge.__main__ import _load_config
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(
            "codebase_knowledge:\n"
            "  max_content_length: 1000\n"
            "  bm25_k1: 2.0\n"
            "  unknown_field: ignored\n"
        )
        config = _load_config(str(yaml_path))
        assert config.max_content_length == 1000
        assert config.bm25_k1 == 2.0
        assert config.bm25_b == 0.75  # default preserved

    def test_config_loading_missing_file(self, tmp_path):
        config = _load_config = None
        from mcp_servers.codebase_knowledge.__main__ import _load_config
        config = _load_config(str(tmp_path / "nonexistent.yaml"))
        assert config.max_content_length == 50_000  # all defaults


# ---------------------------------------------------------------------------
# Stemmer truth table
# ---------------------------------------------------------------------------

class TestStemmer:
    @pytest.mark.parametrize("word,expected", [
        ("indexing", "index"),
        ("indexed", "index"),
        ("caching", "cach"),
        ("cached", "cach"),
        ("cache", "cach"),
        ("running", "run"),
        ("stopped", "stop"),
        ("planned", "plan"),
        ("create", "creat"),
        ("created", "creat"),
        ("creating", "creat"),
        ("database", "databas"),
        ("databases", "databas"),
        ("service", "servic"),
        ("services", "servic"),
        ("patterns", "pattern"),
        ("pattern", "pattern"),
        ("names", "name"),
        ("classes", "class"),
        ("class", "class"),
        ("processes", "process"),
        ("process", "process"),
        ("processed", "process"),
        ("added", "add"),
        ("node", "node"),
        ("api", "api"),
        ("type", "type"),
        ("types", "type"),
        ("configurable", "configur"),
        ("configure", "configur"),
        ("extensible", "extens"),
        ("handler", "handl"),
        ("handle", "handl"),
        ("handling", "handl"),
        ("container", "contain"),
        ("contain", "contain"),
        ("scalable", "scal"),
        ("scaling", "scal"),
    ])
    def test_stem_truth_table(self, word, expected):
        assert _stem(word) == expected


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_tokenize_removes_stopwords(self):
        tokens = _tokenize("the quick brown fox is in the box")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "in" not in tokens

    def test_tokenize_stems(self):
        tokens = _tokenize("running databases")
        assert "run" in tokens
        assert "databas" in tokens

    def test_tokenize_basic(self):
        tokens = _tokenize("Hello World 123")
        assert "hello" in tokens
        assert "world" in tokens
        assert "123" in tokens

    def test_tokenize_empty(self):
        assert _tokenize("") == []

    def test_tokenize_only_stopwords(self):
        assert _tokenize("the and is") == []


# ---------------------------------------------------------------------------
# Tag normalization
# ---------------------------------------------------------------------------

class TestTagNormalization:
    def test_normalize_case(self):
        assert _normalize_tags(["Database", "API"]) == ["database", "api"]

    def test_normalize_whitespace(self):
        assert _normalize_tags([" database ", "  api  "]) == ["database", "api"]

    def test_normalize_dedup(self):
        assert _normalize_tags(["Database", "database", "DATABASE"]) == ["database"]

    def test_normalize_empty(self):
        assert _normalize_tags(None) == []
        assert _normalize_tags([]) == []

    def test_normalize_preserves_order(self):
        assert _normalize_tags(["Beta", "Alpha"]) == ["beta", "alpha"]


# ---------------------------------------------------------------------------
# Slugify
# ---------------------------------------------------------------------------

class TestSlugify:
    def test_basic(self):
        assert _slugify("My Decision Title!") == "my-decision-title"

    def test_max_length(self):
        assert len(_slugify("a" * 200)) <= 80

    def test_collision_same_slug(self):
        assert _slugify("Use PostgreSQL") == _slugify("Use PostgreSQL!")


# ---------------------------------------------------------------------------
# Step 2: Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_content_too_long(self, custom_config_store):
        with pytest.raises(ValueError, match="Content exceeds"):
            custom_config_store.store_decision("Title", "x" * 101)

    def test_title_too_long(self, custom_config_store):
        with pytest.raises(ValueError, match="Title/name exceeds"):
            custom_config_store.store_decision("x" * 21, "content")

    def test_too_many_tags(self, custom_config_store):
        with pytest.raises(ValueError, match="Too many tags"):
            custom_config_store.store_decision("Title", "content", tags=["a", "b", "c", "d"])

    def test_tag_too_long(self, custom_config_store):
        with pytest.raises(ValueError, match="exceeds"):
            custom_config_store.store_decision("Title", "content", tags=["x" * 11])

    def test_query_too_long(self, custom_config_store):
        custom_config_store.store_decision("Title", "content")
        with pytest.raises(ValueError, match="Query exceeds"):
            custom_config_store.query_decisions("x" * 51)

    def test_invalid_limit_zero(self, store):
        store.store_decision("Title", "content")
        with pytest.raises(ValueError, match="Limit must be"):
            store.query_decisions("test", limit=0)

    def test_invalid_limit_too_high(self, store):
        store.store_decision("Title", "content")
        with pytest.raises(ValueError, match="Limit must be"):
            store.query_decisions("test", limit=101)

    def test_negative_offset(self, store):
        store.store_decision("Title", "content")
        with pytest.raises(ValueError, match="Offset must be"):
            store.query_decisions("test", offset=-1)

    def test_boundary_values_pass(self, store):
        store.store_decision("Title", "content")
        results, _ = store.query_decisions("test", limit=1, offset=0)
        assert isinstance(results, list)

    def test_convention_content_validation(self, custom_config_store):
        with pytest.raises(ValueError, match="Content exceeds"):
            custom_config_store.update_conventions("cat", "x" * 101)


# ---------------------------------------------------------------------------
# Step 1: BM25 search & ranking
# ---------------------------------------------------------------------------

class TestBM25Search:
    def test_basic_search(self, store):
        store.store_decision("Use PostgreSQL", "Chosen for ACID compliance", tags=["database"])
        results, total = store.query_decisions("postgresql database")
        assert total >= 1
        assert results[0]["title"] == "Use PostgreSQL"

    def test_key_in_results(self, store):
        key = store.store_decision("Use PostgreSQL", "For storage")
        results, _ = store.query_decisions("postgresql")
        assert results[0]["_key"] == key

    def test_results_are_copies(self, store):
        """Mutating query results must not mutate MemoryStore internals."""
        store.store_decision("Test", "content")
        results, _ = store.query_decisions("test")
        results[0]["extra"] = "injected"
        # Re-query — original should not have the injected field.
        results2, _ = store.query_decisions("test")
        assert "extra" not in results2[0]

    def test_empty_corpus(self, store):
        results, total = store.query_decisions("anything")
        assert results == []
        assert total == 0

    def test_empty_query_tokens(self, store):
        """Query with only stopwords returns all entries."""
        store.store_decision("Title", "content")
        results, total = store.query_decisions("the and is")
        assert total == 1

    def test_tag_filter(self, store):
        store.store_decision("Use Redis", "For caching", tags=["cache"])
        store.store_decision("Use PostgreSQL", "For storage", tags=["database"])
        results, total = store.query_decisions("storage", tags=["database"])
        assert all("database" in r.get("tags", []) for r in results)

    def test_query_side_tag_normalization(self, store):
        """Tags=["DATABASE"] should match stored tags=["database"]."""
        store.store_decision("Use PostgreSQL", "For storage", tags=["database"])
        results, total = store.query_decisions("storage", tags=["DATABASE"])
        assert total == 1

    def test_limit(self, store):
        for i in range(10):
            store.store_decision(f"Decision {i}", f"Content about topic {i}")
        results, total = store.query_decisions("topic", limit=3)
        assert len(results) <= 3
        assert total == 10

    def test_pattern_search(self, store):
        store.store_pattern("Repository Pattern", "Abstract data access", tags=["architecture"])
        results, total = store.query_patterns("repository data access")
        assert total >= 1
        assert results[0]["name"] == "Repository Pattern"

    def test_pattern_tag_filter(self, store):
        store.store_pattern("Singleton", "Single instance pattern", tags=["creational"])
        store.store_pattern("Observer", "Event subscription pattern", tags=["behavioral"])
        results, _ = store.query_patterns("pattern", tags=["creational"])
        assert all("creational" in r.get("tags", []) for r in results)


# ---------------------------------------------------------------------------
# Golden ranking tests (G1-G5)
# ---------------------------------------------------------------------------

class TestGoldenRanking:
    def test_g1_term_relevance(self, store):
        """REST query finds REST document first."""
        store.store_decision("Use REST API", "REST API for web services")
        store.store_decision("Add database indexes", "Add indexes for performance")
        results, _ = store.query_decisions("REST API web")
        assert results[0]["title"] == "Use REST API"

    def test_g2_stemmer_cross_inflection(self, store):
        """'database scaling' finds 'databases' via stemmer."""
        store.store_decision("PostgreSQL scaling", "PostgreSQL handles our databases efficiently")
        store.store_decision("Use Redis", "Use Redis for caching")
        results, _ = store.query_decisions("database scaling")
        assert results[0]["title"] == "PostgreSQL scaling"

    def test_g3_stemmer_suffix_recall(self, store):
        """'authentication' matches 'Authentication pattern' via stem 'authentic'."""
        store.store_pattern("Authentication pattern", "JWT authentication flow")
        store.store_pattern("Logging pattern", "Structured logging with correlation IDs")
        # Add filler entries
        for i in range(8):
            store.store_pattern(f"Filler pattern {i}", f"Filler content about topic {i}")
        results, _ = store.query_patterns("authentication")
        assert results[0]["name"] == "Authentication pattern"

    def test_g4_recency_boost(self, store):
        """Newer entry ranks above older with equal text relevance."""
        key1 = store.store_decision("Old Decision", "About authentication security")
        # Backdate the first entry
        entry = store._store.read("kb_decisions", key1)
        entry["created_at"] = time.time() - 25 * 86400  # 25 days old
        store._store.write("kb_decisions", key1, entry)

        key2 = store.store_decision("New Decision", "About authentication security")
        # This one is fresh (just created)

        results, _ = store.query_decisions("authentication security")
        assert len(results) == 2
        assert results[0]["_key"] == key2  # newer first

    def test_g5_length_normalization(self, store):
        """Short focused doc outranks long doc with incidental mention."""
        store.store_decision("REST API design", "REST API design principles and best practices")
        long_content = " ".join(f"topic{i} discussion" for i in range(100)) + " REST mentioned once"
        store.store_decision("Long document", long_content)
        results, _ = store.query_decisions("REST API")
        assert results[0]["title"] == "REST API design"


# ---------------------------------------------------------------------------
# Step 3: CRUD lifecycle
# ---------------------------------------------------------------------------

class TestDeleteDecision:
    def test_delete_found(self, store):
        key = store.store_decision("To Delete", "content")
        store.delete_decision(key)
        results, total = store.query_decisions("delete")
        assert total == 0

    def test_delete_not_found(self, store):
        with pytest.raises(KeyError, match="No decision found"):
            store.delete_decision("nonexistent")


class TestDeletePattern:
    def test_delete_found(self, store):
        key = store.store_pattern("To Delete", "content")
        store.delete_pattern(key)
        results, total = store.query_patterns("delete")
        assert total == 0

    def test_delete_not_found(self, store):
        with pytest.raises(KeyError, match="No pattern found"):
            store.delete_pattern("nonexistent")


class TestDeleteConvention:
    def test_delete_found(self, store):
        store.update_conventions("naming", "camelCase")
        store.delete_convention("naming")
        result = store.get_conventions("naming")
        assert result == {}

    def test_delete_not_found(self, store):
        with pytest.raises(KeyError, match="No convention found"):
            store.delete_convention("nonexistent")


class TestUpdateDecision:
    def test_update_content(self, store):
        key = store.store_decision("Test", "old content")
        store.update_decision(key, content="new content")
        results, _ = store.query_decisions("new content")
        assert results[0]["content"] == "new content"

    def test_update_tags(self, store):
        key = store.store_decision("Test", "content", tags=["old"])
        store.update_decision(key, tags=["new"])
        results, _ = store.query_decisions("content")
        assert results[0]["tags"] == ["new"]

    def test_update_not_found(self, store):
        with pytest.raises(KeyError, match="No decision found"):
            store.update_decision("nonexistent", content="x")

    def test_update_preserves_created_at(self, store):
        key = store.store_decision("Test", "content")
        original = store._store.read("kb_decisions", key)
        created_at = original["created_at"]
        store.update_decision(key, content="updated")
        updated = store._store.read("kb_decisions", key)
        assert updated["created_at"] == created_at
        assert "updated_at" in updated

    def test_update_does_not_mutate_store(self, store):
        """Copy-before-mutate: updating should not corrupt MemoryStore."""
        key = store.store_decision("Test", "content", tags=["orig"])
        store.update_decision(key, tags=["changed"])
        # The stored entry should have the new tags
        entry = store._store.read("kb_decisions", key)
        assert entry["tags"] == ["changed"]

    def test_update_with_validation_error(self, custom_config_store):
        key = custom_config_store.store_decision("Test", "content")
        with pytest.raises(ValueError, match="Content exceeds"):
            custom_config_store.update_decision(key, content="x" * 101)


class TestUpdatePattern:
    def test_update_content(self, store):
        key = store.store_pattern("Test", "old content")
        store.update_pattern(key, content="new content")
        results, _ = store.query_patterns("new content")
        assert results[0]["content"] == "new content"

    def test_update_not_found(self, store):
        with pytest.raises(KeyError, match="No pattern found"):
            store.update_pattern("nonexistent", content="x")


class TestSlugCollision:
    def test_auto_suffix(self, store):
        key1 = store.store_decision("Use PostgreSQL", "First")
        key2 = store.store_decision("Use PostgreSQL", "Second")
        assert key1 == "use-postgresql"
        assert key2 == "use-postgresql-2"

    def test_auto_suffix_multiple(self, store):
        store.store_decision("Use PostgreSQL", "First")
        store.store_decision("Use PostgreSQL", "Second")
        key3 = store.store_decision("Use PostgreSQL", "Third")
        assert key3 == "use-postgresql-3"

    def test_store_returns_actual_key(self, store):
        key = store.store_decision("Use PostgreSQL", "First")
        assert key == "use-postgresql"


# ---------------------------------------------------------------------------
# Step 4: Cross-namespace search, pagination, stats
# ---------------------------------------------------------------------------

class TestCrossNamespaceSearch:
    def test_g6_merged_results(self, store):
        """search_knowledge returns mixed results with _namespace labels."""
        store.store_decision("Use caching", "Caching strategy decision")
        store.store_pattern("Cache-aside pattern", "Cache-aside implementation")
        store.update_conventions("caching", "Caching rules and guidelines")
        results, total = store.search_all("caching")
        assert total == 3
        namespaces = {r["_namespace"] for r in results}
        assert namespaces == {"decision", "pattern", "convention"}

    def test_namespace_labels(self, store):
        store.store_decision("Test decision", "content")
        results, _ = store.search_all("test decision")
        assert results[0]["_namespace"] == "decision"

    def test_cross_namespace_empty(self, store):
        results, total = store.search_all("anything")
        assert results == []
        assert total == 0


class TestPagination:
    def test_offset_limit(self, store):
        for i in range(15):
            store.store_decision(f"Decision topic{i}", f"Content about topic{i} relevant")
        page1, total1 = store.query_decisions("topic", limit=5, offset=0)
        page2, total2 = store.query_decisions("topic", limit=5, offset=5)
        page3, total3 = store.query_decisions("topic", limit=5, offset=10)
        assert total1 == total2 == total3 == 15
        assert len(page1) == 5
        assert len(page2) == 5
        assert len(page3) == 5
        # Non-overlapping results
        keys1 = {r["_key"] for r in page1}
        keys2 = {r["_key"] for r in page2}
        keys3 = {r["_key"] for r in page3}
        assert not (keys1 & keys2)
        assert not (keys2 & keys3)

    def test_offset_beyond_results(self, store):
        store.store_decision("Test", "content")
        results, total = store.query_decisions("test", offset=100)
        assert results == []
        assert total == 1


class TestStats:
    def test_counts(self, store):
        store.store_decision("D1", "content")
        store.store_decision("D2", "content")
        store.store_pattern("P1", "content")
        store.update_conventions("C1", "content")
        stats = store.get_stats()
        assert stats["total_entries"] == 4
        assert stats["namespaces"]["decisions"]["count"] == 2
        assert stats["namespaces"]["patterns"]["count"] == 1
        assert stats["namespaces"]["conventions"]["count"] == 1

    def test_top_tags(self, store):
        store.store_decision("D1", "content", tags=["database", "api"])
        store.store_decision("D2", "content", tags=["database"])
        stats = store.get_stats()
        tags_dict = dict(stats["top_tags"])
        assert tags_dict["database"] == 2
        assert tags_dict["api"] == 1

    def test_empty_corpus(self, store):
        stats = store.get_stats()
        assert stats["total_entries"] == 0
        assert stats["oldest_entry"] is None
        assert stats["newest_entry"] is None

    def test_date_range(self, store):
        store.store_decision("D1", "content")
        stats = store.get_stats()
        assert stats["oldest_entry"] is not None
        assert stats["newest_entry"] is not None


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------

class TestConventions:
    def test_update_and_get_convention(self, store):
        store.update_conventions("naming", "Use camelCase for variables")
        result = store.get_conventions("naming")
        assert "naming" in result
        assert result["naming"]["content"] == "Use camelCase for variables"

    def test_get_all_conventions(self, store):
        store.update_conventions("naming", "camelCase")
        store.update_conventions("testing", "Jest + pytest")
        result = store.get_conventions()
        assert len(result) >= 2

    def test_get_nonexistent_convention(self, store):
        result = store.get_conventions("nonexistent")
        assert result == {}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_data_survives_new_instance(self, tmp_path):
        path = str(tmp_path / "persist.json")
        store1 = KnowledgeStore(store=MemoryStore(store_path=path))
        store1.store_decision("Persist Test", "Should survive reload")

        store2 = KnowledgeStore(store=MemoryStore(store_path=path))
        results, total = store2.query_decisions("persist")
        assert total == 1
        assert results[0]["title"] == "Persist Test"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_crud_cycle(self, store):
        """store -> query (verify _key) -> update -> query -> delete -> query."""
        key = store.store_decision("CRUD Test", "original content", tags=["test"])
        results, _ = store.query_decisions("CRUD")
        assert results[0]["_key"] == key

        store.update_decision(key, content="updated content")
        results, _ = store.query_decisions("updated")
        assert results[0]["content"] == "updated content"

        store.delete_decision(key)
        results, total = store.query_decisions("CRUD")
        assert total == 0

    def test_collision_cycle(self, store):
        key1 = store.store_decision("Use PostgreSQL", "First")
        key2 = store.store_decision("Use PostgreSQL", "Second")
        assert key2 == "use-postgresql-2"
        # Both exist
        r1 = store._store.read("kb_decisions", key1)
        r2 = store._store.read("kb_decisions", key2)
        assert r1 is not None
        assert r2 is not None

    def test_tag_normalization_round_trip(self, store):
        store.store_decision("Test", "content", tags=["Database", " database ", "DB"])
        results, total = store.query_decisions("content", tags=["database"])
        assert total == 1
        # Tags stored normalized and deduped
        assert results[0]["tags"] == ["database", "db"]

    def test_copy_safety(self, store):
        """Mutating query results should not affect stored data."""
        store.store_decision("Test", "content")
        results, _ = store.query_decisions("test")
        results[0]["extra"] = "injected"
        results2, _ = store.query_decisions("test")
        assert "extra" not in results2[0]
