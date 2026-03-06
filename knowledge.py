"""KnowledgeStore — BM25 search over decisions, patterns, and conventions.

Backed by MemoryStore with three namespaces:
- kb_decisions: architectural decisions
- kb_patterns:  code patterns and conventions
- kb_conventions: project-wide conventions

No imports from control_plane or execution_plane.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any

from memory.memory_store import MemoryStore

_NS_DECISIONS = "kb_decisions"
_NS_PATTERNS = "kb_patterns"
_NS_CONVENTIONS = "kb_conventions"

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "my", "no", "not", "of",
    "on", "or", "our", "she", "so", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "up", "was", "we",
    "what", "when", "which", "who", "will", "with", "you", "your",
})

# ---------------------------------------------------------------------------
# Stemmer
# ---------------------------------------------------------------------------

_INFLECTION_DOUBLES: frozenset[str] = frozenset("bdglmnprt")


def _stem(word: str) -> str:
    """Minimal suffix-stripping stemmer. No external dependencies."""
    if len(word) <= 3:
        return word
    # Order matters — longest suffixes first; -e before -es/-s.
    # See spec "Suffix ordering rationale". Do not rearrange.
    for suffix, min_stem in [
        ("ation", 3), ("tion", 3), ("sion", 3),
        ("ness", 3), ("ment", 3),
        ("able", 3), ("ible", 3), ("er", 3),
        ("ing", 3), ("ies", 3), ("ed", 3), ("ly", 3),
        ("e", 4), ("es", 4), ("s", 3),
    ]:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_stem:
            stem = word[: -len(suffix)]
            # Guard: don't strip -s from words ending in -ss
            # (class, process, address — not plurals).
            if suffix == "s" and stem.endswith("s"):
                continue
            # De-duplicate inflection-caused doubled consonant.
            if (
                suffix in ("ing", "ed")
                and len(stem) >= 2
                and stem[-1] == stem[-2]
                and stem[-1] in _INFLECTION_DOUBLES
            ):
                candidate = stem[:-1]
                if len(candidate) >= 3:
                    stem = candidate
            return stem
    return word


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Tokenize, stem, and remove stopwords."""
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in raw if w not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Tag normalization
# ---------------------------------------------------------------------------

def _normalize_tags(tags: list[str] | None) -> list[str]:
    """Lowercase and strip tags. Deduplicate preserving order."""
    if not tags:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        normalized = tag.lower().strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


# ---------------------------------------------------------------------------
# Slugify
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug key."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80]


# ---------------------------------------------------------------------------
# BM25 scorer
# ---------------------------------------------------------------------------

def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    idf_map: dict[str, float],
    avgdl: float,
    k1: float,
    b: float,
) -> float:
    """BM25 score for a single document (Lucene/BM25+ IDF variant)."""
    dl = len(doc_tokens)
    freq: dict[str, int] = {}
    for t in doc_tokens:
        freq[t] = freq.get(t, 0) + 1

    score = 0.0
    for term in set(query_tokens):
        tf = freq.get(term, 0)
        idf = idf_map.get(term, 0.0)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avgdl)
        score += idf * numerator / denominator
    return score


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeConfig:
    max_content_length: int = 50_000
    max_title_length: int = 200
    max_tag_length: int = 50
    max_tags_per_entry: int = 20
    max_query_length: int = 500
    max_limit: int = 100
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    recency_window_days: int = 30
    recency_factor: float = 0.15


# ---------------------------------------------------------------------------
# KnowledgeStore
# ---------------------------------------------------------------------------

class KnowledgeStore:
    """Search and store architectural knowledge.

    Args:
        store:      MemoryStore instance for persistence.
        store_path: Path to MemoryStore backing file (used when store is None).
        config:     KnowledgeConfig for validation limits and BM25 tuning.
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        store_path: str = "memory/store.json",
        config: KnowledgeConfig | None = None,
    ) -> None:
        self._store = store if store is not None else MemoryStore(store_path=store_path)
        self._config = config or KnowledgeConfig()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_store_input(self, title_or_name: str, content: str,
                              tags: list[str] | None) -> None:
        if len(title_or_name) > self._config.max_title_length:
            raise ValueError(f"Title/name exceeds {self._config.max_title_length} chars")
        self._validate_content(content)
        self._validate_tags(tags)

    def _validate_content(self, content: str) -> None:
        if len(content) > self._config.max_content_length:
            raise ValueError(f"Content exceeds {self._config.max_content_length} chars")

    def _validate_tags(self, tags: list[str] | None) -> None:
        if tags:
            if len(tags) > self._config.max_tags_per_entry:
                raise ValueError(f"Too many tags (max {self._config.max_tags_per_entry})")
            for tag in tags:
                if len(tag) > self._config.max_tag_length:
                    raise ValueError(
                        f"Tag '{tag[:20]}...' exceeds {self._config.max_tag_length} chars"
                    )

    def _validate_query_input(self, query: str, limit: int, offset: int) -> None:
        if len(query) > self._config.max_query_length:
            raise ValueError(f"Query exceeds {self._config.max_query_length} chars")
        if limit < 1 or limit > self._config.max_limit:
            raise ValueError(f"Limit must be 1-{self._config.max_limit}")
        if offset < 0:
            raise ValueError("Offset must be non-negative")

    # ------------------------------------------------------------------
    # Slug collision handling
    # ------------------------------------------------------------------

    def _unique_key(self, namespace: str, base_key: str) -> str:
        """Return base_key if available, else base_key-2, base_key-3, etc."""
        existing_keys = set(self._store.read_slice(namespace).keys())
        if base_key not in existing_keys:
            return base_key
        n = 2
        while f"{base_key}-{n}" in existing_keys:
            n += 1
        return f"{base_key}-{n}"

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def store_decision(self, title: str, content: str, tags: list[str] | None = None) -> str:
        """Store an architectural decision.

        Returns the assigned key (may include a numeric suffix on collision).
        """
        self._validate_store_input(title, content, tags)
        base_key = _slugify(title)
        key = self._unique_key(_NS_DECISIONS, base_key)
        entry = {
            "title": title,
            "content": content,
            "tags": _normalize_tags(tags),
            "created_at": time.time(),
        }
        self._store.write(_NS_DECISIONS, key, entry)
        return key

    def query_decisions(
        self,
        query: str,
        tags: list[str] | None = None,
        limit: int = 5,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """Query decisions using BM25 scoring.

        Returns (results_with_key, total_matching).
        """
        self._validate_query_input(query, limit, offset)
        return self._search(_NS_DECISIONS, query, tags, limit, offset)

    def delete_decision(self, key: str) -> None:
        """Delete a decision by key. Raises KeyError if not found."""
        existing = self._store.read(_NS_DECISIONS, key)
        if existing is None:
            raise KeyError(f"No decision found with key: {key}")
        self._store.delete(_NS_DECISIONS, key)

    def update_decision(self, key: str, content: str | None = None,
                        tags: list[str] | None = None) -> None:
        """Partial update of a decision. Title is immutable."""
        existing = self._store.read(_NS_DECISIONS, key)
        if existing is None:
            raise KeyError(f"No decision found with key: {key}")
        # Copy before mutating — read() returns a reference to MemoryStore internals.
        updated = dict(existing)
        if content is not None:
            self._validate_content(content)
            updated["content"] = content
        if tags is not None:
            self._validate_tags(tags)
            updated["tags"] = _normalize_tags(tags)
        updated["updated_at"] = time.time()
        self._store.write(_NS_DECISIONS, key, updated)

    # ------------------------------------------------------------------
    # Patterns
    # ------------------------------------------------------------------

    def store_pattern(self, name: str, content: str, tags: list[str] | None = None) -> str:
        """Store a code pattern.

        Returns the assigned key (may include a numeric suffix on collision).
        """
        self._validate_store_input(name, content, tags)
        base_key = _slugify(name)
        key = self._unique_key(_NS_PATTERNS, base_key)
        entry = {
            "name": name,
            "content": content,
            "tags": _normalize_tags(tags),
            "created_at": time.time(),
        }
        self._store.write(_NS_PATTERNS, key, entry)
        return key

    def query_patterns(
        self,
        query: str,
        tags: list[str] | None = None,
        limit: int = 5,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """Query patterns using BM25 scoring.

        Returns (results_with_key, total_matching).
        """
        self._validate_query_input(query, limit, offset)
        return self._search(_NS_PATTERNS, query, tags, limit, offset)

    def delete_pattern(self, key: str) -> None:
        """Delete a pattern by key. Raises KeyError if not found."""
        existing = self._store.read(_NS_PATTERNS, key)
        if existing is None:
            raise KeyError(f"No pattern found with key: {key}")
        self._store.delete(_NS_PATTERNS, key)

    def update_pattern(self, key: str, content: str | None = None,
                       tags: list[str] | None = None) -> None:
        """Partial update of a pattern. Name is immutable."""
        existing = self._store.read(_NS_PATTERNS, key)
        if existing is None:
            raise KeyError(f"No pattern found with key: {key}")
        updated = dict(existing)
        if content is not None:
            self._validate_content(content)
            updated["content"] = content
        if tags is not None:
            self._validate_tags(tags)
            updated["tags"] = _normalize_tags(tags)
        updated["updated_at"] = time.time()
        self._store.write(_NS_PATTERNS, key, updated)

    # ------------------------------------------------------------------
    # Conventions
    # ------------------------------------------------------------------

    def get_conventions(self, category: str | None = None) -> dict[str, Any]:
        """Get project conventions.

        Args:
            category: Optional category filter. None returns all.

        Returns:
            Dict of category -> convention content.
        """
        all_entries = self._store.read_slice(_NS_CONVENTIONS)
        if category:
            entry = all_entries.get(category)
            return {category: entry} if entry else {}
        return dict(all_entries)

    def update_conventions(self, category: str, content: str) -> None:
        """Update a convention category."""
        self._validate_content(content)
        self._store.write(_NS_CONVENTIONS, category, {
            "content": content,
            "updated_at": time.time(),
        })

    def delete_convention(self, category: str) -> None:
        """Delete a convention by category. Raises KeyError if not found."""
        existing = self._store.read(_NS_CONVENTIONS, category)
        if existing is None:
            raise KeyError(f"No convention found with category: {category}")
        self._store.delete(_NS_CONVENTIONS, category)

    # ------------------------------------------------------------------
    # Cross-namespace search
    # ------------------------------------------------------------------

    def search_all(self, query: str, tags: list[str] | None = None,
                   limit: int = 5, offset: int = 0) -> tuple[list[dict], int]:
        """BM25 search over merged corpus of all namespaces."""
        self._validate_query_input(query, limit, offset)

        # 1. Collect all entries with namespace labels.
        all_items: list[tuple[str, dict]] = []
        for ns, label in [
            (_NS_DECISIONS, "decision"),
            (_NS_PATTERNS, "pattern"),
            (_NS_CONVENTIONS, "convention"),
        ]:
            for key, entry in self._store.read_slice(ns).items():
                entry_copy = dict(entry)
                entry_copy["_namespace"] = label
                entry_copy["_key"] = key
                all_items.append((f"{ns}:{key}", entry_copy))

        if not all_items:
            return [], 0

        query_tokens = _tokenize(query)
        norm_tags = _normalize_tags(tags) if tags else None

        if not query_tokens:
            if norm_tags:
                all_items = [
                    (k, e) for k, e in all_items
                    if set(e.get("tags", [])) & set(norm_tags)
                ]
            total = len(all_items)
            page = all_items[offset: offset + limit]
            return [entry for _, entry in page], total

        # 2. Tokenize all entries (single corpus).
        doc_token_lists = [_tokenize(self._entry_text(e)) for _, e in all_items]
        n = len(doc_token_lists)

        # 3. Precompute IDF over the merged corpus (BM25+ variant).
        doc_sets = [set(tl) for tl in doc_token_lists]
        idf_map: dict[str, float] = {}
        for term in set(query_tokens):
            df = sum(1 for ds in doc_sets if term in ds)
            idf_map[term] = math.log((n - df + 0.5) / (df + 0.5) + 1)

        avgdl = sum(len(tl) for tl in doc_token_lists) / max(n, 1)
        avgdl = max(avgdl, 1.0)
        now = time.time()
        k1 = self._config.bm25_k1
        b = self._config.bm25_b

        # 4. Score each entry.
        scored: list[tuple[float, dict]] = []
        for (_, entry), tokens in zip(all_items, doc_token_lists, strict=True):
            if norm_tags:
                entry_tags = set(entry.get("tags", []))
                if not (entry_tags & set(norm_tags)):
                    continue

            score = _bm25_score(query_tokens, tokens, idf_map, avgdl, k1, b)
            ts = max(entry.get("created_at", 0), entry.get("updated_at", 0))
            age_days = (now - ts) / 86400
            if age_days < self._config.recency_window_days:
                score *= 1.0 + self._config.recency_factor * (
                    1 - age_days / self._config.recency_window_days
                )
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        total = len(scored)
        page = scored[offset: offset + limit]
        return [entry for _, entry in page], total

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return corpus statistics for observability."""
        stats: dict[str, Any] = {"namespaces": {}}
        all_tags: dict[str, int] = {}
        oldest = float("inf")
        newest = 0.0

        for ns, label in [
            (_NS_DECISIONS, "decisions"),
            (_NS_PATTERNS, "patterns"),
            (_NS_CONVENTIONS, "conventions"),
        ]:
            entries = self._store.read_slice(ns)
            count = len(entries)
            stats["namespaces"][label] = {"count": count}
            for entry in entries.values():
                for tag in entry.get("tags", []):
                    all_tags[tag] = all_tags.get(tag, 0) + 1
                ts = max(entry.get("created_at", 0), entry.get("updated_at", 0))
                if ts > 0:
                    oldest = min(oldest, ts)
                    newest = max(newest, ts)

        stats["total_entries"] = sum(
            ns["count"] for ns in stats["namespaces"].values()
        )
        stats["top_tags"] = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)[:20]
        stats["oldest_entry"] = oldest if oldest != float("inf") else None
        stats["newest_entry"] = newest if newest > 0 else None
        return stats

    # ------------------------------------------------------------------
    # BM25 search
    # ------------------------------------------------------------------

    def _search(
        self, namespace: str, query: str, tags: list[str] | None,
        limit: int, offset: int = 0,
    ) -> tuple[list[dict], int]:
        """BM25 search. Returns (results_with_key, total_matching)."""
        entries = self._store.read_slice(namespace)
        if not entries:
            return [], 0

        query_tokens = _tokenize(query)

        # Normalize query-side tags so they match write-time normalized tags.
        norm_tags = _normalize_tags(tags) if tags else None

        if not query_tokens:
            items = list(entries.items())
            if norm_tags:
                items = [
                    (k, e) for k, e in items
                    if set(e.get("tags", [])) & set(norm_tags)
                ]
            total = len(items)
            page = items[offset: offset + limit]
            return [dict(entry, _key=key) for key, entry in page], total

        # Build per-doc token lists.
        doc_keys: list[str] = []
        doc_token_lists: list[list[str]] = []
        for key, entry in entries.items():
            doc_keys.append(key)
            doc_token_lists.append(_tokenize(self._entry_text(entry)))

        # Precompute IDF for query terms (once, not per-doc).
        n = len(doc_token_lists)
        doc_sets = [set(tl) for tl in doc_token_lists]
        idf_map: dict[str, float] = {}
        for term in set(query_tokens):
            df = sum(1 for ds in doc_sets if term in ds)
            idf_map[term] = math.log((n - df + 0.5) / (df + 0.5) + 1)

        avgdl = sum(len(tl) for tl in doc_token_lists) / max(n, 1)
        avgdl = max(avgdl, 1.0)
        now = time.time()
        k1 = self._config.bm25_k1
        b = self._config.bm25_b

        scored: list[tuple[float, str, dict]] = []
        for idx, (key, entry) in enumerate(
            zip(doc_keys, entries.values(), strict=True)
        ):
            # Tag filter: skip non-matching entries before scoring.
            if norm_tags:
                entry_tags = set(entry.get("tags", []))
                if not (entry_tags & set(norm_tags)):
                    continue

            score = _bm25_score(query_tokens, doc_token_lists[idx], idf_map, avgdl, k1, b)

            # Recency bonus — multiplicative so it scales with relevance.
            ts = max(entry.get("created_at", 0), entry.get("updated_at", 0))
            age_days = (now - ts) / 86400
            if age_days < self._config.recency_window_days:
                score *= 1.0 + self._config.recency_factor * (
                    1 - age_days / self._config.recency_window_days
                )

            scored.append((score, key, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        total = len(scored)
        page = scored[offset: offset + limit]
        return [dict(entry, _key=key) for _, key, entry in page], total

    def _entry_text(self, entry: dict) -> str:
        """Extract searchable text from an entry."""
        parts = []
        for field in ("title", "name", "content", "category"):
            val = entry.get(field, "")
            if val:
                parts.append(str(val))
        for tag in entry.get("tags", []):
            parts.append(str(tag))
        return " ".join(parts)
