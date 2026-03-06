# Codebase Knowledge MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io/) server that gives LLM agents structured access to architectural decisions, code patterns, and project conventions. Backed by BM25 search with stemming, stopword removal, and document-length normalization — no embedding model required.

## Why this exists

Agents working on a codebase need to know *why* things are the way they are, not just *what* the code does. This server stores and retrieves three categories of knowledge:

| Namespace | What it captures | Example |
|-----------|-----------------|---------|
| **Decisions** | Architectural choices and their rationale | "Use PostgreSQL for ACID compliance" |
| **Patterns** | Reusable code patterns and implementation strategies | "Repository pattern for data access" |
| **Conventions** | Project-wide standards and rules | "Use camelCase for variables, UTC for timestamps" |

Each entry is searchable by content and filterable by tags. Results include a `_key` field so agents can update or delete entries without guessing internal identifiers.

## Tools

The server exposes 13 tools over the MCP protocol:

### Write

| Tool | Parameters | Returns |
|------|-----------|---------|
| `store_decision` | `title`, `content`, `tags?` | Key (auto-suffixed on collision: `use-postgresql-2`) |
| `store_pattern` | `name`, `content`, `tags?` | Key |
| `update_decision` | `key`, `content?`, `tags?` | Confirmation (title is immutable) |
| `update_pattern` | `key`, `content?`, `tags?` | Confirmation (name is immutable) |
| `update_conventions` | `category`, `content` | Confirmation |

### Delete

| Tool | Parameters | Returns |
|------|-----------|---------|
| `delete_decision` | `key` | Confirmation or KeyError |
| `delete_pattern` | `key` | Confirmation or KeyError |
| `delete_convention` | `category` | Confirmation or KeyError |

### Search

| Tool | Parameters | Returns |
|------|-----------|---------|
| `query_decisions` | `query`, `tags?`, `limit?`, `offset?` | `{results, total, offset, limit}` |
| `query_patterns` | `query`, `tags?`, `limit?`, `offset?` | `{results, total, offset, limit}` |
| `search_knowledge` | `query`, `tags?`, `limit?`, `offset?` | Cross-namespace results with `_namespace` labels |

### Read

| Tool | Parameters | Returns |
|------|-----------|---------|
| `get_conventions` | `category?` | All conventions or single category |
| `knowledge_stats` | — | Entry counts, top tags, date range per namespace |

## Search engine

### BM25

Ranking uses the Lucene/BM25+ IDF variant rather than raw TF-IDF:

```
IDF(t) = log((N - df + 0.5) / (df + 0.5) + 1)
score(t, d) = IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |d| / avgdl))
```

The `+1` inside `log()` prevents negative IDF for terms appearing in more than half the corpus — important when the knowledge base has fewer than 20 entries. Parameters `k1` (term frequency saturation) and `b` (length normalization strength) are configurable.

**Why not TF-IDF?** Raw TF-IDF has no document-length normalization. A 500-word document that mentions "REST" once will outscore a 10-word document titled "REST API design" because term frequency is divided by total tokens. BM25's saturation curve and length normalization fix this.

**Why not embeddings?** Zero external dependencies. No embedding model to provision, no vector database to maintain, no API calls at query time. BM25 is the right tool for a corpus of hundreds of entries with keyword-heavy technical vocabulary.

### Stemmer

A 40-line suffix stripper handles English inflections common in programming terminology:

| Suffix group | Examples | Behavior |
|-------------|----------|----------|
| `-ing`, `-ed` | running&rarr;run, stopped&rarr;stop | Doubled-consonant de-dup (`bdglmnprt`) |
| `-e` | create&rarr;creat, cache&rarr;cach | Min stem length 4 |
| `-s`, `-es` | patterns&rarr;pattern, classes&rarr;class | Double-s guard (class, process stay intact) |
| `-able`, `-ible`, `-er` | configurable&rarr;configur, handler&rarr;handl | Aligns with `-e` and `-ing` forms |
| `-ation`, `-tion`, `-sion` | configuration&rarr;configur | Nominal suffixes |
| `-ness`, `-ment`, `-ly` | awareness&rarr;awar, deployment&rarr;deploy | Adjectival/adverbial |

This means a query for `"database scaling"` matches a document containing `"databases scaled efficiently"` — both reduce to stems `databas` and `scal`.

**Known trade-offs:** The `-er` suffix over-stems root words like "server" (to "serv") and "cluster" (to "clust"), but this is consistent — both query and document produce the same stem, so matches still work. No irregular forms ("ran" vs "run") or synonym resolution ("db" vs "database").

### Stopwords

53 common English words are filtered at tokenization time. A query for `"use the REST API for authentication"` becomes `["rest", "api", "authentic"]` — three signal-carrying stems instead of seven tokens diluted by noise.

### Recency

Multiplicative boost for entries newer than `recency_window_days` (default 30):

```
score *= 1.0 + recency_factor * (1 - age_days / recency_window_days)
```

Uses `max(created_at, updated_at)` so updated entries and conventions (which only have `updated_at`) get proper treatment. Multiplicative means zero-relevance documents stay at zero — a recent but irrelevant entry won't surface.

### Cross-namespace search

`search_knowledge` builds a single merged corpus from all three namespaces before computing BM25. This is necessary because IDF values depend on corpus size — computing BM25 separately per namespace produces incomparable scores. Results include `_namespace` ("decision", "pattern", or "convention") so callers know what they're looking at.

## Configuration

All values have sensible defaults. Config is loaded from `config/codebase_knowledge.yaml`:

```yaml
codebase_knowledge:
  # Storage limits
  max_content_length: 50000     # ~50KB per entry
  max_title_length: 200
  max_tag_length: 50
  max_tags_per_entry: 20
  max_query_length: 500
  max_limit: 100                # Max results per query page

  # BM25 tuning
  bm25_k1: 1.5                 # Higher = more weight on term frequency
  bm25_b: 0.75                 # Higher = more penalty for long documents

  # Recency
  recency_window_days: 30
  recency_factor: 0.15         # 15% boost at day 0, linear decay to 0 at day 30
```

The server works without this file — all fields have hardcoded defaults in `KnowledgeConfig`.

## Usage

```bash
# Start the server (stdio transport)
python -m mcp_servers.codebase_knowledge

# With custom store path
python -m mcp_servers.codebase_knowledge --store /path/to/store.json

# With custom config
python -m mcp_servers.codebase_knowledge --config /path/to/config.yaml
```

### MCP client configuration

```json
{
  "mcpServers": {
    "codebase-knowledge": {
      "command": "python",
      "args": ["-m", "mcp_servers.codebase_knowledge"],
      "cwd": "/path/to/ai-orchestrator"
    }
  }
}
```

## Input validation

All inputs are validated before storage:

| Constraint | Default | Error |
|-----------|---------|-------|
| Content length | 50,000 chars | `ValueError` |
| Title/name length | 200 chars | `ValueError` |
| Tag length | 50 chars per tag | `ValueError` |
| Tags per entry | 20 | `ValueError` |
| Query length | 500 chars | `ValueError` |
| Result limit | 1-100 | `ValueError` |
| Offset | >= 0 | `ValueError` |

Tags are normalized at write time: lowercased, stripped, deduplicated. `["Database", " database ", "DB"]` becomes `["database", "db"]`.

## Slug collision handling

When two entries produce the same slug (e.g., `"Use PostgreSQL"` and `"Use PostgreSQL!"`), the server auto-appends a numeric suffix instead of silently overwriting:

```
store_decision("Use PostgreSQL", ...)  -> "use-postgresql"
store_decision("Use PostgreSQL", ...)  -> "use-postgresql-2"
store_decision("Use PostgreSQL", ...)  -> "use-postgresql-3"
```

The returned key always reflects the actual stored key, so the caller can immediately use it for updates or deletes.

## Storage

Data is persisted via `MemoryStore` — a JSON file at `memory/store.json` (configurable). Three namespaces: `kb_decisions`, `kb_patterns`, `kb_conventions`. Writes are atomic (temp file + rename). No external database required.

## Architecture

```
__main__.py          Config loading, KnowledgeStore wiring
    |
server.py            FastMCP instance, 13 @mcp.tool() functions
    |
knowledge.py         KnowledgeStore class, BM25 engine, stemmer, validation
    |
memory_store.py      JSON file persistence (shared with orchestrator)
```

Zero imports from `control_plane` or `execution_plane`. The server is a standalone process that communicates over stdio.

## Tests

128 tests covering:

- **Stemmer truth table** — 37 word pairs validated against expected stems
- **Golden ranking** — 6 test cases asserting specific ranking outcomes (term relevance, stemmer recall, recency boost, length normalization, cross-namespace)
- **CRUD lifecycle** — store, query, update, delete for all three namespaces
- **Validation** — boundary values for all input constraints
- **Pagination** — offset/limit, non-overlapping pages, offset beyond results
- **Tag normalization** — case folding, whitespace stripping, deduplication
- **Copy safety** — mutating query results does not corrupt stored data
- **FastMCP integration** — tool count, tool names, round-trip dispatch

```bash
pytest tests/test_knowledge_store.py tests/test_codebase_knowledge_server.py -v
```
