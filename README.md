# 🧠 DevMind — Developer Productivity Intelligence

**Multi-Agent AI System · LangGraph · CockroachDB pgvector · Groq LLM · MCP Protocol**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-green?style=flat-square)
![CockroachDB](https://img.shields.io/badge/CockroachDB-pgvector-red?style=flat-square)
![MCP](https://img.shields.io/badge/MCP-FastMCP-purple?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-Mixtral-orange?style=flat-square)
![RAG](https://img.shields.io/badge/RAG-Vector%20Search-teal?style=flat-square)

---

> **Ask Claude Desktop anything about your project's entire history — code, issues, decisions — and get answers backed by persistent vector memory.**
>
> *"Why did we stop using Redis?"* · *"Who owns auth/middleware.py?"* · *"What ADRs happened last sprint?"*

---

## Table of Contents

1. [What is DevMind?](#what-is-devmind)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Data Flow Diagram](#data-flow-diagram)
5. [Database Schema](#database-schema)
6. [LangGraph Orchestration](#langgraph-orchestration)
7. [MCP Tools Reference](#mcp-tools-reference)
8. [Tech Stack](#tech-stack)
9. [Project Structure](#project-structure)
10. [Quick Start](#quick-start)
11. [Interview Q&A](#interview-qa)

---

## What is DevMind?

DevMind is a **multi-agent, RAG-powered MCP server** that gives Claude Desktop deep, persistent knowledge of your software project. It indexes GitHub repositories (PRs, commits, issues) and stores everything as searchable **384-dimensional vector embeddings** in CockroachDB. When you ask a question, a LangGraph orchestrator routes it to the right specialist agents, retrieves semantically similar chunks, and synthesizes a single cited answer.

**Problems DevMind solves that no other tool does today:**

| Question | Without DevMind | With DevMind |
|---|---|---|
| *"Why did we stop using Redis?"* | Hours reading PRs manually | Instant — decision agent retrieves the rationale |
| *"Who has context on auth middleware?"* | Ask teammates, check git blame | Instant — code agent finds all PR authors |
| *"Find bugs similar to this stack trace"* | Keyword search, often misses | Semantic similarity across all indexed issues |
| *"What did we decide in last sprint?"* | Read meeting notes, PRs | Summarized from arch_decisions table |
| *"What did I tell DevMind about API design?"* | Forgotten forever | Persisted in vector memory, retrieved semantically |

---

## Key Features

### Multi-Agent Intelligence
- **4 specialist agents** (Memory, Code, Issues, Decisions) — each owns its data domain
- **LLM-powered intent router** classifies each query and selects only relevant agents
- **Synthesis node** merges multi-agent results into a single coherent, cited answer

### Semantic Search (RAG)
- All data embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local, free)
- **Cosine similarity** ANN search via CockroachDB pgvector (`<=>` operator)
- Cross-domain search: one query spans conversations, code, issues, and decisions simultaneously
- `top_k` configurable per agent call

### GitHub Knowledge Ingestion
- Indexes **PRs** (title, body, changed files), **commits** (messages, SHA, author), and **GitHub Issues** (state, labels, body)
- **Architectural Decision Record (ADR) extraction** — heuristic keyword scanner detects decision-like language in PR bodies and stores them automatically
- Manual ADR logging via `log_decision` MCP tool
- Idempotent — `ON CONFLICT DO NOTHING` prevents duplicate ingestion

### MCP Server (Model Context Protocol)
- **9 registered tools** usable directly from Claude Desktop
- **Lazy initialization** — server responds to MCP handshake in `<1s`, defers heavy imports to first tool call
- **Background thread pre-warming** — services initialized in parallel with Claude Desktop startup
- stdio transport (JSON-RPC over stdin/stdout) — zero network overhead

### Persistent Vector Memory
- Conversations auto-stored as vector embeddings for future retrieval
- **5 CockroachDB tables** with pgvector support
- Knowledge graph edges table (`knowledge_edges`) linking people, files, issues, and decisions
- Thread-safe async writes via daemon threads

### Production-Ready Design
- Connection pooling: `pool_size=5`, `max_overflow=2`, `pool_recycle=1800`
- CockroachDB dialect fix applied (`cockroachdb+psycopg2`) for v22+ compatibility
- Error isolation — one agent failing does not break others
- Structured logging with clear error boundaries

---

## System Architecture

```
+-------------------------------------------------------------------------+
|                          USER LAYER                                      |
|                                                                         |
|   Claude Desktop  ------  Natural Language Question  ----------------> |
|       (GUI)              "Why did we stop using Redis?"                 |
+--------------------------------+----------------------------------------+
                                 |  MCP Protocol (JSON-RPC over stdio)
                                 v
+-------------------------------------------------------------------------+
|                        MCP SERVER LAYER                                  |
|                                                                         |
|   mcp_server.py  (FastMCP)                                              |
|   +----------------------------------------------------------------------+ |
|   |  Registered Tools (9):                                           |  |
|   |  ask_devmind . search_code . search_issues . search_decisions    |  |
|   |  search_memories . find_file_owner . log_decision                |  |
|   |  index_github_repo . devmind_stats                               |  |
|   +-------------------------------+--------------------------------------+ |
+-------------------------------+-----------------------------------------+
                                |
                                v
+-------------------------------------------------------------------------+
|                     ORCHESTRATION LAYER (LangGraph)                      |
|                                                                         |
|   agents/orchestrator.py  -- StateGraph                                 |
|                                                                         |
|   +-------------+    +------------------+    +----------------------+  |
|   |   CLASSIFY  +---->   RUN_AGENTS     +---->     SYNTHESIZE       |  |
|   |             |    |                  |    |                      |  |
|   | Groq LLM    |    | Sequential exec  |    | Groq LLM merges all  |  |
|   | routes to   |    | of relevant      |    | results into one     |  |
|   | domains:    |    | specialist agents|    | cited answer         |  |
|   | memory/code |    |                  |    |                      |  |
|   | issues/     |    |                  |    |                      |  |
|   | decisions   |    |                  |    |                      |  |
|   +-------------+    +------------------+    +----------------------+  |
+-------------------------------+-----------------------------------------+
                                |  (routed calls)
          +---------------------+------------------+---------------------+
          v                     v                  v                     v
+-----------------+  +------------------+  +------------+  +------------------+
|  MEMORY AGENT  |  |   CODE AGENT     |  | ISSUE AGENT|  | DECISION AGENT   |
|                 |  |                  |  |            |  |                  |
| memory_agent.py |  | code_agent.py    |  | issue_agent|  | decision_agent   |
|                 |  |                  |  |   .py      |  |     .py          |
| Recall past     |  | Search PRs,      |  | Find GitHub|  | Extract & search |
| DevMind         |  | commits, file    |  | Issues by  |  | ADRs from PRs    |
| conversations   |  | diffs, authors   |  | description|  | & manual logs    |
+--------+--------+  +--------+---------+  +------+-----+  +--------+---------+
         |                    |                   |                  |
         +--------------------+-------------------+------------------+
                                       |
                                       v
+-------------------------------------------------------------------------+
|                      PERSISTENCE LAYER                                   |
|                                                                         |
|   memory_store.py  --  MemoryStore class (shared by all agents)         |
|                                                                         |
|   +------------------------------------------------------------------+  |
|   |         CockroachDB Serverless  (pgvector, IVFFlat)             |  |
|   |                                                                 |  |
|   |  agent_memory    code_chunks    gh_issues                       |  |
|   |  (VECTOR 384)    (VECTOR 384)   (VECTOR 384)                    |  |
|   |                                                                 |  |
|   |  arch_decisions  knowledge_edges                                |  |
|   |  (VECTOR 384)    (graph links)                                  |  |
|   +------------------------------------------------------------------+  |
+--------------------------------------+----------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------+
|                       INGESTION LAYER                                    |
|                                                                         |
|   ingestion/github_ingester.py                                          |
|   CodeAgent.index_repo()   IssueAgent.index_repo_issues()              |
|   DecisionAgent.extract_and_store_decisions_from_prs()                 |
|                         |                                               |
|                         v                                               |
|              GitHub API (PyGithub)                                      |
|              PRs . Commits . Issues . Changed Files                     |
+-------------------------------------------------------------------------+
```

---

## Data Flow Diagram

```
INGESTION FLOW
==============

GitHub Repo --> PyGithub API
                     |
          +----------+-----------+
          v          v           v
    PR bodies    Commit msgs  Issue text
    PR files     Author/SHA   Labels/State
          |          |           |
          +----------+-----------+
                     |
                     v
          sentence-transformers
          all-MiniLM-L6-v2
          (384-dim embedding)
                     |
                     v
          CockroachDB (pgvector)
          ON CONFLICT DO NOTHING


QUERY FLOW
==========

User Query (Natural Language)
      |
      v
Groq Mixtral (router, temp=0.1)
"Which domains are relevant?"
-> ["code", "decisions"]
      |
      v
Sequential agent calls:
  CodeAgent.search_code()         --> SELECT ... ORDER BY embedding <=> query_vec LIMIT 5
  DecisionAgent.search_decisions() --> cosine similarity top-K
      |
      v
Groq Mixtral (synthesizer, temp=0.3)
Merges retrieved chunks + cites PR#, issue#, authors
      |
      v
Final Answer --> MCP response --> Claude Desktop
```

---

## Database Schema

```sql
-- 1. Conversation Memory
CREATE TABLE agent_memory (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  STRING       NOT NULL,
    user_prompt TEXT         NOT NULL,
    ai_response TEXT         NOT NULL,
    full_turn   TEXT         NOT NULL,
    embedding   VECTOR(384),           -- cosine similarity search
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 2. GitHub Code Chunks (PRs, commits, file diffs)
CREATE TABLE code_chunks (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    repo        STRING       NOT NULL,
    file_path   STRING       NOT NULL,
    chunk_text  TEXT         NOT NULL,
    author      STRING       NOT NULL DEFAULT '',
    pr_number   INT,
    commit_sha  STRING       NOT NULL DEFAULT '',
    chunk_type  STRING       NOT NULL DEFAULT 'code',  -- pr_body|pr_files|commit_msg
    embedding   VECTOR(384),
    indexed_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 3. GitHub Issues
CREATE TABLE gh_issues (
    id           UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    repo         STRING       NOT NULL,
    issue_number INT          NOT NULL,
    title        TEXT         NOT NULL,
    body         TEXT         NOT NULL DEFAULT '',
    state        STRING       NOT NULL DEFAULT 'open',
    labels       STRING       NOT NULL DEFAULT '',
    author       STRING       NOT NULL DEFAULT '',
    embedding    VECTOR(384),
    indexed_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (repo, issue_number)            -- idempotent upserts
);

-- 4. Architectural Decisions (ADRs)
CREATE TABLE arch_decisions (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    repo        STRING       NOT NULL,
    title       TEXT         NOT NULL,
    rationale   TEXT         NOT NULL,
    source_type STRING       NOT NULL DEFAULT 'pr',   -- pr|manual
    source_ref  STRING       NOT NULL DEFAULT '',
    author      STRING       NOT NULL DEFAULT '',
    embedding   VECTOR(384),
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 5. Knowledge Graph Edges
CREATE TABLE knowledge_edges (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    from_type   STRING       NOT NULL,  -- person|file|issue|decision
    from_ref    STRING       NOT NULL,
    to_type     STRING       NOT NULL,
    to_ref      STRING       NOT NULL,
    relation    STRING       NOT NULL,  -- authored|references|caused
    repo        STRING       NOT NULL DEFAULT '',
    weight      FLOAT        NOT NULL DEFAULT 1.0,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);
```

**Indexes:**
```sql
CREATE INDEX code_chunks_repo_idx       ON code_chunks (repo, file_path);
CREATE INDEX gh_issues_repo_state_idx   ON gh_issues (repo, state);
CREATE INDEX knowledge_edges_from_idx   ON knowledge_edges (from_type, from_ref);
CREATE INDEX knowledge_edges_to_idx     ON knowledge_edges (to_type, to_ref);
```

---

## LangGraph Orchestration

The orchestrator implements a **3-node sequential StateGraph**:

```
[ENTRY] --> classify --> run_agents --> synthesize --> [END]
```

### OrchestratorState (TypedDict)

```python
class OrchestratorState(TypedDict):
    query:            str          # the user's question
    intent:           list[str]    # ["memory", "code", "issues", "decisions"]
    memory_results:   list[dict]   # from MemoryAgent
    code_results:     list[dict]   # from CodeAgent
    issue_results:    list[dict]   # from IssueAgent
    decision_results: list[dict]   # from DecisionAgent
    final_answer:     str          # synthesized response
```

### Node Responsibilities

| Node | LLM Call | Input | Output |
|---|---|---|---|
| `classify` | Groq (temp=0.1) | `query` | `intent: list[str]` |
| `run_agents` | None (pure Python) | `query + intent` | 4 results lists |
| `synthesize` | Groq (temp=0.3) | All results | `final_answer` |

> **Design Note:** Originally attempted parallel fan-out (classify -> 4 agents simultaneously) but encountered `INVALID_CONCURRENT_GRAPH_UPDATE` errors due to LangGraph's immutable state model. Resolved by consolidating to a single `run_agents` node that executes agents sequentially with isolated error handling per agent.

---

## MCP Tools Reference

| Tool | Description | Key Parameters |
|---|---|---|
| `ask_devmind` | **Main tool** — auto-routes question to all relevant agents, synthesizes answer | `question: str` |
| `search_code` | Semantic search over PRs, commits, file changes | `query, repo?, top_k` |
| `search_issues` | Find GitHub Issues by natural language description | `query, repo?, state?, top_k` |
| `search_decisions` | Find architectural decisions and rationale | `query, repo?, top_k` |
| `search_memories` | Search past DevMind conversations | `query, top_k` |
| `find_file_owner` | Who worked on a specific file path | `file_path, repo?` |
| `log_decision` | Manually record an ADR into knowledge base | `repo, title, rationale, author` |
| `index_github_repo` | Index a GitHub repo (PRs + commits + issues + decisions) | `repo_name, max_prs, max_commits, max_issues` |
| `devmind_stats` | Show knowledge base row counts per domain | — |

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **AI Orchestration** | LangGraph `StateGraph` | Deterministic agent routing with typed shared state |
| **LLM (routing + synthesis)** | Groq `mixtral-8x7b-32768` | Free API, fast inference, 32K context window |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, 384-dim, fast on CPU |
| **Vector Database** | CockroachDB Serverless + pgvector | Distributed SQL + vector search in one system; free tier |
| **Similarity Search** | pgvector `<=>` cosine distance | ANN search with IVFFlat index |
| **MCP Server** | FastMCP (official Python SDK) | Zero-boilerplate MCP tool registration |
| **GitHub Integration** | PyGithub | Typed API client for PRs, commits, issues |
| **Database ORM** | SQLAlchemy 2.x | Connection pooling, dialect management |
| **Terminal UI** | Rich | Pretty output for CLI commands |
| **Env Management** | python-dotenv | Clean environment configuration |

---

## Project Structure

```
TimepassProject/
|
+-- mcp_server.py              <- MCP entry point (Claude Desktop connects here)
|                                 9 registered FastMCP tools, lazy init, stdio transport
|
+-- memory_store.py            <- Shared CockroachDB vector memory layer
|                                 MemoryStore class: embed, search, store for all domains
|
+-- agent.py                   <- Original standalone terminal chat agent (preserved)
|
+-- setup_db.py                <- Database initialisation script
|
+-- run.bat                    <- Windows one-click launcher
|
+-- agents/
|   +-- orchestrator.py        <- LangGraph StateGraph (classify -> run_agents -> synthesize)
|   +-- memory_agent.py        <- Conversation memory search (agent_memory table)
|   +-- code_agent.py          <- GitHub PR/commit indexing & search (code_chunks table)
|   +-- issue_agent.py         <- GitHub Issues indexing & search (gh_issues table)
|   +-- decision_agent.py      <- ADR extraction from PRs + manual logging (arch_decisions)
|
+-- ingestion/
|   +-- github_ingester.py     <- Standalone scheduled ingestion pipeline
|
+-- db/
|   +-- schema.sql             <- Full CockroachDB schema (5 tables + indexes)
|
+-- requirements.txt           <- All Python dependencies
+-- .env.example               <- Environment variable template
+-- pytest.ini                 <- Test configuration
+-- test_unit.py               <- Unit tests (agent logic, store, tools)
+-- test_integration.py        <- Integration tests (DB + MCP end-to-end)
+-- CLAUDE_DESKTOP_SETUP.md   <- Claude Desktop configuration guide
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- CockroachDB Serverless account (free tier works)
- Groq API key (free — `console.groq.com`)
- GitHub Personal Access Token (repo read scope)
- Claude Desktop installed

### 1. Clone and install

```bash
git clone <repo-url>
cd TimepassProject
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env
# Edit .env with your credentials
```

```env
COCKROACH_DB_URI=postgresql://user:pass@host:26257/defaultdb?sslmode=verify-full
GROQ_API_KEY=gsk_...
GITHUB_TOKEN=github_pat_...
GITHUB_REPO=owner/your-repo-name
```

### 3. Initialize the database

```bash
python mcp_server.py --init-db
```

Creates 5 tables: `agent_memory`, `code_chunks`, `gh_issues`, `arch_decisions`, `knowledge_edges`

### 4. Index your GitHub repo

```bash
python mcp_server.py --ingest
```

Or with custom limits:
```bash
python -m ingestion.github_ingester --repo owner/myrepo --prs 100 --issues 500
```

### 5. Configure Claude Desktop

Add to `%APPDATA%\Claude\claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "devmind": {
      "command": "python",
      "args": ["d:/Projects/TimepassProject/mcp_server.py"],
      "env": {
        "COCKROACH_DB_URI": "...",
        "GROQ_API_KEY": "...",
        "GITHUB_TOKEN": "...",
        "GITHUB_REPO": "owner/repo"
      }
    }
  }
}
```

Restart Claude Desktop. DevMind is live!

See [`CLAUDE_DESKTOP_SETUP.md`](CLAUDE_DESKTOP_SETUP.md) for the full guide.

### CLI Commands

```bash
python mcp_server.py --init-db    # create tables
python mcp_server.py --ingest     # index repo
python mcp_server.py --stats      # show knowledge base stats
python mcp_server.py              # start MCP server (Claude Desktop)
python agent.py                   # original terminal chat agent
run.bat                           # Windows all-in-one launcher
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `COCKROACH_DB_URI` | YES | CockroachDB connection string |
| `GROQ_API_KEY` | YES | Groq API key for LLM (routing + synthesis) |
| `GROQ_MODEL` | optional | Override default model (`mixtral-8x7b-32768`) |
| `GITHUB_TOKEN` | YES | PAT with `repo` read scope |
| `GITHUB_REPO` | YES | Default repo to ingest (e.g. `owner/repo`) |

---

## Interview Q&A

> Comprehensive preparation guide for technical interviews about this project.

---

### Architecture & Design

**Q: Walk me through the overall architecture of DevMind.**

**A:** DevMind has 5 layers:
1. **Claude Desktop** sends natural language queries via MCP protocol (JSON-RPC over stdio)
2. **MCP Server** (`mcp_server.py`) exposes 9 FastMCP tools — the main one being `ask_devmind`
3. **LangGraph Orchestrator** runs a 3-node StateGraph: `classify -> run_agents -> synthesize`
4. **4 Specialist Agents** (Memory, Code, Issue, Decision) each own a knowledge domain
5. **MemoryStore** is the shared persistence layer backed by CockroachDB pgvector

---

**Q: Why did you use LangGraph instead of a simple function chain?**

**A:** LangGraph provides:
- **Typed shared state** (`TypedDict`) that flows through all nodes — avoids passing data manually between functions
- **Graph visualization** and debuggability — can introspect the state machine
- **Conditional routing** — we can add conditional edges later (e.g., skip memory agent if no past sessions)
- **Checkpointing** capability for long-running or resumable workflows
- **Separation of concerns** — each node is a standalone, testable function

---

**Q: Why did you consolidate from parallel fan-out to sequential `run_agents`?**

**A:** The initial design had 4 parallel edges (classify -> memory, code, issues, decisions simultaneously). This caused `INVALID_CONCURRENT_GRAPH_UPDATE` errors because LangGraph nodes running concurrently cannot write to overlapping state keys without explicit merge reducers. The solution was a single `run_agents` node that calls each agent sequentially but with isolated `try/except` blocks — one agent failing does not block others, and state updates are deterministic.

---

**Q: How does the RAG (Retrieval Augmented Generation) pipeline work?**

**A:**
1. **Ingestion**: GitHub content -> `sentence-transformers` embedding -> CockroachDB pgvector insert
2. **Retrieval**: User query -> embed -> cosine similarity search (`<=>` operator) -> top-K rows
3. **Augmentation**: Retrieved chunks formatted as context with metadata (PR number, author, file path)
4. **Generation**: Groq LLM synthesizes a cited answer using the retrieved context

---

**Q: Why CockroachDB instead of a dedicated vector DB like Pinecone or Weaviate?**

**A:** Key trade-offs:
- **Unified system** — all relational data (issues, PRs) and vectors in one DB, no sync needed
- **ACID transactions** — vector writes are part of the same transaction as metadata
- **pgvector compatibility** — standard SQL with `VECTOR(384)` column type, familiar tooling
- **Free serverless tier** — appropriate for a developer-tooling side project
- **Trade-off**: Dedicated vector DBs (Pinecone, Qdrant) would offer better ANN performance at massive scale, but CockroachDB is sufficient for thousands to low millions of records

---

**Q: What embedding model did you choose and why?**

**A:** `sentence-transformers/all-MiniLM-L6-v2` — 384-dimensional vectors.

Reasons:
- **Free and local** — no API cost, no rate limits, no data privacy concerns
- **Fast on CPU** — approximately 90MB download, runs well without GPU
- **Sufficient quality** for code and developer prose
- **Trade-off**: Specialized code embeddings (`CodeBERT`, `text-embedding-3-small`) would perform better on exact code syntax matching. The architecture is model-agnostic — `VECTOR_DIM` is a single constant, easy to swap.

---

### MCP Protocol

**Q: What is the Model Context Protocol (MCP)?**

**A:** MCP is an open standard by Anthropic that lets AI assistants (like Claude Desktop) call external tools via a JSON-RPC protocol. The transport here is **stdio** — Claude Desktop spawns `mcp_server.py` as a subprocess and communicates via stdin/stdout. This means:
- Zero network setup — no HTTP server needed
- The server must never write non-JSON to stdout (breaks the protocol)
- stderr is used for logs/diagnostics

---

**Q: Why defer heavy imports to `_init_services()` instead of importing at module level?**

**A:** Claude Desktop has a roughly 60-second timeout for MCP server initialization. Heavy imports like SQLAlchemy, LangChain, and sentence-transformers can take 5-10 seconds each. By deferring them to the first tool call, the server responds to the MCP handshake **instantly** (under 1 second). A background daemon thread then pre-warms services so they are ready before the user asks their first question.

---

### Database Design

**Q: Explain your schema design decisions.**

**A:**
- **UUID primary keys** — distributed-safe, no sequence contention in CockroachDB's distributed architecture
- **`VECTOR(384)` columns** — pgvector extension, enables cosine similarity search with `<=>` operator
- **`ON CONFLICT DO NOTHING`** for code chunks — allows idempotent re-ingestion without duplicates
- **`ON CONFLICT ... DO UPDATE`** for issues — updates state changes (open -> closed) on re-ingestion
- **`knowledge_edges` table** — flexible graph model linking entities (person -> file -> issue -> decision) for future graph traversal queries
- **Composite indexes** on `(repo, file_path)` and `(repo, state)` for common filter patterns

---

**Q: How does the cosine similarity search work in SQL?**

**A:**
```sql
SELECT *,
  ROUND((1 - (embedding <=> '[0.1, 0.2, ...]'::VECTOR(384)))::NUMERIC, 4) AS similarity
FROM code_chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> '[0.1, 0.2, ...]'::VECTOR(384) ASC
LIMIT 5;
```

- `<=>` is the pgvector cosine distance operator (0 = identical, 2 = opposite)
- `1 - distance` converts to similarity score (1 = identical)
- Ordering by ASC distance means closest vectors come first
- An IVFFlat index makes this sub-linear (approximate nearest neighbor)

---

### Agent Design

**Q: How does the intent classification work?**

**A:** The `classify` node sends a structured prompt to Groq Mixtral (temp=0.1 for determinism) asking it to return a JSON array of relevant domains: `["memory", "code", "issues", "decisions"]`. The prompt includes few-shot examples. If the LLM returns invalid JSON, it falls back to all 4 domains. This router call is cheap (512 max tokens) and fast.

---

**Q: How does the Decision Agent extract ADRs from PRs?**

**A:** Two approaches:
1. **Heuristic keyword scanner** (no LLM cost): Checks PR body/title for signals like `"decided to"`, `"switched to"`, `"deprecated"`, `"rationale:"`, `"trade-off"`, etc. If matched, stores the entire PR body as the rationale.
2. **Manual logging**: Developers can call `log_decision(repo, title, rationale)` MCP tool to record decisions not captured in PRs.

This avoids LLM calls at ingestion time, keeping costs zero.

---

**Q: How is error isolation implemented across agents?**

**A:** Each agent call in `run_agents` is wrapped in an independent `try/except`:

```python
if "code" in intent:
    try:
        results["code_results"] = self.code_agent.search_code(query)
    except Exception as e:
        print(f"[DevMind] code_agent error: {e}")
        # results["code_results"] stays [] — other agents still run
```

If the Code Agent DB connection fails, Issue Agent and Decision Agent still run. The synthesizer generates the best answer it can from available results.

---

### Scalability & Production

**Q: How would you scale this for a team of 100 engineers?**

**A:**
1. **Scheduled ingestion** — cron job running `github_ingester.py` every hour/day
2. **Index optimization** — IVFFlat lists parameter tuned for dataset size (sqrt(N) rule)
3. **Caching** — Redis cache on top-K results for repeated queries
4. **Multiple repos** — `repo` column enables multi-repo support already
5. **Async agent calls** — replace sequential with `asyncio.gather()` now that state conflicts are understood
6. **Dedicated vector DB** — migrate to Qdrant or pgvector on Postgres for higher QPS
7. **Auth** — add MCP authentication layer, per-user session isolation

---

**Q: What are the limitations of the current implementation?**

**A:**
1. **Sequential agent calls** — not parallelized; 4 agents x ~200ms DB query = ~800ms total latency
2. **No streaming** — Groq response is awaited synchronously; could stream tokens via MCP for faster UX
3. **Embedding quality** — MiniLM is general-purpose; code-specific models would improve code search precision
4. **No IVFFlat index yet** — schema uses regular indexes; ANN index would need `CREATE INDEX ... USING ivfflat`
5. **No freshness** — ingestion is manual/batch, not real-time (no GitHub webhooks)
6. **Single-user** — session IDs are generated per run; no multi-user auth

---

**Q: How would you add real-time ingestion?**

**A:** Set up a **GitHub webhook** -> HTTP endpoint (FastAPI) -> enqueue to a job queue (Celery/RQ) -> worker calls `index_github_repo()` for the relevant PR/issue. This gives near-real-time freshness without polling. The current `ON CONFLICT DO NOTHING / DO UPDATE` design already supports idempotent upserts, so the worker can safely re-process events.

---

### Testing

**Q: How is the system tested?**

**A:**
- **`test_unit.py`** — Unit tests for individual agent logic, MemoryStore methods, MCP tool handlers; mocks DB and LLM calls
- **`test_integration.py`** — End-to-end tests against a real CockroachDB test database; tests the full MCP tool -> agent -> DB -> response pipeline
- **`pytest.ini`** — Configures test discovery and markers for unit vs integration

---

**Q: How would you test the LangGraph orchestrator in isolation?**

**A:** Inject mock agents:
```python
mock_code_agent = Mock()
mock_code_agent.search_code.return_value = [{"similarity": 0.9, "preview": "auth middleware..."}]
orchestrator = Orchestrator(
    memory_agent=Mock(), code_agent=mock_code_agent,
    issue_agent=Mock(), decision_agent=Mock()
)
result = orchestrator.query("who worked on auth?")
assert "auth" in result.lower()
```
This tests routing and synthesis without any DB or LLM calls.

---

### Design Tradeoffs

**Q: What would you do differently if starting from scratch?**

**A:**
1. **Async from the start** — `asyncpg` + `asyncio` throughout for non-blocking DB queries
2. **Streaming responses** — MCP supports streaming; users would see partial answers faster
3. **Structured output for classification** — use Groq JSON mode to guarantee valid JSON from the router instead of `try/except json.loads`
4. **Dedicated ADR format** — ask engineers to write PRs with a standard ADR template for cleaner extraction
5. **Evaluation harness** — a set of golden Q&A pairs to measure retrieval precision@K over time

---

*Built with passion — demonstrating multi-agent RAG, LangGraph orchestration, CockroachDB pgvector, and MCP protocol integration.*
