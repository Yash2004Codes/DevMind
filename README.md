# DevMind — Developer Productivity Intelligence
### Multi-Agent MCP System · LangGraph · CockroachDB · Amazon Bedrock · GitHub

> Ask Claude Desktop anything about your project's entire history — code, issues, decisions — and get answers backed by persistent vector memory.

---

## What is DevMind?

DevMind is a **multi-agent MCP server** that gives Claude Desktop deep knowledge of your software project. It indexes your GitHub repository (PRs, commits, issues) and stores everything as searchable vector embeddings in CockroachDB. When you ask Claude a question, DevMind's LangGraph orchestrator routes it to the right specialist agents and synthesizes a single, cited answer.

**Questions DevMind can answer that no other tool can today:**
- *"Why did we stop using Redis for caching?"*
- *"Who has the most context on the authentication module?"*
- *"Find all bugs similar to this stack trace"*
- *"What architectural decisions were made in the last sprint?"*
- *"What did we discuss about the database schema design?"*

---

## Architecture

```
Claude Desktop
    │ MCP Protocol (stdio)
    ▼
mcp_server.py  ←── FastMCP entry point · 8 registered tools
    │
    ▼
agents/orchestrator.py  ←── LangGraph StateGraph
    │
    ├─► agents/memory_agent.py    ──► agent_memory  (vector)
    ├─► agents/code_agent.py      ──► code_chunks   (vector) + GitHub API
    ├─► agents/issue_agent.py     ──► gh_issues     (vector) + GitHub API
    └─► agents/decision_agent.py  ──► arch_decisions (vector)
                │
                ▼
        memory_store.py  ←── Shared CockroachDB pgvector layer
                │
                ▼
        CockroachDB Serverless
```

### Why this architecture?

| Problem | Solution |
|---|---|
| LLMs forget between sessions | CockroachDB vector store persists everything |
| Searching GitHub is siloed | All domains indexed into one unified vector store |
| "Why" questions have no answer | Decision Agent extracts rationale from PRs |
| Single agent can't handle all queries | LangGraph routes to specialist agents |
| Context window overflow | RAG retrieves only top-K relevant chunks |

---

## MCP Tools (what Claude Desktop can call)

| Tool | What it does |
|---|---|
| `ask_devmind` | Main tool — routes to all relevant agents, synthesizes answer |
| `search_code` | Semantic search over PRs, commits, file changes |
| `search_issues` | Find GitHub Issues by description/error |
| `search_decisions` | Find architectural decisions and their rationale |
| `search_memories` | Search past DevMind conversations |
| `find_file_owner` | Who worked on a specific file |
| `log_decision` | Manually record an architectural decision |
| `index_github_repo` | Index a GitHub repo into the knowledge base |
| `devmind_stats` | Knowledge base statistics |

---

## Quick Start

### Prerequisites
- Python 3.10+
- CockroachDB Serverless account (free tier works)
- AWS account with Bedrock enabled (Claude 3.5 Sonnet + Titan Embed V2)
- GitHub Personal Access Token (repo read scope)
- Claude Desktop installed

### 1. Clone and install

```bash
cd d:/Projects/TimepassProject
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env
# Edit .env with your real credentials
```

Required variables in `.env`:
```
COCKROACH_DB_URI=postgresql://user:pass@host:26257/defaultdb?sslmode=verify-full
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
GITHUB_TOKEN=github_pat_...
GITHUB_REPO=owner/your-repo-name
```

### 3. Initialize the database

```bash
python setup_db.py
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

See [`CLAUDE_DESKTOP_SETUP.md`](CLAUDE_DESKTOP_SETUP.md) for the full guide.

**Quick version** — add to `%APPDATA%\Claude\claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "devmind": {
      "command": "python",
      "args": ["d:/Projects/TimepassProject/mcp_server.py"],
      "env": {
        "COCKROACH_DB_URI": "...",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "...",
        "AWS_SECRET_ACCESS_KEY": "...",
        "GITHUB_TOKEN": "...",
        "GITHUB_REPO": "owner/repo"
      }
    }
  }
}
```

Restart Claude Desktop. DevMind is live! 🚀

---

## Running Directly (without Claude Desktop)

```bash
# Windows launcher (recommended)
run.bat

# Or directly:
python mcp_server.py --init-db    # create tables
python mcp_server.py --ingest     # index repo
python mcp_server.py --stats      # show stats
python mcp_server.py              # start MCP server
python agent.py                   # original terminal chat agent
```

---

## Database Schema

```sql
agent_memory    -- Conversation history (original, preserved)
code_chunks     -- GitHub PR bodies, commit messages, file diffs
gh_issues       -- GitHub Issues with state, labels, authors
arch_decisions  -- Architectural decisions extracted from PRs
knowledge_edges -- Links between people, files, issues, decisions
```

All tables with `VECTOR(1536)` columns and IVFFlat cosine similarity indexes.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `COCKROACH_DB_URI` | ✅ | CockroachDB connection string |
| `AWS_REGION` | ✅ | e.g. `us-east-1` |
| `AWS_ACCESS_KEY_ID` | ✅ | IAM key with `bedrock:InvokeModel` |
| `AWS_SECRET_ACCESS_KEY` | ✅ | IAM secret |
| `GITHUB_TOKEN` | ✅ | PAT with `repo` read scope |
| `GITHUB_REPO` | ✅ | Default repo to ingest (e.g. `owner/repo`) |

---

## Project Structure

```
TimepassProject/
├── mcp_server.py              ← MCP entry point (Claude Desktop connects here)
├── memory_store.py            ← Shared CockroachDB vector memory layer
├── agent.py                   ← Original terminal chat agent (preserved)
├── setup_db.py                ← Database initialisation script
├── run.bat                    ← Windows launcher
│
├── agents/
│   ├── orchestrator.py        ← LangGraph multi-agent orchestrator
│   ├── memory_agent.py        ← Conversation memory search
│   ├── code_agent.py          ← GitHub PR/commit indexing & search
│   ├── issue_agent.py         ← GitHub Issues indexing & search
│   └── decision_agent.py      ← Architectural decision extraction & search
│
├── ingestion/
│   └── github_ingester.py     ← Standalone scheduled ingestion pipeline
│
├── db/
│   └── schema.sql             ← Full CockroachDB schema (all 5 tables)
│
├── requirements.txt           ← All Python dependencies
├── .env.example               ← Environment variable template
└── CLAUDE_DESKTOP_SETUP.md   ← Claude Desktop configuration guide
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI Orchestration | LangGraph (StateGraph, conditional routing) |
| LLM | Groq Llama 3.1 70B (Free API) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| MCP Server | FastMCP (Python SDK) |
| Vector Database | CockroachDB Serverless (pgvector, IVFFlat) |
| GitHub Integration | PyGithub |
| ORM | SQLAlchemy 2.x |
| Terminal UI | Rich |
