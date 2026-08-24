# DevMind — Claude Desktop Setup Guide

## What you need (3 things only)

| Credential | Where to get it | Cost |
|---|---|---|
| **CockroachDB URI** | cockroachlabs.com → free Serverless cluster | Free |
| **Anthropic API Key** | console.anthropic.com → API Keys | $5 free credit |
| **GitHub PAT** | github.com/settings/tokens (repo read scope) | Free |

---

## Step 1 — Create your `.env` file

In the project folder, copy `.env.example` to `.env`:
```bash
copy .env.example .env
```

Open `.env` and fill in your 3 credentials:
```
COCKROACH_DB_URI=postgresql://user:pass@your-cluster.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full
ANTHROPIC_API_KEY=sk-ant-api03-...
GITHUB_TOKEN=github_pat_...
GITHUB_REPO=owner/your-repo-name
```

---

## Step 2 — Initialize the database

```bash
python setup_db.py
```

This creates all 5 tables in CockroachDB. Takes ~5 seconds.

---

## Step 3 — Index your GitHub repo

```bash
python mcp_server.py --ingest
```

This downloads and indexes your repo's PRs, commits, and issues.
**First run downloads the local AI model (~90 MB) — wait for it.**
Takes 2–10 minutes depending on repo size.

---

## Step 4 — Configure Claude Desktop

### Find the config file
```
Windows: %APPDATA%\Claude\claude_desktop_config.json
```
Open it (create it if it doesn't exist).

### Add the DevMind server block

```json
{
  "mcpServers": {
    "devmind": {
      "command": "python",
      "args": ["d:/Projects/TimepassProject/mcp_server.py"],
      "env": {
        "COCKROACH_DB_URI": "postgresql://user:pass@host:26257/defaultdb?sslmode=verify-full",
        "ANTHROPIC_API_KEY": "sk-ant-api03-your-key",
        "GITHUB_TOKEN": "github_pat_your-token",
        "GITHUB_REPO": "owner/your-repo"
      }
    }
  }
}
```

> **Note:** Replace all values with your real credentials.

---

## Step 5 — Restart Claude Desktop

Close and reopen Claude Desktop.
You'll see a 🔧 hammer icon — that means DevMind tools are loaded!

---

## Step 6 — Start asking questions!

Try these in Claude Desktop:

```
Index my GitHub repo first:
  "Use devmind to index the repo owner/myrepo"

Then ask questions:
  "Ask DevMind: why did we stop using Redis?"
  "Search DevMind for issues about authentication"
  "Find who works on the payments module"
  "What architectural decisions were made recently?"
  "What did I ask DevMind last time about the database?"
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "ANTHROPIC_API_KEY not set" | Add it to the `env` block in claude_desktop_config.json |
| "COCKROACH_DB_URI not set" | Same — add it to the env block |
| DevMind not showing in Claude | Restart Claude Desktop completely |
| Slow first query | Normal — local AI model loading for first time |
| "No results found" | Run `index_github_repo` tool first |

---

## Run without Claude Desktop (terminal test)

```bash
# Check stats
python mcp_server.py --stats

# Re-index a repo
python -m ingestion.github_ingester --repo owner/myrepo

# Original chat agent (still works)
python agent.py
```
