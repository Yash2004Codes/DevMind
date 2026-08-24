"""
mcp_server.py — DevMind Multi-Agent MCP Server

╔══════════════════════════════════════════════════════════════════════════╗
║  DevMind: Developer Productivity Intelligence                           ║
║  Multi-Agent MCP Server powered by LangGraph + CockroachDB + Bedrock   ║
╚══════════════════════════════════════════════════════════════════════════╝

This is the MCP server entry point. Claude Desktop connects to this process
and can invoke any of the registered tools below.

Architecture:
  Claude Desktop
    └─► MCP Protocol (stdio)
          └─► This server (mcp_server.py)
                ├─► Orchestrator (LangGraph)
                │     ├─► Memory Agent   → CockroachDB agent_memory
                │     ├─► Code Agent     → CockroachDB code_chunks + GitHub API
                │     ├─► Issue Agent    → CockroachDB gh_issues + GitHub API
                │     └─► Decision Agent → CockroachDB arch_decisions
                └─► Direct tool calls (for ingestion / admin)

Claude Desktop config (~/.claude/claude_desktop_config.json):
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
          "GITHUB_REPO": "owner/repo-name"
        }
      }
    }
  }

Usage:
  python mcp_server.py               # start MCP server (stdio transport)
  python mcp_server.py --init-db     # create all DB tables then exit
  python mcp_server.py --ingest      # index GITHUB_REPO then exit
  python mcp_server.py --stats       # show knowledge base stats then exit
"""

from __future__ import annotations

import os
import sys
import argparse

from dotenv import load_dotenv
load_dotenv()

import sqlalchemy as sa
from fastmcp import FastMCP

from memory_store           import MemoryStore
from agents.memory_agent    import MemoryAgent
from agents.code_agent      import CodeAgent
from agents.issue_agent     import IssueAgent
from agents.decision_agent  import DecisionAgent
from agents.orchestrator    import Orchestrator


# ── Global MCP Server Instance ──────────────────────────────────────────────────

mcp = FastMCP("DevMind")

# ── Lazy globals (initialized on first tool call) ─────────────────────────────
_store:          MemoryStore   | None = None
_orchestrator:   Orchestrator  | None = None
_code_agent:     CodeAgent     | None = None
_issue_agent:    IssueAgent    | None = None
_decision_agent: DecisionAgent | None = None


def _get_engine() -> sa.Engine:
    db_uri = os.environ["COCKROACH_DB_URI"]
    # Use cockroachdb dialect — fixes version string parsing for CockroachDB v22+
    if "cockroachdb+psycopg2" not in db_uri:
        db_uri = (
            db_uri
            .replace("postgresql+psycopg2://", "cockroachdb+psycopg2://", 1)
            .replace("postgresql://", "cockroachdb+psycopg2://", 1)
            .replace("postgres://",   "cockroachdb+psycopg2://", 1)
        )
    return sa.create_engine(
        db_uri,
        pool_pre_ping = True,
        pool_size     = 5,
        max_overflow  = 2,
        pool_recycle  = 1800,
    )


def _build_local_embeddings():
    """
    Build a local sentence-transformers embedding model.
    Downloads ~90 MB on first run, then cached locally. Completely free.
    all-MiniLM-L6-v2 → 384-dim vectors, fast on CPU.
    """
    from langchain_community.embeddings import FastEmbedEmbeddings
    return FastEmbedEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def _init_services():
    """
    Lazily initialize all services on the first tool call.
    Uses Anthropic API for LLM + local sentence-transformers for embeddings.
    """
    global _store, _orchestrator, _code_agent, _issue_agent, _decision_agent

    if _store is not None:
        return  # already initialized

    github_token = os.getenv("GITHUB_TOKEN", "")

    embeddings   = _build_local_embeddings()
    engine       = _get_engine()
    _store       = MemoryStore(engine, embeddings)

    _code_agent     = CodeAgent(_store, github_token)
    _issue_agent    = IssueAgent(_store, github_token)
    _decision_agent = DecisionAgent(_store, github_token=github_token)
    memory_agent    = MemoryAgent(_store)

    _orchestrator = Orchestrator(
        memory_agent   = memory_agent,
        code_agent     = _code_agent,
        issue_agent    = _issue_agent,
        decision_agent = _decision_agent,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MCP TOOLS — Claude Desktop can call any of these
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def ask_devmind(question: str) -> str:
    """
    Ask DevMind anything about your project's history, code, issues, or decisions.

    This is the MAIN tool. DevMind will automatically route your question to the
    right combination of agents (Memory, Code, Issue, Decision) and synthesize
    a comprehensive answer.

    Examples:
    - "Why did we stop using Redis for caching?"
    - "Who has worked on the authentication module?"
    - "Find issues similar to this error: connection timeout after 30s"
    - "What architectural decisions were made in the last sprint?"
    - "What did we discuss about the database schema last week?"
    - "Summarize all changes related to the payments feature"
    """
    _init_services()
    return _orchestrator.query(question)


@mcp.tool()
def search_code(
    query: str,
    repo: str = "",
    top_k: int = 5,
) -> str:
    """
    Search GitHub PRs, commits, and code changes semantically.

    Use this when you want to find specific code changes, understand what
    changed in a module, or find who authored a particular piece of code.

    Args:
        query: What you're looking for (e.g. "Redis connection pool setup")
        repo:  Optional repo filter (e.g. "myorg/api-service"). Leave empty to search all.
        top_k: Number of results (default 5)

    Returns:
        Formatted list of matching code chunks with file paths, authors, PR numbers.
    """
    _init_services()
    results = _code_agent.search_code(
        query,
        repo  = repo if repo else None,
        top_k = top_k,
    )
    if not results:
        return "No matching code chunks found. Try indexing your repo first with `index_github_repo`."

    lines = [f"Found {len(results)} code matches for: '{query}'\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. [{r['similarity']:.2f}] {r['chunk_type'].upper()} | "
            f"{r['file_path']} | by {r['author']}"
            + (f" | PR #{r['pr_number']}" if r.get("pr_number") else "")
            + f"\n   {r['preview']}\n"
        )
    return "\n".join(lines)


@mcp.tool()
def search_issues(
    query: str,
    repo: str = "",
    state: str = "",
    top_k: int = 5,
) -> str:
    """
    Search GitHub Issues semantically — find bugs, feature requests, and discussions.

    Args:
        query: Natural language description (e.g. "memory leak in auth service")
        repo:  Optional repo filter. Leave empty to search all repos.
        state: Optional filter: 'open', 'closed', or '' for all
        top_k: Number of results (default 5)

    Returns:
        Formatted list of matching issues with titles, states, labels, authors.
    """
    _init_services()
    results = _issue_agent.search_issues(
        query,
        repo  = repo if repo else None,
        state = state if state else None,
        top_k = top_k,
    )
    if not results:
        return "No matching issues found. Try indexing your repo first with `index_github_repo`."

    lines = [f"Found {len(results)} issues for: '{query}'\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. [{r['similarity']:.2f}] #{r['issue_number']} [{r['state'].upper()}] "
            f"{r['title']}\n"
            f"   Labels: {r['labels'] or 'none'} | Author: {r['author']}\n"
            f"   {r['body_preview']}\n"
        )
    return "\n".join(lines)


@mcp.tool()
def search_decisions(
    query: str,
    repo: str = "",
    top_k: int = 5,
) -> str:
    """
    Search architectural decisions — find out WHY technical choices were made.

    Use this for questions like:
    - "Why did we choose PostgreSQL over MongoDB?"
    - "What was the reason we deprecated the v1 API?"
    - "Why did we switch away from Redux?"

    Args:
        query: Your "why did we" or "what was the reason" question
        repo:  Optional repo filter
        top_k: Number of results (default 5)

    Returns:
        Formatted list of relevant architectural decisions with rationale.
    """
    _init_services()
    results = _decision_agent.search_decisions(
        query,
        repo  = repo if repo else None,
        top_k = top_k,
    )
    if not results:
        return (
            "No architectural decisions found matching that query.\n"
            "Tip: Run `index_github_repo` to extract decisions from your PRs, "
            "or use `log_decision` to manually record a decision."
        )

    lines = [f"Found {len(results)} decisions for: '{query}'\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. [{r['similarity']:.2f}] {r['title']}\n"
            f"   Source: {r['source_type']} #{r['source_ref']} | By: {r['author']}\n"
            f"   Rationale: {r['rationale']}\n"
        )
    return "\n".join(lines)


@mcp.tool()
def search_memories(
    query: str,
    top_k: int = 5,
) -> str:
    """
    Search past conversations with DevMind semantically.

    Use this to recall things you've discussed before, like:
    - "What did I ask about the API design?"
    - "Find our conversation about database migration"

    Args:
        query: What you're trying to recall
        top_k: Number of results (default 5)

    Returns:
        Formatted list of past conversation turns ranked by relevance.
    """
    _init_services()
    from agents.memory_agent import MemoryAgent
    mem_agent = MemoryAgent(_store)
    results   = mem_agent.search_memories(query, top_k=top_k)

    if not results:
        return "No matching past conversations found."

    lines = [f"Found {len(results)} past conversations for: '{query}'\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. [{r['similarity']:.2f}] {r['stored_at']}\n"
            f"   You:      {r['user_prompt'][:150]}\n"
            f"   DevMind:  {r['ai_response'][:200]}\n"
        )
    return "\n".join(lines)


@mcp.tool()
def find_file_owner(
    file_path: str,
    repo: str = "",
) -> str:
    """
    Find who has worked on a specific file based on indexed PRs and commits.

    Args:
        file_path: Path of the file (e.g. "src/auth/middleware.py")
        repo:      Optional repo filter

    Returns:
        List of contributors with their PRs linked to that file.
    """
    _init_services()
    owners = _code_agent.get_file_owners(
        file_path,
        repo  = repo if repo else None,
        top_k = 5,
    )
    if not owners:
        return f"No contributors found for '{file_path}'. Is the repo indexed?"

    lines = [f"Contributors to '{file_path}':\n"]
    for o in owners:
        lines.append(
            f"  • {o['author']}"
            + (f" — PR #{o['pr_number']}" if o.get("pr_number") else "")
            + f" | last indexed: {o['indexed_at']}"
        )
    return "\n".join(lines)


@mcp.tool()
def log_decision(
    repo: str,
    title: str,
    rationale: str,
    author: str = "me",
) -> str:
    """
    Manually record an architectural decision into DevMind's knowledge base.

    Use this to preserve important decisions that weren't captured in PRs/issues.

    Args:
        repo:      Repository this decision belongs to (e.g. "myorg/api")
        title:     Short title (e.g. "Switched from REST to GraphQL")
        rationale: Full reasoning (e.g. "REST was causing over-fetching; GraphQL cuts payload by 60%")
        author:    Who made this decision (default: "me")

    Returns:
        Confirmation message.
    """
    _init_services()
    result = _decision_agent.store_manual_decision(
        repo      = repo,
        title     = title,
        rationale = rationale,
        author    = author,
    )
    if result.get("status") == "ok":
        return f"✅ Decision logged: '{title}'"
    return f"❌ Error: {result.get('message')}"


@mcp.tool()
def index_github_repo(
    repo_name: str,
    max_prs: int = 50,
    max_commits: int = 100,
    max_issues: int = 200,
) -> str:
    """
    Index a GitHub repository into DevMind's knowledge base.

    This ingests PRs, commits, issues, and architectural decisions.
    Run this once per repo, then re-run periodically to pick up new activity.

    Args:
        repo_name:   Full GitHub repo name (e.g. "octocat/Hello-World")
        max_prs:     Max PRs to index (default 50)
        max_commits: Max commits to index (default 100)
        max_issues:  Max issues to index (default 200)

    Returns:
        Summary of what was indexed.

    Note:
        Requires GITHUB_TOKEN in environment for private repos.
        Public repos work without a token (rate-limited to 60 req/hr).
    """
    _init_services()

    lines = [f"🔄 Indexing {repo_name}...\n"]

    # Index code (PRs + commits)
    code_counts = _code_agent.index_repo(
        repo_name, max_prs=max_prs, max_commits=max_commits
    )
    lines.append(
        f"💻 Code:      {code_counts['prs_indexed']} PRs, "
        f"{code_counts['commits_indexed']} commits "
        f"({code_counts['errors']} errors)"
    )

    # Index issues
    issue_counts = _issue_agent.index_repo_issues(
        repo_name, state="all", max_issues=max_issues
    )
    lines.append(
        f"🐛 Issues:    {issue_counts['indexed']} indexed "
        f"({issue_counts['errors']} errors)"
    )

    # Extract decisions from PRs
    decision_counts = _decision_agent.extract_and_store_decisions_from_prs(
        repo_name, max_prs=max_prs
    )
    lines.append(
        f"🏛️  Decisions:  {decision_counts['decisions_found']} extracted "
        f"from {decision_counts['prs_scanned']} PRs"
    )

    lines.append(f"\n✅ Indexing complete for {repo_name}")
    lines.append("You can now ask DevMind questions about this repo!")
    return "\n".join(lines)


@mcp.tool()
def devmind_stats() -> str:
    """
    Show DevMind knowledge base statistics — how much is indexed per domain.

    Returns:
        Table of row counts for each knowledge domain.
    """
    _init_services()
    stats = _store.get_stats()
    lines = [
        "📊 DevMind Knowledge Base Stats",
        "─" * 35,
        f"💬 Conversations:   {stats.get('conversations', 0):>6,} records",
        f"💻 Code chunks:     {stats.get('code', 0):>6,} records",
        f"🐛 Issues:          {stats.get('issues', 0):>6,} records",
        f"🏛️  Decisions:       {stats.get('decisions', 0):>6,} records",
        "─" * 35,
        f"   Total:           {sum(stats.values()):>6,} records",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CLI — for setup tasks (run directly, not via MCP)
# ══════════════════════════════════════════════════════════════════════════════

def _cli_init_db():
    """Create all database tables defined in db/schema.sql."""
    import pathlib
    schema_path = pathlib.Path(__file__).parent / "db" / "schema.sql"
    schema_sql  = schema_path.read_text(encoding="utf-8")

    engine = _get_engine()
    # Split on semicolons and execute each statement
    statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            if stmt.startswith("--") or not stmt:
                continue
            try:
                conn.execute(sa.text(stmt))
                print(f"  ✓ {stmt[:60].strip()}...")
            except Exception as e:
                print(f"  ⚠ Skipped: {e}")
    print("\n✅ All tables created successfully.")


def _cli_ingest():
    """Index the repo specified in GITHUB_REPO env var."""
    repo = os.getenv("GITHUB_REPO", "")
    if not repo:
        print("❌ Set GITHUB_REPO=owner/repo-name in your .env")
        sys.exit(1)
    print(f"Indexing {repo}...")
    result = index_github_repo(repo)
    print(result)


def _cli_stats():
    """Print knowledge base stats."""
    print(devmind_stats())


def main():
    parser = argparse.ArgumentParser(
        prog        = "mcp_server.py",
        description = "DevMind Multi-Agent MCP Server",
    )
    parser.add_argument("--init-db",  action="store_true", help="Create DB tables then exit")
    parser.add_argument("--ingest",   action="store_true", help="Index GITHUB_REPO then exit")
    parser.add_argument("--stats",    action="store_true", help="Show stats then exit")
    args = parser.parse_args()

    if args.init_db:
        _cli_init_db()
        sys.exit(0)

    if args.ingest:
        _init_services()
        _cli_ingest()
        sys.exit(0)

    if args.stats:
        _init_services()
        _cli_stats()
        sys.exit(0)

    # Default: start the MCP server (Claude Desktop connects here via stdio)
    mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
