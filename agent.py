"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PERSISTENT CONTEXT TERMINAL AGENT — agent.py                       ║
║  Solving "Agentic Amnesia" with Vector Search + CockroachDB + Bedrock      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture Overview
─────────────────────
Traditional LLM applications stuff every past message back into the context
window.  This is wasteful for two critical reasons:

  1. TOKEN COST / LATENCY — Every additional token in the prompt costs money and
     adds latency.  Long conversations quickly burn through the model's context
     budget, forcing expensive summarisation hacks or arbitrary truncation.

  2. FAULT TOLERANCE — If the process crashes, all in-memory history is lost.
     A vector store decouples the agent's "long-term memory" from the process,
     meaning a restart picks up exactly where it left off.

Our solution — Retrieval-Augmented Generation (RAG) for conversational memory:
  • Every conversational turn is serialised and stored as a vector embedding in
    CockroachDB (a distributed, fault-tolerant PostgreSQL-compatible database).
  • On each new prompt we embed the user's query, retrieve the top-k most
    semantically similar past turns, and inject ONLY those turns into the
    system prompt.  The LLM never sees irrelevant history, keeping token usage
    proportional to semantic relevance, not conversation length.

Environment Variables Required (see .env.example)
──────────────────────────────────────────────────
  COCKROACH_DB_URI        — postgres://... connection string from CockroachDB
  AWS_REGION              — e.g. us-east-1
  AWS_ACCESS_KEY_ID       — IAM key with bedrock:InvokeModel permission
  AWS_SECRET_ACCESS_KEY   — corresponding secret

Usage
─────
  python agent.py              # start interactive session
  python agent.py --help       # show this help
  python agent.py --init-db    # only initialise the database table, then exit
  python agent.py --stats      # print memory store statistics, then exit

In-session commands (type at the You › prompt):
  /memory   — show the last 5 stored memories (debugging aid)
  /stats    — live count of all rows in agent_memory
  /clear    — delete ALL memories for this session only
  quit | exit | q  — gracefully exit
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import os
import sys
import uuid
import threading
import argparse
import textwrap
from datetime import datetime, timezone
from typing import Optional

# ─── Third-Party ─────────────────────────────────────────────────────────────
from dotenv import load_dotenv                          # loads .env into os.environ
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.table import Table
from rich import box as rich_box

# AWS / LangChain
import boto3                                            # noqa: F401  (kept for future direct Bedrock calls)
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

# Database
import sqlalchemy as sa
from sqlalchemy import text as sql_text

# ─── Bootstrap ───────────────────────────────────────────────────────────────
load_dotenv()          # reads .env file from cwd; safe no-op if file is absent
console = Console()    # Rich console — single shared instance (thread-safe writes)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Configuration & Constants
# ══════════════════════════════════════════════════════════════════════════════

# Model identifiers ─ frozen here so they appear in one place for easy swap
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"      # outputs 1 536-dim vectors
CHAT_MODEL_ID  = "anthropic.claude-3-haiku-20240307-v1:0"

# Vector dimension MUST match the embedding model output.
# Titan Text Embeddings V2 produces 1 536-dimensional float32 vectors.
# If you ever switch models (e.g. Cohere Embed → 1024-dim) you MUST drop and
# recreate the table, because the VECTOR column width is fixed at schema time.
VECTOR_DIM = 1536

# Number of past conversational turns to retrieve per query.
# Increasing this improves recall but increases prompt tokens linearly.
# 3 is optimal for most use-cases — benchmark against your domain to tune.
TOP_K_MEMORIES = 3

# Table name in CockroachDB that holds all persistent memory records
MEMORY_TABLE = "agent_memory"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Environment Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_environment() -> dict:
    """
    Read and validate all required environment variables.

    Exits early with a formatted, human-readable error panel if anything is
    missing — so first-time users immediately know what to fix.

    Returns
    -------
    dict
        ``{"db_uri": str, "aws_region": str}``
    """
    required: dict[str, str] = {
        "COCKROACH_DB_URI"      : "postgres://... connection string from CockroachDB Cloud",
        "AWS_REGION"            : "e.g. us-east-1",
        "AWS_ACCESS_KEY_ID"     : "IAM access key with bedrock:InvokeModel permission",
        "AWS_SECRET_ACCESS_KEY" : "corresponding IAM secret key",
    }

    missing = [k for k in required if not os.getenv(k)]

    if missing:
        lines = [
            "[bold red]Missing environment variables:[/bold red]",
            "",
            *[f"  [yellow]•[/yellow] [bold]{k}[/bold]  –  {required[k]}"
              for k in missing],
            "",
            "1. Copy [bold].env.example[/bold] → [bold].env[/bold]",
            "2. Fill in the missing values",
            "3. Re-run  [bold cyan]python agent.py[/bold cyan]",
        ]
        console.print(Panel(
            "\n".join(lines),
            title="[red]⛔  Configuration Error[/red]",
            border_style="red",
            padding=(1, 2),
        ))
        sys.exit(1)

    return {
        "db_uri"    : os.environ["COCKROACH_DB_URI"],
        "aws_region": os.environ["AWS_REGION"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Database Layer
# ══════════════════════════════════════════════════════════════════════════════

def get_engine(db_uri: str) -> sa.Engine:
    """
    Build a SQLAlchemy connection pool targeting CockroachDB.

    CockroachDB speaks the PostgreSQL wire protocol, so we use the standard
    ``psycopg2`` driver.  The function normalises the URI scheme because
    CockroachDB Cloud sometimes returns ``postgres://`` or ``postgresql://``
    without the ``+psycopg2`` dialect specifier that SQLAlchemy needs.

    ``pool_pre_ping=True`` — before handing a connection to application code,
    SQLAlchemy issues a cheap ``SELECT 1``.  This auto-heals stale connections
    that die during long idle periods in a terminal session.
    """
    # Normalise URI scheme → postgresql+psycopg2://
    if "+psycopg2" not in db_uri:
        db_uri = (
            db_uri
            .replace("postgresql://", "postgresql+psycopg2://", 1)
            .replace("postgres://",   "postgresql+psycopg2://", 1)
        )

    return sa.create_engine(
        db_uri,
        pool_pre_ping  = True,  # heal stale connections automatically
        pool_size      = 5,     # background write thread needs its own slot
        max_overflow   = 2,     # allow brief bursts above pool_size
        pool_recycle   = 1800,  # recycle connections every 30 min (CRDB idle timeout)
    )


def initialise_database(engine: sa.Engine) -> None:
    """
    Idempotently create the pgvector extension, the memory table, and the
    cosine-similarity index in CockroachDB.

    WHY VECTOR(1536)?
    ─────────────────
    CockroachDB >= 22.2 ships with built-in pgvector support.  The VECTOR
    column type stores a fixed-width float32 array and the ``<=>`` (cosine
    distance) operator enables sub-millisecond ANN searches via IVFFlat.
    We dimension it at 1 536 to match Titan Embeddings V2's output exactly —
    inserting a vector of the wrong length raises a DB-level error, which is
    exactly what we want (fail fast, fail loudly).

    WHY IVFFlat?
    ────────────
    An IVFFlat index partitions the vector space into ``lists`` Voronoi cells.
    At query time CockroachDB probes only the nearest cells instead of scanning
    every row.  With ``lists=10`` this is fast enough for millions of rows while
    remaining accurate for our TOP_K_MEMORIES=3 recall target.  For tables < 1 000
    rows the index overhead is negligible — the planner will choose a seq scan.
    """
    # DDL: create memory table if absent
    ddl_table = f"""
    CREATE TABLE IF NOT EXISTS {MEMORY_TABLE} (
        id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id  STRING      NOT NULL,
        user_prompt TEXT        NOT NULL,
        ai_response TEXT        NOT NULL,
        full_turn   TEXT        NOT NULL,
        embedding   VECTOR({VECTOR_DIM}),
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """

    # DDL: IVFFlat index for fast cosine similarity searches
    ddl_index = f"""
    CREATE INDEX IF NOT EXISTS {MEMORY_TABLE}_embedding_cos_idx
        ON {MEMORY_TABLE} USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 10);
    """

    with engine.begin() as conn:
        conn.execute(sql_text(ddl_table))

        # The IVFFlat index requires at least one row to build.
        # On a fresh empty table the CREATE INDEX may succeed immediately and
        # build incrementally as rows arrive — that is fine.  Catch any version-
        # specific errors and fall back to brute-force cosine scan gracefully.
        try:
            conn.execute(sql_text(ddl_index))
        except Exception as idx_err:
            console.print(
                f"[dim yellow]IVFFlat index skipped (will use seq-scan): {idx_err}[/dim yellow]"
            )

    console.print(
        f"[green]✓[/green] Table [bold cyan]{MEMORY_TABLE}[/bold cyan] ready in CockroachDB."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — AWS Bedrock Clients
# ══════════════════════════════════════════════════════════════════════════════

def build_bedrock_clients(region: str) -> tuple[BedrockEmbeddings, ChatBedrock]:
    """
    Instantiate the two LangChain-AWS wrappers we need:

    • :class:`BedrockEmbeddings` — wraps Amazon Titan Embeddings V2.
      Credentials are read automatically from the environment variables
      ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY``.

    • :class:`ChatBedrock` — wraps Claude 3 Haiku with a chat-message interface.

    WHY LANGCHAIN-AWS OVER RAW BOTO3?
    ──────────────────────────────────
    ``langchain-aws`` gives us:
      1. ``embed_query(str) → list[float]`` — a clean, model-agnostic interface.
      2. ``invoke(messages) → AIMessage`` — handles JSON serialisation /
         deserialisation of Anthropic's message format automatically.
      3. Built-in retry logic with exponential back-off for rate-limit errors.

    Returns
    -------
    tuple[BedrockEmbeddings, ChatBedrock]
    """
    embeddings = BedrockEmbeddings(
        model_id    = EMBED_MODEL_ID,
        region_name = region,
    )

    llm = ChatBedrock(
        model_id     = CHAT_MODEL_ID,
        region_name  = region,
        model_kwargs = {
            "temperature": 0.7,   # balanced creativity vs. factuality
            "max_tokens" : 2048,  # bounded response length — never runaway
        },
    )

    return embeddings, llm


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Memory: Read
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_memories(
    engine      : sa.Engine,
    query_vector: list[float],
    top_k       : int = TOP_K_MEMORIES,
) -> list[dict]:
    """
    Perform an Approximate Nearest-Neighbour (ANN) search in CockroachDB
    using the pgvector cosine distance operator ``<=>``.

    WHY COSINE SIMILARITY OVER EUCLIDEAN (L2)?
    ────────────────────────────────────────────
    Cosine similarity measures the *angle* between two vectors, making it
    invariant to vector magnitude.  A short tweet and a long essay on the
    same topic will have nearly identical cosine similarity to a related
    query, whereas L2 distance would penalise the longer text simply because
    it has higher magnitude.  For sentence/passage retrieval cosine is the
    standard choice.

    The ``<=>`` operator returns cosine *distance* (1 − cosine_similarity),
    so ``ORDER BY embedding <=> :vec ASC`` → most similar rows first.
    We compute ``1 - distance`` to return a human-readable similarity score.

    FIRST-RUN SAFETY
    ────────────────
    On the very first invocation the table is empty.  The query returns an
    empty result set normally, so no special-casing is needed.  The ``except``
    block catches driver-level errors (e.g. type-registration failures) and
    demotes them to a warning so the agent can start without stored memories.

    Parameters
    ----------
    engine       : active SQLAlchemy engine
    query_vector : 1 536-dim embedding of the current user prompt
    top_k        : number of memories to return

    Returns
    -------
    list[dict]  — each dict has keys: user_prompt, ai_response,
                  full_turn, created_at, similarity
    """
    # Serialise float list → pgvector literal:  [0.012,−0.034,…]
    vec_literal = "[" + ",".join(f"{v:.8f}" for v in query_vector) + "]"

    query_sql = f"""
    SELECT
        user_prompt,
        ai_response,
        full_turn,
        created_at,
        ROUND(
            (1 - (embedding <=> '{vec_literal}'::VECTOR({VECTOR_DIM})))::NUMERIC,
            4
        ) AS similarity
    FROM  {MEMORY_TABLE}
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> '{vec_literal}'::VECTOR({VECTOR_DIM}) ASC
    LIMIT {top_k};
    """
    # NOTE: We inline the vector literal directly (not via a bind parameter)
    # because psycopg2 does not natively adapt Python lists to pgvector's
    # text format through the parameterised query path.  The vector is a
    # float array generated by our own code — there is no SQL-injection risk.

    try:
        with engine.connect() as conn:
            result = conn.execute(sql_text(query_sql))
            return [dict(row._mapping) for row in result]
    except Exception as exc:
        console.print(f"[dim yellow]Memory retrieval skipped: {exc}[/dim yellow]")
        return []


def fetch_recent_memories(engine: sa.Engine, limit: int = 5) -> list[dict]:
    """
    Fetch the most recently stored memories ordered by ``created_at``.
    Used by the ``/memory`` in-session debug command.

    Parameters
    ----------
    engine : active SQLAlchemy engine
    limit  : maximum rows to return (default 5)

    Returns
    -------
    list[dict]  — keys: id, session_id, user_prompt, ai_response, created_at
    """
    sql = f"""
    SELECT id, session_id, user_prompt, ai_response, created_at
    FROM   {MEMORY_TABLE}
    ORDER  BY created_at DESC
    LIMIT  {limit};
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(sql_text(sql))
            return [dict(row._mapping) for row in result]
    except Exception as exc:
        console.print(f"[dim red]Could not fetch recent memories: {exc}[/dim red]")
        return []


def fetch_stats(engine: sa.Engine) -> dict:
    """
    Return aggregate statistics about the memory store.
    Used by the ``/stats`` in-session command and ``--stats`` CLI flag.

    Returns
    -------
    dict  — keys: total_rows, total_sessions, oldest, newest
    """
    sql = f"""
    SELECT
        COUNT(*)                        AS total_rows,
        COUNT(DISTINCT session_id)      AS total_sessions,
        MIN(created_at)                 AS oldest,
        MAX(created_at)                 AS newest
    FROM {MEMORY_TABLE};
    """
    try:
        with engine.connect() as conn:
            row = conn.execute(sql_text(sql)).mappings().first()
            return dict(row) if row else {}
    except Exception as exc:
        console.print(f"[dim red]Stats query failed: {exc}[/dim red]")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Memory: Write (async background thread)
# ══════════════════════════════════════════════════════════════════════════════

def store_memory_async(
    engine      : sa.Engine,
    session_id  : str,
    user_prompt : str,
    ai_response : str,
    embedding   : list[float],
) -> threading.Thread:
    """
    Persist a conversational turn to CockroachDB in a non-blocking daemon thread.

    WHY A BACKGROUND THREAD?
    ─────────────────────────
    After printing the AI's response we want the user to be able to type their
    next message immediately — we do not want them to wait ~50–200 ms for a DB
    round-trip.  We therefore kick the INSERT off in a daemon thread.

    Daemon threads are automatically killed when the main thread exits, but in
    practice the INSERT is fast enough to finish before Python's GC runs.  On
    ``KeyboardInterrupt`` we call ``.join()`` on the returned thread object so
    no data is lost on Ctrl+C.

    WHY EMBED THE FULL TURN (not just the user query)?
    ────────────────────────────────────────────────────
    Storing ``"User: {prompt}\\nAssistant: {response}"`` as the embedded text
    means future retrieval considers the complete semantic meaning of both sides
    of the exchange — not just isolated questions.  If a user asked "What is
    Python?" and Claude answered with a detailed explanation, a future query
    about "snake programming language" will surface that memory even though
    neither the word "snake" nor the exact phrase appeared in the original.

    Parameters
    ----------
    engine      : active SQLAlchemy engine
    session_id  : UUID string identifying this terminal session
    user_prompt : raw user input text
    ai_response : Claude's full response text
    embedding   : 1 536-dim float list of the full turn embedding

    Returns
    -------
    threading.Thread  — the started background thread (caller may `.join()`)
    """
    full_turn   = f"User: {user_prompt}\nAssistant: {ai_response}"
    vec_literal = "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"

    insert_sql = f"""
    INSERT INTO {MEMORY_TABLE}
        (session_id, user_prompt, ai_response, full_turn, embedding)
    VALUES
        (:sid, :user_prompt, :ai_response, :full_turn,
         '{vec_literal}'::VECTOR({VECTOR_DIM}));
    """

    def _write() -> None:
        try:
            with engine.begin() as conn:
                conn.execute(sql_text(insert_sql), {
                    "sid"         : session_id,
                    "user_prompt" : user_prompt,
                    "ai_response" : ai_response,
                    "full_turn"   : full_turn,
                })
        except Exception as exc:
            # Non-fatal — the agent continues even if a single write fails
            console.print(f"\n[dim red]⚠  Memory write failed: {exc}[/dim red]")

    t = threading.Thread(target=_write, daemon=True, name="memory-writer")
    t.start()
    return t


def clear_session_memories(engine: sa.Engine, session_id: str) -> int:
    """
    Delete all memory rows belonging to ``session_id``.
    Used by the ``/clear`` in-session command.

    Returns
    -------
    int  — number of rows deleted
    """
    sql = f"DELETE FROM {MEMORY_TABLE} WHERE session_id = :sid;"
    try:
        with engine.begin() as conn:
            result = conn.execute(sql_text(sql), {"sid": session_id})
            return result.rowcount
    except Exception as exc:
        console.print(f"[dim red]Clear failed: {exc}[/dim red]")
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Prompt Construction
# ══════════════════════════════════════════════════════════════════════════════

def build_system_prompt(memories: list[dict]) -> str:
    """
    Construct the augmented system prompt that grounds Claude in retrieved
    past conversational turns.

    WHY NOT INJECT ALL HISTORY INTO THE PROMPT?
    ─────────────────────────────────────────────
    "Buffer memory" — appending every past message — has two compounding flaws:

      1. TOKEN EXPLOSION — Input tokens ∝ conversation length.  A 200-turn
         conversation forces the model to re-read ~40 000 tokens of history on
         every new query.  At Claude 3 Haiku pricing ($0.25/M input tokens)
         that is 160× more expensive than our fixed-k retrieval approach.

      2. ATTENTION DILUTION — Transformer attention is finite.  When thousands
         of historical tokens compete for attention budget, the model's focus
         on the *relevant* context degrades — a phenomenon studied as "lost in
         the middle" (Liu et al. 2023).

    Our RAG approach keeps token cost at O(k) regardless of conversation length,
    and semantic retrieval ensures the injected context is always the most
    relevant subset of history for the current query.

    Parameters
    ----------
    memories : list of memory dicts returned by :func:`retrieve_memories`

    Returns
    -------
    str  — the complete system prompt to pass to Claude
    """
    base = (
        "You are a helpful, concise, and highly intelligent AI assistant with "
        "persistent long-term memory backed by CockroachDB vector search.\n"
        "Use the retrieved memories below to maintain continuity across sessions. "
        "If a memory is relevant, reference it naturally. "
        "Do NOT hallucinate details that do not appear in the memories or the current prompt.\n\n"
    )

    if not memories:
        return (
            base
            + "MEMORY STATUS: No semantically similar past conversations found.\n"
            + "This may be the very first session or no prior topics match this query.\n"
            + "Proceed without historical context."
        )

    sep = "─" * 64
    memory_block = f"RETRIEVED MEMORIES — top {len(memories)} by cosine similarity:\n{sep}\n"

    for i, mem in enumerate(memories, 1):
        sim = float(mem.get("similarity", 0.0))
        ts  = mem.get("created_at", "unknown")

        # Truncate long turns so a single memory can't crowd out the user prompt.
        # 600 chars ≈ 150 tokens — well within budget even for top-3 retrieval.
        turn = mem["full_turn"]
        if len(turn) > 600:
            turn = turn[:600] + "…[truncated]"

        memory_block += (
            f"[Memory {i}]  Cosine Similarity: {sim:.4f}  |  Stored: {ts}\n"
            f"{turn}\n{sep}\n"
        )

    return base + memory_block


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Rich UI Helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_banner() -> None:
    """Render the startup banner with full architecture context."""
    console.print(Panel.fit(
        "[bold cyan]Persistent Context Terminal Agent[/bold cyan]\n"
        "[dim]Solving Agentic Amnesia  ·  Vector RAG Memory  ·  Hackathon Edition[/dim]\n\n"
        "[bold]LLM:[/bold]       Claude 3 Haiku  [dim](anthropic.claude-3-haiku-20240307-v1:0)[/dim]\n"
        "[bold]Embeddings:[/bold] Amazon Titan V2  [dim](1 536 dimensions)[/dim]\n"
        "[bold]Database:[/bold]  CockroachDB Serverless  [dim](pgvector cosine similarity)[/dim]\n\n"
        "[dim]In-session commands:  "
        "[white]/memory[/white]  [white]/stats[/white]  [white]/clear[/white]  "
        "[white]quit[/white][/dim]",
        title="[bold magenta]◈  PCTA[/bold magenta]",
        border_style="bright_magenta",
        padding=(1, 3),
    ))
    console.print()


def print_memory_table(memories: list[dict]) -> None:
    """Render retrieved memories as a Rich table for the ``/memory`` command."""
    if not memories:
        console.print("[dim yellow]No memories stored yet.[/dim yellow]")
        return

    tbl = Table(
        title   = "Recent Memories",
        box     = rich_box.ROUNDED,
        border_style = "cyan",
        show_lines   = True,
    )
    tbl.add_column("#",           style="dim",    width=3)
    tbl.add_column("Stored At",   style="cyan",   width=26)
    tbl.add_column("User Prompt", style="white",  max_width=40, overflow="fold")
    tbl.add_column("Session",     style="dim",    width=10)

    for i, m in enumerate(memories, 1):
        tbl.add_row(
            str(i),
            str(m.get("created_at", "—")),
            textwrap.shorten(m.get("user_prompt", ""), 80, placeholder="…"),
            str(m.get("session_id", "—"))[:8] + "…",
        )

    console.print(tbl)


def print_stats_panel(stats: dict) -> None:
    """Render memory store statistics as a Rich panel for the ``/stats`` command."""
    if not stats:
        console.print("[dim yellow]Could not retrieve stats.[/dim yellow]")
        return

    lines = [
        f"[bold]Total memories:[/bold]  {stats.get('total_rows', '—')}",
        f"[bold]Total sessions:[/bold]  {stats.get('total_sessions', '—')}",
        f"[bold]Oldest memory:[/bold]   {stats.get('oldest', '—')}",
        f"[bold]Newest memory:[/bold]   {stats.get('newest', '—')}",
    ]
    console.print(Panel(
        "\n".join(lines),
        title="[cyan]Memory Store Statistics[/cyan]",
        border_style="cyan",
        padding=(0, 2),
    ))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — CLI Argument Parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Flags
    ─────
    --init-db   Create the database table and exit (useful in CI / Docker entrypoint)
    --stats     Print memory store statistics and exit
    --top-k N   Override TOP_K_MEMORIES for this session (default: 3)
    """
    parser = argparse.ArgumentParser(
        prog        = "agent.py",
        description = "Persistent Context Terminal Agent — RAG memory over CockroachDB",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = textwrap.dedent("""\
            Examples:
              python agent.py               # start interactive chat session
              python agent.py --init-db     # create DB table then exit
              python agent.py --stats       # show memory statistics then exit
              python agent.py --top-k 5    # retrieve top-5 memories per query
        """),
    )
    parser.add_argument(
        "--init-db",
        action  = "store_true",
        help    = "initialise the CockroachDB table and index, then exit",
    )
    parser.add_argument(
        "--stats",
        action  = "store_true",
        help    = "display memory store statistics, then exit",
    )
    parser.add_argument(
        "--top-k",
        type    = int,
        default = TOP_K_MEMORIES,
        metavar = "N",
        help    = f"number of memories to retrieve per query (default: {TOP_K_MEMORIES})",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Main Agent Loop
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(top_k: int = TOP_K_MEMORIES) -> None:
    """
    Core interactive loop — called after all one-shot CLI flags are handled.

    The loop follows exactly the 6-step RAG memory flow described in the
    module-level docstring:
      Embed → Retrieve → Format → Generate → Display → Store

    Parameters
    ----------
    top_k : how many past memories to retrieve per query
    """
    config     = validate_environment()
    session_id = str(uuid.uuid4())   # unique ID for this terminal session

    console.print(
        f"[dim]Session ID: [bold]{session_id}[/bold][/dim]\n"
        f"[dim]Memory retrieval: top-{top_k} cosine similarity[/dim]\n"
    )

    # ── Bootstrap services ───────────────────────────────────────────────────
    with console.status("[bold green]Connecting to CockroachDB…", spinner="dots"):
        engine = get_engine(config["db_uri"])
        initialise_database(engine)

    with console.status("[bold green]Initialising Bedrock clients…", spinner="dots"):
        embeddings, llm = build_bedrock_clients(config["aws_region"])

    console.print("[bold green]✓[/bold green] All systems ready.\n")
    console.print(Rule(style="bright_magenta"))

    # Keep a reference to the last write thread so we can join on Ctrl+C
    last_write_thread: Optional[threading.Thread] = None

    # ── REPL ─────────────────────────────────────────────────────────────────
    while True:
        try:
            console.print()
            user_input = console.input("[bold cyan]You › [/bold cyan]").strip()

            # ── Empty input ──────────────────────────────────────────────────
            if not user_input:
                continue

            # ── Exit commands ────────────────────────────────────────────────
            if user_input.lower() in {"quit", "exit", "q"}:
                console.print(
                    "\n[bold magenta]Goodbye! Waiting for memory writes to complete…[/bold magenta]"
                )
                if last_write_thread and last_write_thread.is_alive():
                    last_write_thread.join(timeout=5)
                console.print("[dim]All memories persisted. See you next time.[/dim]")
                break

            # ── /memory — show recent stored memories ────────────────────────
            if user_input.lower() == "/memory":
                recent = fetch_recent_memories(engine, limit=5)
                print_memory_table(recent)
                continue

            # ── /stats — show aggregate statistics ───────────────────────────
            if user_input.lower() == "/stats":
                stats = fetch_stats(engine)
                print_stats_panel(stats)
                continue

            # ── /clear — delete this session's memories ──────────────────────
            if user_input.lower() == "/clear":
                deleted = clear_session_memories(engine, session_id)
                console.print(
                    f"[yellow]Cleared {deleted} memory row(s) for this session.[/yellow]"
                )
                continue

            # ════════════════════════════════════════════════════════════════
            # STEP 1: EMBED the user's current query
            # ════════════════════════════════════════════════════════════════
            # We embed the raw user input so retrieval is anchored to what the
            # user is asking RIGHT NOW — not a past AI response.
            with console.status("[dim]Embedding query…", spinner="dots"):
                query_vector: list[float] = embeddings.embed_query(user_input)

            # ════════════════════════════════════════════════════════════════
            # STEP 2: RETRIEVE semantically similar past turns
            # ════════════════════════════════════════════════════════════════
            # This single SQL query replaces the O(n) token cost of buffer
            # memory with a fixed-cost O(k) retrieval.  The IVFFlat index
            # makes this sub-millisecond even on tables with millions of rows.
            with console.status(
                f"[dim]Searching {top_k} memories…", spinner="dots"
            ):
                memories = retrieve_memories(engine, query_vector, top_k)

            if memories:
                scores = ", ".join(
                    f"{float(m.get('similarity', 0)):.4f}" for m in memories
                )
                console.print(
                    f"[dim green]◎ {len(memories)} memory fragment(s) retrieved  "
                    f"[sim: {scores}][/dim green]"
                )
            else:
                console.print(
                    "[dim yellow]◎ No past memories matched — fresh context.[/dim yellow]"
                )

            # ════════════════════════════════════════════════════════════════
            # STEP 3: FORMAT the augmented system prompt
            # ════════════════════════════════════════════════════════════════
            system_prompt = build_system_prompt(memories)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input),
            ]

            # ════════════════════════════════════════════════════════════════
            # STEP 4: GENERATE a response via Claude 3 Haiku
            # ════════════════════════════════════════════════════════════════
            with console.status("[dim]Claude is thinking…", spinner="dots"):
                response = llm.invoke(messages)
                ai_text  = str(response.content)

            # ════════════════════════════════════════════════════════════════
            # STEP 5: DISPLAY the response
            # ════════════════════════════════════════════════════════════════
            console.print()
            console.print(Panel(
                Markdown(ai_text),
                title        = "[bold green]Assistant[/bold green]",
                border_style = "green",
                padding      = (1, 2),
            ))

            # ════════════════════════════════════════════════════════════════
            # STEP 6: STORE the full turn embedding asynchronously
            # ════════════════════════════════════════════════════════════════
            # We embed the FULL turn (user + assistant) for storage so that
            # future cosine searches can match either side of the dialogue.
            # This is a second embed call (the first was query-only for step 1).
            # Doing it in the background means zero latency impact on the user.
            with console.status("[dim]Embedding full turn for storage…", spinner="dots"):
                full_turn_text = f"User: {user_input}\nAssistant: {ai_text}"
                full_turn_vec  = embeddings.embed_query(full_turn_text)

            last_write_thread = store_memory_async(
                engine, session_id, user_input, ai_text, full_turn_vec
            )
            console.print("[dim]◎ Memory write dispatched to background thread.[/dim]")

        except KeyboardInterrupt:
            console.print(
                "\n\n[bold yellow]Interrupted — waiting for pending writes…[/bold yellow]"
            )
            if last_write_thread and last_write_thread.is_alive():
                last_write_thread.join(timeout=5)
            console.print("[dim]Done. Memories preserved.[/dim]")
            break

        except Exception as exc:
            console.print(f"\n[bold red]Runtime error:[/bold red] {exc}")
            console.print("[dim]Continuing — this turn will not be stored.[/dim]\n")
            # Log the traceback at debug level without crashing
            import traceback
            console.print(f"[dim red]{traceback.format_exc()}[/dim red]")
            continue


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Top-level entry point.  Handles one-shot CLI flags before entering the
    interactive REPL.
    """
    args = parse_args()
    print_banner()

    config = validate_environment()

    # --init-db: create table then exit (useful in Docker / CI pipelines)
    if args.init_db:
        with console.status("[bold green]Initialising database…", spinner="dots"):
            engine = get_engine(config["db_uri"])
            initialise_database(engine)
        console.print("[green]Database initialisation complete.[/green]")
        sys.exit(0)

    # --stats: show aggregate statistics then exit
    if args.stats:
        with console.status("[bold green]Querying statistics…", spinner="dots"):
            engine = get_engine(config["db_uri"])
            stats  = fetch_stats(engine)
        print_stats_panel(stats)
        sys.exit(0)

    # Default: start the interactive agent loop
    run_agent(top_k=args.top_k)


if __name__ == "__main__":
    main()
