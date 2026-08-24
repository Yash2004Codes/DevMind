"""
memory_store.py — Shared CockroachDB vector memory layer used by all agents.

This module extracts and refactors the database + embedding logic from the
original agent.py into a clean, reusable class that every specialist agent
and the MCP server can import.
"""

from __future__ import annotations

import os
import threading
from typing import Optional, Any
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy import text as sql_text

# ── Constants ────────────────────────────────────────────────────────────────
# sentence-transformers/all-MiniLM-L6-v2 produces 384-dim vectors (free, local)
# Titan Embed V2 produces 1536-dim vectors (AWS Bedrock, paid)
# We default to local embeddings — override VECTOR_DIM if you switch models.
VECTOR_DIM    = 384   # matches all-MiniLM-L6-v2
TOP_K_DEFAULT = 5


class MemoryStore:
    """
    Unified interface to the CockroachDB pgvector memory backend.

    Supports four logical domains, each with its own table:
      • conversations  — raw chat turns (original agent_memory table)
      • code_chunks    — GitHub PR / commit / file excerpts
      • issues         — GitHub Issues / Jira tickets
      • decisions      — architectural decisions extracted from PRs & issues
    """

    TABLES = {
        "conversations": "agent_memory",
        "code":          "code_chunks",
        "issues":        "gh_issues",
        "decisions":     "arch_decisions",
    }

    def __init__(self, engine: sa.Engine, embeddings: Any):
        self.engine     = engine
        self.embeddings = embeddings
        self._lock      = threading.Lock()

    # ── Embedding helper ─────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """Embed a string using Amazon Titan V2. Returns 1536-dim float list."""
        return self.embeddings.embed_query(text)

    def _vec_literal(self, vec: list[float]) -> str:
        return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"

    # ── ANN search (generic) ─────────────────────────────────────────────────

    def search(
        self,
        query: str,
        domain: str = "conversations",
        top_k: int  = TOP_K_DEFAULT,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Semantic search across any domain table.

        Parameters
        ----------
        query   : natural language query string (will be embedded)
        domain  : one of 'conversations', 'code', 'issues', 'decisions'
        top_k   : number of results to return
        filters : optional dict of column=value equality filters

        Returns
        -------
        list[dict] — rows with similarity score added
        """
        table  = self.TABLES.get(domain, "agent_memory")
        vec    = self.embed(query)
        vlit   = self._vec_literal(vec)

        where  = f"WHERE embedding IS NOT NULL"
        if filters:
            for col, val in filters.items():
                where += f" AND {col} = '{val}'"

        sql = f"""
        SELECT *,
            ROUND((1 - (embedding <=> '{vlit}'::VECTOR({VECTOR_DIM})))::NUMERIC, 4)
                AS similarity
        FROM  {table}
        {where}
        ORDER BY embedding <=> '{vlit}'::VECTOR({VECTOR_DIM}) ASC
        LIMIT {top_k};
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sql_text(sql))
                return [dict(row._mapping) for row in result]
        except Exception as exc:
            return [{"error": str(exc)}]

    def cross_domain_search(
        self,
        query: str,
        top_k_per_domain: int = 3,
    ) -> dict[str, list[dict]]:
        """
        Search ALL domains simultaneously and return results grouped by domain.
        Used by the orchestrator for broad queries.
        """
        results = {}
        for domain in self.TABLES:
            hits = self.search(query, domain=domain, top_k=top_k_per_domain)
            if hits and "error" not in hits[0]:
                results[domain] = hits
        return results

    # ── Write: conversations ─────────────────────────────────────────────────

    def store_conversation_async(
        self,
        session_id: str,
        user_prompt: str,
        ai_response: str,
        embedding: list[float],
    ) -> threading.Thread:
        """Non-blocking: persist a chat turn in a background daemon thread."""
        full_turn = f"User: {user_prompt}\nAssistant: {ai_response}"
        vlit      = self._vec_literal(embedding)

        sql = f"""
        INSERT INTO agent_memory
            (session_id, user_prompt, ai_response, full_turn, embedding)
        VALUES
            (:sid, :user_prompt, :ai_response, :full_turn,
             '{vlit}'::VECTOR({VECTOR_DIM}));
        """

        def _write():
            try:
                with self.engine.begin() as conn:
                    conn.execute(sql_text(sql), {
                        "sid":         session_id,
                        "user_prompt": user_prompt,
                        "ai_response": ai_response,
                        "full_turn":   full_turn,
                    })
            except Exception:
                pass  # non-fatal

        t = threading.Thread(target=_write, daemon=True, name="mem-writer")
        t.start()
        return t

    # ── Write: code chunks ───────────────────────────────────────────────────

    def store_code_chunk(
        self,
        repo: str,
        file_path: str,
        chunk_text: str,
        metadata: dict,
        embedding: Optional[list[float]] = None,
    ) -> None:
        """Store a GitHub code/PR/commit chunk synchronously."""
        if embedding is None:
            embedding = self.embed(chunk_text)
        vlit = self._vec_literal(embedding)

        sql = f"""
        INSERT INTO code_chunks
            (repo, file_path, chunk_text, author, pr_number, commit_sha,
             chunk_type, embedding)
        VALUES
            (:repo, :file_path, :chunk_text, :author, :pr_number,
             :commit_sha, :chunk_type,
             '{vlit}'::VECTOR({VECTOR_DIM}))
        ON CONFLICT DO NOTHING;
        """
        with self.engine.begin() as conn:
            conn.execute(sql_text(sql), {
                "repo":       repo,
                "file_path":  file_path,
                "chunk_text": chunk_text[:4000],
                "author":     metadata.get("author", ""),
                "pr_number":  metadata.get("pr_number"),
                "commit_sha": metadata.get("commit_sha", ""),
                "chunk_type": metadata.get("chunk_type", "code"),
            })

    # ── Write: issues ────────────────────────────────────────────────────────

    def store_issue(
        self,
        repo: str,
        issue_number: int,
        title: str,
        body: str,
        state: str,
        labels: list[str],
        author: str,
        embedding: Optional[list[float]] = None,
    ) -> None:
        """Store a GitHub Issue record."""
        full_text = f"Issue #{issue_number}: {title}\n\n{body}"
        if embedding is None:
            embedding = self.embed(full_text)
        vlit = self._vec_literal(embedding)

        sql = f"""
        INSERT INTO gh_issues
            (repo, issue_number, title, body, state, labels, author, embedding)
        VALUES
            (:repo, :issue_number, :title, :body, :state, :labels, :author,
             '{vlit}'::VECTOR({VECTOR_DIM}))
        ON CONFLICT (repo, issue_number) DO UPDATE
            SET state = EXCLUDED.state,
                body  = EXCLUDED.body,
                embedding = EXCLUDED.embedding;
        """
        with self.engine.begin() as conn:
            conn.execute(sql_text(sql), {
                "repo":         repo,
                "issue_number": issue_number,
                "title":        title,
                "body":         (body or "")[:4000],
                "state":        state,
                "labels":       ",".join(labels),
                "author":       author,
            })

    # ── Write: decisions ─────────────────────────────────────────────────────

    def store_decision(
        self,
        repo: str,
        title: str,
        rationale: str,
        source_type: str,
        source_ref: str,
        author: str,
        embedding: Optional[list[float]] = None,
    ) -> None:
        """Store an architectural decision record."""
        full_text = f"Decision: {title}\nRationale: {rationale}"
        if embedding is None:
            embedding = self.embed(full_text)
        vlit = self._vec_literal(embedding)

        sql = f"""
        INSERT INTO arch_decisions
            (repo, title, rationale, source_type, source_ref, author, embedding)
        VALUES
            (:repo, :title, :rationale, :source_type, :source_ref, :author,
             '{vlit}'::VECTOR({VECTOR_DIM}))
        ON CONFLICT DO NOTHING;
        """
        with self.engine.begin() as conn:
            conn.execute(sql_text(sql), {
                "repo":        repo,
                "title":       title,
                "rationale":   rationale[:4000],
                "source_type": source_type,
                "source_ref":  source_ref,
                "author":      author,
            })

    # ── Stats ────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return row counts for all domain tables."""
        stats = {}
        for domain, table in self.TABLES.items():
            try:
                with self.engine.connect() as conn:
                    row = conn.execute(
                        sql_text(f"SELECT COUNT(*) AS cnt FROM {table}")
                    ).mappings().first()
                    stats[domain] = int(row["cnt"]) if row else 0
            except Exception:
                stats[domain] = -1
        return stats
