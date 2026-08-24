-- ─────────────────────────────────────────────────────────────────────────────
-- DevMind Multi-Agent MCP — Extended CockroachDB Schema
-- Run this ONCE against your CockroachDB cluster to add the new tables.
-- The original agent_memory table is preserved as-is.
-- ─────────────────────────────────────────────────────────────────────────────

-- ── 1. Ensure pgvector extension (CockroachDB ≥ 22.2 has it built-in) ────────
-- CREATE EXTENSION IF NOT EXISTS vector;  -- uncomment if using plain Postgres

-- ── 2. Original table (keep as-is, already exists) ───────────────────────────
CREATE TABLE IF NOT EXISTS agent_memory (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  STRING      NOT NULL,
    user_prompt TEXT        NOT NULL,
    ai_response TEXT        NOT NULL,
    full_turn   TEXT        NOT NULL,
    embedding   VECTOR(384),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- ── 3. GitHub code chunks table ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS code_chunks (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    repo        STRING      NOT NULL,
    file_path   STRING      NOT NULL,
    chunk_text  TEXT        NOT NULL,
    author      STRING      NOT NULL DEFAULT '',
    pr_number   INT,
    commit_sha  STRING      NOT NULL DEFAULT '',
    chunk_type  STRING      NOT NULL DEFAULT 'code',
    embedding   VECTOR(384),
    indexed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);


CREATE INDEX IF NOT EXISTS code_chunks_repo_idx
    ON code_chunks (repo, file_path);

-- ── 4. GitHub Issues table ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gh_issues (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    repo         STRING      NOT NULL,
    issue_number INT         NOT NULL,
    title        TEXT        NOT NULL,
    body         TEXT        NOT NULL DEFAULT '',
    state        STRING      NOT NULL DEFAULT 'open',
    labels       STRING      NOT NULL DEFAULT '',
    author       STRING      NOT NULL DEFAULT '',
    embedding    VECTOR(384),
    indexed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (repo, issue_number)
);


CREATE INDEX IF NOT EXISTS gh_issues_repo_state_idx
    ON gh_issues (repo, state);

-- ── 5. Architectural Decisions table ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS arch_decisions (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    repo        STRING      NOT NULL,
    title       TEXT        NOT NULL,
    rationale   TEXT        NOT NULL,
    source_type STRING      NOT NULL DEFAULT 'pr',
    source_ref  STRING      NOT NULL DEFAULT '',
    author      STRING      NOT NULL DEFAULT '',
    embedding   VECTOR(384),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- ── 6. Knowledge graph edges ──────────────────────────────────────────────────
-- Links entities (people, files, issues, decisions) together.
CREATE TABLE IF NOT EXISTS knowledge_edges (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    from_type   STRING      NOT NULL,
    from_ref    STRING      NOT NULL,
    to_type     STRING      NOT NULL,
    to_ref      STRING      NOT NULL,
    relation    STRING      NOT NULL,
    repo        STRING      NOT NULL DEFAULT '',
    weight      FLOAT       NOT NULL DEFAULT 1.0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS knowledge_edges_from_idx
    ON knowledge_edges (from_type, from_ref);

CREATE INDEX IF NOT EXISTS knowledge_edges_to_idx
    ON knowledge_edges (to_type, to_ref);
