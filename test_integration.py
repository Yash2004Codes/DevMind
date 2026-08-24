"""
test_integration.py — Integration Tests (REAL credentials required)
════════════════════════════════════════════════════════════════════
Tests the live system end-to-end:
  CockroachDB connection → table creation → embed → write → retrieve → clear

Run (requires .env with real credentials):
    python -m pytest test_integration.py -v -s

Flags:
    -s          shows print/console output in real-time
    --timeout=60  in case network is slow (pip install pytest-timeout)
"""

import os
import sys
import uuid
import pytest

# Load .env FIRST before importing agent
from dotenv import load_dotenv
load_dotenv()

# Skip the ENTIRE module if credentials are absent
MISSING = [
    k for k in ("COCKROACH_DB_URI", "AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    if not os.getenv(k)
]
if MISSING:
    pytest.skip(
        f"Integration tests skipped — missing env vars: {', '.join(MISSING)}. "
        "Add them to .env and re-run.",
        allow_module_level=True,
    )

import agent  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def engine():
    """Single engine shared across all tests in this module."""
    config = agent.validate_environment()
    eng    = agent.get_engine(config["db_uri"])
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def bedrock_clients():
    """Single pair of Bedrock clients shared across the module."""
    config = agent.validate_environment()
    emb, llm = agent.build_bedrock_clients(config["aws_region"])
    return emb, llm


@pytest.fixture(scope="module")
def test_session_id():
    """A unique session ID used to tag all rows created by this test run."""
    return f"test-{uuid.uuid4()}"


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATABASE CONNECTIVITY
# ══════════════════════════════════════════════════════════════════════════════

class TestDatabaseConnectivity:
    def test_engine_connects(self, engine):
        """A simple SELECT 1 should succeed — proves connectivity."""
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
        assert result == 1, "Expected SELECT 1 = 1"

    def test_table_created_idempotently(self, engine):
        """Calling initialise_database() twice must not raise an error."""
        agent.initialise_database(engine)   # first call
        agent.initialise_database(engine)   # second call — must be idempotent

    def test_table_exists_after_init(self, engine):
        """The agent_memory table must be queryable after initialisation."""
        from sqlalchemy import text
        agent.initialise_database(engine)
        with engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT COUNT(*) FROM {agent.MEMORY_TABLE};")
            ).scalar()
        assert isinstance(result, int), "Row count must be an integer"


# ══════════════════════════════════════════════════════════════════════════════
# 2. AMAZON BEDROCK — EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

class TestEmbeddings:
    def test_embed_query_returns_list(self, bedrock_clients):
        emb, _ = bedrock_clients
        vec = emb.embed_query("Hello, CockroachDB!")
        assert isinstance(vec, list), "Embedding must be a list"

    def test_embed_query_correct_dimension(self, bedrock_clients):
        emb, _ = bedrock_clients
        vec = emb.embed_query("Test sentence for dimension check.")
        assert len(vec) == agent.VECTOR_DIM, \
            f"Expected {agent.VECTOR_DIM} dimensions, got {len(vec)}"

    def test_embed_query_all_floats(self, bedrock_clients):
        emb, _ = bedrock_clients
        vec = emb.embed_query("Floats only, please.")
        assert all(isinstance(v, float) for v in vec), \
            "Every element in the embedding must be a float"

    def test_two_similar_sentences_high_cosine_similarity(self, bedrock_clients):
        """Semantically close sentences should have cosine similarity > 0.85."""
        import math
        emb, _ = bedrock_clients
        v1 = emb.embed_query("The capital of France is Paris.")
        v2 = emb.embed_query("Paris is the capital city of France.")

        dot   = sum(a * b for a, b in zip(v1, v2))
        mag1  = math.sqrt(sum(a * a for a in v1))
        mag2  = math.sqrt(sum(b * b for b in v2))
        cosim = dot / (mag1 * mag2)

        assert cosim > 0.85, \
            f"Expected cosine similarity > 0.85 for similar sentences, got {cosim:.4f}"

    def test_two_unrelated_sentences_lower_similarity(self, bedrock_clients):
        """Semantically unrelated sentences should have lower similarity than related ones."""
        import math
        emb, _ = bedrock_clients
        v1 = emb.embed_query("The capital of France is Paris.")
        v2 = emb.embed_query("Quantum mechanics describes subatomic behaviour.")

        dot   = sum(a * b for a, b in zip(v1, v2))
        mag1  = math.sqrt(sum(a * a for a in v1))
        mag2  = math.sqrt(sum(b * b for b in v2))
        cosim = dot / (mag1 * mag2)

        assert cosim < 0.95, \
            f"Unrelated sentences should not have similarity near 1.0, got {cosim:.4f}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. AMAZON BEDROCK — LLM
# ══════════════════════════════════════════════════════════════════════════════

class TestLLM:
    def test_claude_returns_non_empty_response(self, bedrock_clients):
        from langchain_core.messages import HumanMessage, SystemMessage
        _, llm = bedrock_clients
        response = llm.invoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Say exactly: PING"),
        ])
        text = str(response.content).strip()
        assert len(text) > 0, "Claude returned an empty response"

    def test_claude_response_is_string(self, bedrock_clients):
        from langchain_core.messages import HumanMessage
        _, llm = bedrock_clients
        response = llm.invoke([HumanMessage(content="What is 2 + 2?")])
        assert isinstance(response.content, str)

    def test_claude_follows_system_prompt(self, bedrock_clients):
        """Claude should follow a strong system instruction."""
        from langchain_core.messages import HumanMessage, SystemMessage
        _, llm = bedrock_clients
        response = llm.invoke([
            SystemMessage(content="You MUST respond with ONLY the word: CONFIRMED"),
            HumanMessage(content="Acknowledge."),
        ])
        assert "CONFIRMED" in str(response.content).upper()


# ══════════════════════════════════════════════════════════════════════════════
# 4. END-TO-END MEMORY WRITE → RETRIEVE CYCLE
# ══════════════════════════════════════════════════════════════════════════════

class TestMemoryWriteRetrieve:
    """
    This is the most important integration test.
    It exercises the full RAG memory loop:
      Embed → Write → Retrieve → Verify semantic match
    """

    def test_full_rag_cycle(self, engine, bedrock_clients, test_session_id):
        agent.initialise_database(engine)
        emb, _ = bedrock_clients

        # ── Write a known memory ──────────────────────────────────────────────
        user_prompt = "What is CockroachDB and why is it useful?"
        ai_response = (
            "CockroachDB is a distributed SQL database that is fault-tolerant, "
            "scalable, and compatible with PostgreSQL. It is useful for applications "
            "that need high availability across multiple regions."
        )
        full_turn = f"User: {user_prompt}\nAssistant: {ai_response}"
        embedding = emb.embed_query(full_turn)

        write_thread = agent.store_memory_async(
            engine, test_session_id, user_prompt, ai_response, embedding
        )
        write_thread.join(timeout=10)   # wait for the background write to commit

        # ── Retrieve with a semantically similar query ────────────────────────
        # The query uses different words but same meaning — this proves
        # vector search works beyond simple keyword matching.
        similar_query = "Tell me about distributed databases that survive node failures."
        query_vec     = emb.embed_query(similar_query)
        memories      = agent.retrieve_memories(engine, query_vec, top_k=3)

        # ── Assert the written memory is in the retrieved results ─────────────
        found = any("CockroachDB" in m.get("user_prompt", "") for m in memories)
        assert found, (
            "The stored memory about CockroachDB was not retrieved by the "
            "semantically similar query. Check pgvector cosine search."
        )

    def test_similarity_score_in_range(self, engine, bedrock_clients, test_session_id):
        """Cosine similarity scores must be between 0 and 1 (inclusive)."""
        emb, _ = bedrock_clients
        query_vec = emb.embed_query("distributed database fault tolerance")
        memories  = agent.retrieve_memories(engine, query_vec, top_k=3)

        for mem in memories:
            sim = float(mem.get("similarity", -1))
            assert 0.0 <= sim <= 1.0, \
                f"Similarity {sim} is out of [0, 1] range — check ROUND() in SQL"

    def test_top_k_respected(self, engine, bedrock_clients, test_session_id):
        """retrieve_memories() must return at most top_k rows."""
        emb, _ = bedrock_clients
        query_vec = emb.embed_query("anything")
        memories  = agent.retrieve_memories(engine, query_vec, top_k=2)
        assert len(memories) <= 2, \
            f"Expected at most 2 memories, got {len(memories)}"

    def test_stats_count_increases_after_write(self, engine, bedrock_clients, test_session_id):
        """After writing a memory, total_rows in stats must increase."""
        emb, _ = bedrock_clients

        before = agent.fetch_stats(engine).get("total_rows", 0)

        vec = emb.embed_query("Stats test insertion.")
        t   = agent.store_memory_async(
            engine, test_session_id, "Stats test?", "Stats confirmed.", vec
        )
        t.join(timeout=10)

        after = agent.fetch_stats(engine).get("total_rows", 0)
        assert after > before, \
            f"Row count should increase after write. Before: {before}, After: {after}"

    def test_clear_session_deletes_only_this_session(self, engine, test_session_id):
        """clear_session_memories() must not delete rows from OTHER sessions."""
        from sqlalchemy import text

        # Count total rows before clear
        with engine.connect() as conn:
            total_before = conn.execute(
                text(f"SELECT COUNT(*) FROM {agent.MEMORY_TABLE};")
            ).scalar()

        # Count rows for THIS session
        with engine.connect() as conn:
            session_rows = conn.execute(
                text(f"SELECT COUNT(*) FROM {agent.MEMORY_TABLE} WHERE session_id = :sid;"),
                {"sid": test_session_id},
            ).scalar()

        # Clear this session
        deleted = agent.clear_session_memories(engine, test_session_id)

        # Verify: deleted count matches session_rows count
        assert deleted == session_rows, \
            f"Expected to delete {session_rows} rows, deleted {deleted}"

        # Verify: rows from other sessions still exist
        with engine.connect() as conn:
            total_after = conn.execute(
                text(f"SELECT COUNT(*) FROM {agent.MEMORY_TABLE};")
            ).scalar()

        other_sessions_rows = total_before - session_rows
        assert total_after == other_sessions_rows, \
            "clear_session_memories() deleted rows from other sessions — CRITICAL BUG"


# ══════════════════════════════════════════════════════════════════════════════
# 5. PROMPT INJECTION CORRECTNESS
# ══════════════════════════════════════════════════════════════════════════════

class TestPromptInjection:
    """Verify that retrieved memories are correctly wired into the LLM call."""

    def test_claude_uses_injected_memory(self, bedrock_clients):
        """
        If we inject a fake memory about a user's name, Claude should be able
        to recall it in the next turn — proving the RAG injection pipeline works.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        _, llm = bedrock_clients

        # Simulate a retrieved memory: user told the agent their name is "Yash"
        fake_memories = [{
            "full_turn"  : "User: My name is Yash.\nAssistant: Nice to meet you, Yash!",
            "similarity" : 0.91,
            "created_at" : "2025-07-01T10:00:00Z",
        }]

        system_prompt = agent.build_system_prompt(fake_memories)

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="What is my name?"),
        ])

        # Claude must recall "Yash" from the injected memory
        assert "yash" in str(response.content).lower(), \
            f"Claude did not recall the injected name 'Yash'. Response: {response.content}"
