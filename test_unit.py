"""
test_unit.py — Unit Tests (NO credentials required)
════════════════════════════════════════════════════
Tests every pure-Python function in agent.py using mocks.
No CockroachDB URI, no AWS keys — runs completely offline.

Run:
    python -m pytest test_unit.py -v
    python -m pytest test_unit.py -v --tb=short    # shorter tracebacks
"""

import pytest
import threading
from unittest.mock import MagicMock, patch, call
from types import SimpleNamespace

# ── Import functions under test ───────────────────────────────────────────────
# We patch os.environ before importing agent so validate_environment() doesn't
# immediately sys.exit() during module load.
import os
os.environ.setdefault("COCKROACH_DB_URI",       "postgresql://fake:fake@localhost/fake")
os.environ.setdefault("AWS_REGION",             "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID",      "FAKEID")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY",  "FAKESECRET")

import agent  # noqa: E402  (import after env patching)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ENVIRONMENT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateEnvironment:
    """validate_environment() must catch missing vars and return a config dict."""

    def test_returns_dict_when_all_vars_present(self):
        config = agent.validate_environment()
        assert "db_uri"     in config
        assert "aws_region" in config

    def test_exits_when_var_missing(self):
        """If any required env var is absent the function must call sys.exit(1)."""
        original = os.environ.pop("AWS_REGION", None)
        try:
            with pytest.raises(SystemExit) as exc_info:
                agent.validate_environment()
            assert exc_info.value.code == 1
        finally:
            # Restore so other tests are not affected
            if original:
                os.environ["AWS_REGION"] = original


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATABASE URI NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

class TestGetEngine:
    """get_engine() must normalise any valid CockroachDB URI format."""

    @patch("agent.sa.create_engine")
    def test_postgres_scheme_normalised(self, mock_create):
        mock_create.return_value = MagicMock()
        agent.get_engine("postgres://user:pass@host:26257/db")
        called_uri = mock_create.call_args[0][0]
        assert called_uri.startswith("postgresql+psycopg2://"), \
            f"Expected psycopg2 scheme, got: {called_uri}"

    @patch("agent.sa.create_engine")
    def test_postgresql_scheme_normalised(self, mock_create):
        mock_create.return_value = MagicMock()
        agent.get_engine("postgresql://user:pass@host:26257/db")
        called_uri = mock_create.call_args[0][0]
        assert "postgresql+psycopg2://" in called_uri

    @patch("agent.sa.create_engine")
    def test_already_correct_scheme_unchanged(self, mock_create):
        mock_create.return_value = MagicMock()
        uri = "postgresql+psycopg2://user:pass@host:26257/db"
        agent.get_engine(uri)
        called_uri = mock_create.call_args[0][0]
        assert called_uri == uri

    @patch("agent.sa.create_engine")
    def test_pool_pre_ping_enabled(self, mock_create):
        mock_create.return_value = MagicMock()
        agent.get_engine("postgres://a:b@c:26257/d")
        kwargs = mock_create.call_args[1]
        assert kwargs.get("pool_pre_ping") is True


# ══════════════════════════════════════════════════════════════════════════════
# 3. SYSTEM PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildSystemPrompt:
    """build_system_prompt() must produce valid text for both empty and non-empty memory."""

    def test_empty_memories_returns_no_context_message(self):
        prompt = agent.build_system_prompt([])
        assert "No semantically similar" in prompt
        assert "Proceed without historical context" in prompt

    def test_non_empty_memories_included(self):
        memories = [
            {
                "full_turn"  : "User: What is Python?\nAssistant: A programming language.",
                "similarity" : 0.92,
                "created_at" : "2025-01-01T10:00:00Z",
            }
        ]
        prompt = agent.build_system_prompt(memories)
        assert "What is Python?" in prompt
        assert "0.9200" in prompt   # similarity score formatted to 4 dp
        assert "Memory 1" in prompt

    def test_multiple_memories_all_included(self):
        memories = [
            {"full_turn": f"User: Q{i}\nAssistant: A{i}", "similarity": 0.9 - i * 0.1,
             "created_at": "2025-01-01"}
            for i in range(3)
        ]
        prompt = agent.build_system_prompt(memories)
        assert "Memory 1" in prompt
        assert "Memory 2" in prompt
        assert "Memory 3" in prompt

    def test_long_turn_truncated(self):
        long_text = "X" * 2000
        memories  = [{"full_turn": long_text, "similarity": 0.8, "created_at": "now"}]
        prompt    = agent.build_system_prompt(memories)
        # The turn must be truncated — prompt must not contain the full 2 000 chars verbatim
        assert "…[truncated]" in prompt

    def test_prompt_always_starts_with_role_definition(self):
        prompt = agent.build_system_prompt([])
        assert prompt.startswith("You are a helpful")

    def test_prompt_contains_anti_hallucination_instruction(self):
        prompt = agent.build_system_prompt([])
        assert "hallucinate" in prompt.lower()


# ══════════════════════════════════════════════════════════════════════════════
# 4. VECTOR SERIALISATION
# ══════════════════════════════════════════════════════════════════════════════

class TestVectorSerialisation:
    """
    The inline vector literal format must produce valid pgvector SQL syntax.
    Format expected by CockroachDB: '[0.12345678,−0.87654321,…]'
    """

    def _make_vec_str(self, values: list[float]) -> str:
        """Replicate the serialisation logic used in agent.py."""
        return "[" + ",".join(f"{v:.8f}" for v in values) + "]"

    def test_correct_bracket_format(self):
        vec = [0.1, 0.2, 0.3]
        result = self._make_vec_str(vec)
        assert result.startswith("[")
        assert result.endswith("]")

    def test_8_decimal_places(self):
        vec = [1.0 / 3.0]
        result = self._make_vec_str(vec)
        # 0.33333333 — exactly 8 decimal places
        assert "0.33333333" in result

    def test_negative_values_preserved(self):
        vec = [-0.5, 0.5]
        result = self._make_vec_str(vec)
        assert "-0.50000000" in result

    def test_full_dimension_vector(self):
        vec = [0.001] * agent.VECTOR_DIM   # 1 536-dim vector
        result = self._make_vec_str(vec)
        count  = result.count(",")
        assert count == agent.VECTOR_DIM - 1, \
            f"Expected {agent.VECTOR_DIM - 1} commas, got {count}"


# ══════════════════════════════════════════════════════════════════════════════
# 5. MEMORY RETRIEVAL (mocked DB)
# ══════════════════════════════════════════════════════════════════════════════

class TestRetrieveMemories:
    """retrieve_memories() must handle empty tables and normal results gracefully."""

    def _make_engine(self, rows: list[dict]) -> MagicMock:
        """Build a mock SQLAlchemy engine that returns ``rows`` on execute."""
        mock_row_list = [SimpleNamespace(_mapping=r) for r in rows]

        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter(mock_row_list))

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn
        return mock_engine

    def test_empty_table_returns_empty_list(self):
        engine = self._make_engine([])
        result = agent.retrieve_memories(engine, [0.0] * agent.VECTOR_DIM)
        assert result == []

    def test_rows_returned_as_dicts(self):
        row = {
            "user_prompt": "Hello",
            "ai_response": "Hi there!",
            "full_turn"  : "User: Hello\nAssistant: Hi there!",
            "created_at" : "2025-01-01",
            "similarity" : 0.95,
        }
        engine = self._make_engine([row])
        result = agent.retrieve_memories(engine, [0.0] * agent.VECTOR_DIM)
        assert len(result) == 1
        assert result[0]["user_prompt"] == "Hello"

    def test_db_exception_returns_empty_list(self):
        """DB errors must not crash the agent — return [] instead."""
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("connection refused")
        result = agent.retrieve_memories(mock_engine, [0.0] * agent.VECTOR_DIM)
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# 6. ASYNC MEMORY WRITE
# ══════════════════════════════════════════════════════════════════════════════

class TestStoreMemoryAsync:
    """store_memory_async() must start a background thread and return it."""

    def test_returns_thread(self):
        mock_engine = MagicMock()
        mock_conn   = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__  = MagicMock(return_value=False)
        mock_engine.begin.return_value = mock_conn

        thread = agent.store_memory_async(
            engine      = mock_engine,
            session_id  = "test-session-id",
            user_prompt = "What is vector search?",
            ai_response = "It is ANN retrieval in high-dimensional space.",
            embedding   = [0.1] * agent.VECTOR_DIM,
        )
        assert isinstance(thread, threading.Thread)

    def test_thread_is_daemon(self):
        mock_engine = MagicMock()
        mock_conn   = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__  = MagicMock(return_value=False)
        mock_engine.begin.return_value = mock_conn

        thread = agent.store_memory_async(
            mock_engine, "sid", "prompt", "response", [0.0] * agent.VECTOR_DIM
        )
        assert thread.daemon is True, "Write thread must be a daemon thread"

    def test_write_executes_insert(self):
        mock_engine = MagicMock()
        mock_conn   = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__  = MagicMock(return_value=False)
        mock_engine.begin.return_value = mock_conn

        thread = agent.store_memory_async(
            mock_engine, "sid", "test prompt", "test response", [0.5] * agent.VECTOR_DIM
        )
        thread.join(timeout=3)  # wait for write to complete
        assert mock_conn.execute.called, "INSERT was never executed"


# ══════════════════════════════════════════════════════════════════════════════
# 7. CONSTANTS SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

class TestConstants:
    """Verify model IDs and dimension constants haven't been accidentally changed."""

    def test_vector_dim_is_1536(self):
        assert agent.VECTOR_DIM == 1536, \
            "VECTOR_DIM must be 1536 to match Titan Embeddings V2"

    def test_embed_model_id(self):
        assert "titan-embed-text-v2" in agent.EMBED_MODEL_ID

    def test_chat_model_id(self):
        assert "claude-3-haiku" in agent.CHAT_MODEL_ID

    def test_memory_table_name(self):
        assert agent.MEMORY_TABLE == "agent_memory"

    def test_top_k_is_positive(self):
        assert agent.TOP_K_MEMORIES > 0


# ══════════════════════════════════════════════════════════════════════════════
# 8. FETCH STATS (mocked DB)
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchStats:
    def test_returns_dict_with_expected_keys(self):
        fake_row = {
            "total_rows"     : 42,
            "total_sessions" : 3,
            "oldest"         : "2025-01-01",
            "newest"         : "2025-07-24",
        }
        mock_result  = MagicMock()
        mock_result.mappings.return_value.first.return_value = fake_row

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__  = MagicMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        stats = agent.fetch_stats(mock_engine)
        assert stats["total_rows"]      == 42
        assert stats["total_sessions"]  == 3

    def test_db_exception_returns_empty_dict(self):
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("timeout")
        result = agent.fetch_stats(mock_engine)
        assert result == {}


# ══════════════════════════════════════════════════════════════════════════════
# 9. CLEAR SESSION MEMORIES (mocked DB)
# ══════════════════════════════════════════════════════════════════════════════

class TestClearSessionMemories:
    def test_returns_rowcount(self):
        mock_result = MagicMock()
        mock_result.rowcount = 7

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__  = MagicMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_conn

        deleted = agent.clear_session_memories(mock_engine, "test-session")
        assert deleted == 7

    def test_exception_returns_zero(self):
        mock_engine = MagicMock()
        mock_engine.begin.side_effect = Exception("db error")
        result = agent.clear_session_memories(mock_engine, "sid")
        assert result == 0
