"""
agents/memory_agent.py — Memory Agent

Responsible for:
  • Storing and retrieving conversational memory
  • Cross-session semantic search
  • Session statistics and management

Exposes functions that are registered as MCP tools in mcp_server.py.
"""

from __future__ import annotations

from memory_store import MemoryStore


class MemoryAgent:
    """
    Wraps the MemoryStore conversation domain.
    Provides clean, tool-friendly methods for the MCP server.
    """

    def __init__(self, store: MemoryStore):
        self.store = store

    def search_memories(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search past conversation history semantically.

        Parameters
        ----------
        query : What to search for (natural language)
        top_k : Number of results to return

        Returns
        -------
        List of past conversation turns ranked by relevance.
        Each item has: user_prompt, ai_response, similarity, created_at
        """
        results = self.store.search(query, domain="conversations", top_k=top_k)
        # Clean up fields for MCP output
        cleaned = []
        for r in results:
            if "error" in r:
                continue
            cleaned.append({
                "user_prompt":  r.get("user_prompt", ""),
                "ai_response":  r.get("ai_response", "")[:500] + "…"
                                if len(r.get("ai_response", "")) > 500
                                else r.get("ai_response", ""),
                "similarity":   float(r.get("similarity", 0)),
                "stored_at":    str(r.get("created_at", "")),
                "session_id":   str(r.get("session_id", ""))[:8] + "…",
            })
        return cleaned

    def get_stats(self) -> dict:
        """
        Return memory store statistics.

        Returns
        -------
        dict with keys: conversations, code, issues, decisions (row counts)
        """
        return self.store.get_stats()

    def cross_domain_search(
        self,
        query: str,
        top_k_per_domain: int = 3,
    ) -> dict:
        """
        Search ALL memory domains (conversations, code, issues, decisions)
        simultaneously for a given query.

        Parameters
        ----------
        query            : Natural language search query
        top_k_per_domain : Results per domain (default 3)

        Returns
        -------
        dict keyed by domain name, each containing a list of relevant results.
        """
        raw = self.store.cross_domain_search(query, top_k_per_domain)
        # Summarize for readable MCP output
        summary = {}
        for domain, hits in raw.items():
            summary[domain] = [
                {
                    "similarity": float(h.get("similarity", 0)),
                    "text": _extract_text(domain, h)[:300],
                }
                for h in hits if "error" not in h
            ]
        return summary


def _extract_text(domain: str, row: dict) -> str:
    """Pull the most human-readable text field from any domain row."""
    if domain == "conversations":
        return row.get("full_turn", row.get("user_prompt", ""))
    if domain == "code":
        return row.get("chunk_text", "")
    if domain == "issues":
        return f"Issue #{row.get('issue_number')}: {row.get('title', '')}\n{row.get('body', '')}"
    if domain == "decisions":
        return f"Decision: {row.get('title', '')}\nRationale: {row.get('rationale', '')}"
    return str(row)
