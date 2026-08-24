"""
agents/decision_agent.py — Decision Agent

Responsible for:
  • Extracting architectural decisions from PRs and issues
  • Storing decisions in the vector store
  • Answering "why did we choose X over Y?" questions
  • Summarizing decisions made in a time period

Decision extraction uses Claude to read PR bodies/commit messages
and identify decision-like content ("we chose X because Y").
"""

from __future__ import annotations

import os
from typing import Optional

from memory_store import MemoryStore


# Keywords that signal an architectural decision was made
DECISION_SIGNALS = [
    "decided to", "we chose", "switched to", "replaced", "deprecated",
    "abandoned", "migrated to", "refactored", "we will use", "going with",
    "instead of", "we avoid", "not using", "removed", "reason:", "rationale:",
    "because of", "trade-off", "trade off", "adr", "architecture decision",
]


class DecisionAgent:
    """
    Manages architectural decisions extracted from GitHub PRs and issues,
    and supports semantic search over the decision log.
    """

    def __init__(
        self,
        store: MemoryStore,
        llm=None,
        github_token: Optional[str] = None,
    ):
        self.store        = store
        self.llm          = llm  # ChatBedrock instance, used for extraction
        self.github_token = github_token or os.getenv("GITHUB_TOKEN", "")
        self._gh          = None

    def _get_github(self):
        if self._gh is None:
            try:
                from github import Github
                self._gh = Github(self.github_token) if self.github_token else Github()
            except ImportError:
                raise RuntimeError("Run: pip install PyGithub")
        return self._gh

    # ── Search ───────────────────────────────────────────────────────────────

    def search_decisions(
        self,
        query: str,
        repo: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Find architectural decisions relevant to a query.

        Parameters
        ----------
        query : Natural language question about a past decision
                e.g. "why did we stop using MongoDB?"
                     "what database did we choose and why?"
        repo  : Optional repo filter
        top_k : Number of results

        Returns
        -------
        List of decisions with title, rationale, source, author, similarity.

        Example
        -------
        search_decisions("Redis caching decision")
        search_decisions("why PostgreSQL instead of MySQL")
        """
        filters = {"repo": repo} if repo else None
        results = self.store.search(
            query, domain="decisions", top_k=top_k, filters=filters
        )
        return [
            {
                "title":       r.get("title", ""),
                "rationale":   r.get("rationale", "")[:500],
                "source_type": r.get("source_type", ""),
                "source_ref":  r.get("source_ref", ""),
                "author":      r.get("author", ""),
                "repo":        r.get("repo", ""),
                "similarity":  float(r.get("similarity", 0)),
                "recorded_at": str(r.get("created_at", "")),
            }
            for r in results if "error" not in r
        ]

    # ── Extraction & Ingestion ────────────────────────────────────────────────

    def extract_and_store_decisions_from_prs(
        self,
        repo_name: str,
        max_prs: int = 30,
    ) -> dict:
        """
        Scan recent PRs for architectural decision signals and store them.

        Uses keyword heuristics (no LLM cost) to detect decision-like text,
        then stores them with the PR as the source reference.

        Parameters
        ----------
        repo_name : Full repo name (e.g. "owner/myrepo")
        max_prs   : Maximum PRs to scan

        Returns
        -------
        dict with: {decisions_found, prs_scanned, errors}
        """
        gh     = self._get_github()
        repo   = gh.get_repo(repo_name)
        counts = {"decisions_found": 0, "prs_scanned": 0, "errors": 0}

        pulls = repo.get_pulls(state="all", sort="updated", direction="desc")
        for pr in list(pulls)[:max_prs]:
            counts["prs_scanned"] += 1
            try:
                body   = (pr.body or "").lower()
                title  = pr.title.lower()
                text   = title + " " + body

                # Check if this PR contains decision signals
                if not any(sig in text for sig in DECISION_SIGNALS):
                    continue

                # Use full body as rationale (truncated)
                rationale = (
                    f"PR #{pr.number}: {pr.title}\n\n"
                    + (pr.body or "No description")
                )[:2000]

                self.store.store_decision(
                    repo        = repo_name,
                    title       = f"[PR #{pr.number}] {pr.title}",
                    rationale   = rationale,
                    source_type = "pr",
                    source_ref  = str(pr.number),
                    author      = pr.user.login if pr.user else "",
                )
                counts["decisions_found"] += 1
            except Exception:
                counts["errors"] += 1

        return counts

    def store_manual_decision(
        self,
        repo: str,
        title: str,
        rationale: str,
        author: str = "manual",
    ) -> dict:
        """
        Manually log an architectural decision directly into the decision store.

        Parameters
        ----------
        repo      : Repository this decision applies to
        title     : Short title of the decision (e.g. "Chose PostgreSQL over MongoDB")
        rationale : Full reasoning / trade-off explanation
        author    : Who made or recorded this decision

        Returns
        -------
        dict with status confirmation.

        Example
        -------
        store_manual_decision(
            repo="myorg/api",
            title="Switched from REST to GraphQL",
            rationale="REST endpoints were multiplying and clients needed flexible queries. GraphQL reduces over-fetching by 60%.",
            author="alice"
        )
        """
        try:
            self.store.store_decision(
                repo        = repo,
                title       = title,
                rationale   = rationale,
                source_type = "manual",
                source_ref  = "manual_entry",
                author      = author,
            )
            return {"status": "ok", "decision": title}
        except Exception as e:
            return {"status": "error", "message": str(e)}
