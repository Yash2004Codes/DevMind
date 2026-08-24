"""
agents/issue_agent.py — Issue Agent

Responsible for:
  • Indexing GitHub Issues into the vector store
  • Semantic search over issues
  • Finding similar bugs/issues to a given description
  • Summarizing open vs. closed issue patterns
"""

from __future__ import annotations

import os
from typing import Optional

from memory_store import MemoryStore


class IssueAgent:
    """
    Interfaces with GitHub Issues API and the MemoryStore issues domain.
    """

    def __init__(self, store: MemoryStore, github_token: Optional[str] = None):
        self.store        = store
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

    def search_issues(
        self,
        query: str,
        repo: Optional[str] = None,
        state: Optional[str] = None,  # 'open' | 'closed' | None (all)
        top_k: int = 5,
    ) -> list[dict]:
        """
        Semantically search GitHub Issues for anything matching the query.

        Parameters
        ----------
        query : Natural language description (e.g. "memory leak in auth service")
        repo  : Optional filter by repo name (e.g. "owner/myrepo")
        state : Optional filter by issue state ('open' or 'closed')
        top_k : Number of results

        Returns
        -------
        List of matching issues with title, state, labels, author, similarity.

        Example
        -------
        search_issues("Redis connection timeout errors")
        search_issues("login fails on mobile", state="closed")
        """
        filters = {}
        if repo:
            filters["repo"] = repo
        if state:
            filters["state"] = state

        results = self.store.search(
            query, domain="issues", top_k=top_k,
            filters=filters if filters else None
        )
        return [
            {
                "issue_number": r.get("issue_number"),
                "title":        r.get("title", ""),
                "state":        r.get("state", ""),
                "labels":       r.get("labels", ""),
                "author":       r.get("author", ""),
                "repo":         r.get("repo", ""),
                "body_preview": r.get("body", "")[:300],
                "similarity":   float(r.get("similarity", 0)),
            }
            for r in results if "error" not in r
        ]

    def find_similar_issues(
        self,
        description: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Given a bug description or stack trace, find the most similar
        past issues (open or closed).

        Parameters
        ----------
        description : Bug description, error message, or stack trace
        top_k       : Number of similar issues to return

        Returns
        -------
        List of similar issues ranked by cosine similarity.
        """
        return self.search_issues(description, top_k=top_k)

    # ── Ingestion ────────────────────────────────────────────────────────────

    def index_repo_issues(
        self,
        repo_name: str,
        state: str = "all",
        max_issues: int = 200,
    ) -> dict:
        """
        Index all GitHub Issues from a repository into the vector store.

        Parameters
        ----------
        repo_name  : Full repo name (e.g. "owner/myrepo")
        state      : 'open', 'closed', or 'all'
        max_issues : Maximum number of issues to index

        Returns
        -------
        dict with: {indexed, errors}
        """
        gh     = self._get_github()
        repo   = gh.get_repo(repo_name)
        counts = {"indexed": 0, "errors": 0}

        try:
            issues = repo.get_issues(state=state, sort="updated", direction="desc")
            for issue in list(issues)[:max_issues]:
                # Skip pull requests (GitHub API returns them as issues too)
                if issue.pull_request:
                    continue
                try:
                    labels = [lbl.name for lbl in issue.labels]
                    self.store.store_issue(
                        repo         = repo_name,
                        issue_number = issue.number,
                        title        = issue.title,
                        body         = issue.body or "",
                        state        = issue.state,
                        labels       = labels,
                        author       = issue.user.login if issue.user else "",
                    )
                    counts["indexed"] += 1
                except Exception:
                    counts["errors"] += 1
        except Exception as e:
            counts["errors"] += 1

        return counts

    def get_issue_summary(
        self,
        repo: Optional[str] = None,
    ) -> dict:
        """
        Return a high-level summary of issues in the knowledge base.

        Parameters
        ----------
        repo : Optional filter by repo

        Returns
        -------
        dict with open_count, closed_count from stored data.
        """
        stats = self.store.get_stats()
        return {
            "total_indexed_issues": stats.get("issues", 0),
            "note": "Use search_issues() to find specific issues semantically.",
        }
