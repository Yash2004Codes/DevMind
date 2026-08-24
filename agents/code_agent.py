"""
agents/code_agent.py — Code Agent

Responsible for:
  • Indexing GitHub PRs, commits, and file diffs into the vector store
  • Searching code chunks semantically
  • Answering "who owns this file?", "what changed in module X?"

Exposes functions that are registered as MCP tools in mcp_server.py.
"""

from __future__ import annotations

import os
from typing import Optional

from memory_store import MemoryStore


class CodeAgent:
    """
    Interfaces with the GitHub API to ingest code context,
    and the MemoryStore code domain for semantic search.
    """

    def __init__(self, store: MemoryStore, github_token: Optional[str] = None):
        self.store         = store
        self.github_token  = github_token or os.getenv("GITHUB_TOKEN", "")
        self._gh           = None  # lazy-loaded PyGithub client

    def _get_github(self):
        """Lazily initialize the PyGithub client."""
        if self._gh is None:
            try:
                from github import Github
                self._gh = Github(self.github_token) if self.github_token else Github()
            except ImportError:
                raise RuntimeError(
                    "PyGithub is not installed. Run: pip install PyGithub"
                )
        return self._gh

    # ── Search ───────────────────────────────────────────────────────────────

    def search_code(
        self,
        query: str,
        repo: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Semantic search across indexed code chunks, PRs, and commits.

        Parameters
        ----------
        query : Natural language description of what you're looking for
        repo  : Optional filter by repo name (e.g. "owner/myrepo")
        top_k : Number of results

        Returns
        -------
        List of matching code chunks with file path, author, PR number, similarity.

        Example
        -------
        search_code("Redis caching implementation")
        search_code("authentication middleware", repo="myorg/api-service")
        """
        filters = {"repo": repo} if repo else None
        results = self.store.search(
            query, domain="code", top_k=top_k, filters=filters
        )
        return [
            {
                "file_path":  r.get("file_path", ""),
                "repo":       r.get("repo", ""),
                "author":     r.get("author", ""),
                "pr_number":  r.get("pr_number"),
                "chunk_type": r.get("chunk_type", ""),
                "preview":    r.get("chunk_text", "")[:300],
                "similarity": float(r.get("similarity", 0)),
                "indexed_at": str(r.get("indexed_at", "")),
            }
            for r in results if "error" not in r
        ]

    def get_file_owners(
        self,
        file_path: str,
        repo: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Find who has worked on a specific file based on indexed commits and PRs.

        Parameters
        ----------
        file_path : Path of the file (e.g. "src/auth/middleware.py")
        repo      : Optional repo filter
        top_k     : Max contributors to return

        Returns
        -------
        List of {author, pr_number, chunk_type, indexed_at}
        """
        query   = f"changes to file {file_path}"
        filters = {"repo": repo} if repo else None
        results = self.store.search(
            query, domain="code", top_k=top_k * 3, filters=filters
        )
        # Filter by file path prefix match
        owners: dict[str, dict] = {}
        for r in results:
            if file_path.lower() in r.get("file_path", "").lower():
                author = r.get("author", "unknown")
                if author not in owners:
                    owners[author] = {
                        "author":     author,
                        "pr_number":  r.get("pr_number"),
                        "file_path":  r.get("file_path"),
                        "indexed_at": str(r.get("indexed_at", "")),
                    }
        return list(owners.values())[:top_k]

    # ── Ingestion ────────────────────────────────────────────────────────────

    def index_repo(
        self,
        repo_name: str,
        max_prs: int = 50,
        max_commits: int = 100,
    ) -> dict:
        """
        Index a GitHub repository's PRs and recent commits into the vector store.

        Parameters
        ----------
        repo_name   : Full repo name (e.g. "octocat/Hello-World")
        max_prs     : Maximum number of PRs to index
        max_commits : Maximum number of commits to index

        Returns
        -------
        dict with counts: {prs_indexed, commits_indexed, errors}

        Notes
        -----
        This can take 1-5 minutes for large repos.
        Duplicate chunks are ignored via ON CONFLICT DO NOTHING.
        """
        gh      = self._get_github()
        repo    = gh.get_repo(repo_name)
        counts  = {"prs_indexed": 0, "commits_indexed": 0, "errors": 0}

        # ── Index PRs ──────────────────────────────────────────────────────
        try:
            pulls = repo.get_pulls(state="all", sort="updated", direction="desc")
            for pr in list(pulls)[:max_prs]:
                try:
                    # Index PR body/description
                    pr_text = (
                        f"PR #{pr.number}: {pr.title}\n"
                        f"Author: {pr.user.login}\n"
                        f"Branch: {pr.head.ref} → {pr.base.ref}\n\n"
                        f"{pr.body or 'No description'}"
                    )
                    self.store.store_code_chunk(
                        repo      = repo_name,
                        file_path = f"__pr__/{pr.number}",
                        chunk_text= pr_text,
                        metadata  = {
                            "author":     pr.user.login,
                            "pr_number":  pr.number,
                            "commit_sha": pr.head.sha[:8],
                            "chunk_type": "pr_body",
                        },
                    )

                    # Index changed files list
                    files_text = (
                        f"PR #{pr.number} changed files:\n"
                        + "\n".join(
                            f.filename
                            for f in list(pr.get_files())[:30]
                        )
                    )
                    self.store.store_code_chunk(
                        repo      = repo_name,
                        file_path = f"__pr__/{pr.number}/files",
                        chunk_text= files_text,
                        metadata  = {
                            "author":     pr.user.login,
                            "pr_number":  pr.number,
                            "commit_sha": pr.head.sha[:8],
                            "chunk_type": "pr_files",
                        },
                    )
                    counts["prs_indexed"] += 1
                except Exception:
                    counts["errors"] += 1
        except Exception as e:
            counts["errors"] += 1

        # ── Index commits ──────────────────────────────────────────────────
        try:
            commits = repo.get_commits()
            for commit in list(commits)[:max_commits]:
                try:
                    msg    = commit.commit.message
                    author = (
                        commit.author.login
                        if commit.author
                        else commit.commit.author.name
                    )
                    commit_text = (
                        f"Commit {commit.sha[:8]} by {author}:\n{msg}"
                    )
                    self.store.store_code_chunk(
                        repo      = repo_name,
                        file_path = f"__commit__/{commit.sha[:8]}",
                        chunk_text= commit_text,
                        metadata  = {
                            "author":     author,
                            "commit_sha": commit.sha[:8],
                            "chunk_type": "commit_msg",
                        },
                    )
                    counts["commits_indexed"] += 1
                except Exception:
                    counts["errors"] += 1
        except Exception:
            counts["errors"] += 1

        return counts
