"""
ingestion/github_ingester.py — Scheduled GitHub Ingester

Runs on a schedule to keep DevMind's knowledge base up to date.
Can be run manually or triggered as a cron job.

Usage:
    python -m ingestion.github_ingester --repo owner/repo
    python -m ingestion.github_ingester --repo owner/repo --prs 100 --issues 500
"""

from __future__ import annotations

import os
import sys
import argparse
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import sqlalchemy as sa
from langchain_aws import BedrockEmbeddings

from memory_store          import MemoryStore
from agents.code_agent     import CodeAgent
from agents.issue_agent    import IssueAgent
from agents.decision_agent import DecisionAgent


def run_ingestion(
    repo_name:   str,
    max_prs:     int = 50,
    max_commits: int = 100,
    max_issues:  int = 200,
) -> None:
    """
    Full ingestion pipeline for a GitHub repository.
    Indexes PRs, commits, issues, and extracts architectural decisions.
    """
    print(f"\n{'='*60}")
    print(f"  DevMind GitHub Ingester")
    print(f"  Repo:    {repo_name}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ── Setup ─────────────────────────────────────────────────────────────────
    db_uri  = os.environ["COCKROACH_DB_URI"]
    region  = os.environ.get("AWS_REGION", "us-east-1")
    gh_tok  = os.environ.get("GITHUB_TOKEN", "")

    if "cockroachdb+psycopg2" not in db_uri:
        db_uri = (
            db_uri
            .replace("postgresql+psycopg2://", "cockroachdb+psycopg2://", 1)
            .replace("postgresql://", "cockroachdb+psycopg2://", 1)
            .replace("postgres://",   "cockroachdb+psycopg2://", 1)
        )
    engine = sa.create_engine(db_uri, pool_pre_ping=True)

    embeddings = BedrockEmbeddings(
        model_id    = "amazon.titan-embed-text-v2:0",
        region_name = region,
    )
    store = MemoryStore(engine, embeddings)

    code_agent     = CodeAgent(store, gh_tok)
    issue_agent    = IssueAgent(store, gh_tok)
    decision_agent = DecisionAgent(store, github_token=gh_tok)

    # ── 1. Index PRs + Commits ────────────────────────────────────────────────
    print(f"[1/3] Indexing code (PRs + commits)...")
    code_counts = code_agent.index_repo(
        repo_name,
        max_prs     = max_prs,
        max_commits = max_commits,
    )
    print(
        f"      ✓ {code_counts['prs_indexed']} PRs  |  "
        f"{code_counts['commits_indexed']} commits  |  "
        f"{code_counts['errors']} errors"
    )

    # ── 2. Index Issues ───────────────────────────────────────────────────────
    print(f"\n[2/3] Indexing issues...")
    issue_counts = issue_agent.index_repo_issues(
        repo_name,
        state      = "all",
        max_issues = max_issues,
    )
    print(
        f"      ✓ {issue_counts['indexed']} issues  |  "
        f"{issue_counts['errors']} errors"
    )

    # ── 3. Extract Decisions ──────────────────────────────────────────────────
    print(f"\n[3/3] Extracting architectural decisions from PRs...")
    decision_counts = decision_agent.extract_and_store_decisions_from_prs(
        repo_name,
        max_prs = max_prs,
    )
    print(
        f"      ✓ {decision_counts['decisions_found']} decisions found  |  "
        f"scanned {decision_counts['prs_scanned']} PRs"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    stats = store.get_stats()
    print(f"\n{'='*60}")
    print(f"  Ingestion Complete!")
    print(f"  Knowledge Base Totals:")
    print(f"    Conversations: {stats.get('conversations', 0):,}")
    print(f"    Code chunks:   {stats.get('code', 0):,}")
    print(f"    Issues:        {stats.get('issues', 0):,}")
    print(f"    Decisions:     {stats.get('decisions', 0):,}")
    print(f"{'='*60}\n")
    print("DevMind is ready! Open Claude Desktop and start asking questions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DevMind GitHub Ingester — index a GitHub repo into the knowledge base"
    )
    parser.add_argument("--repo",    required=True, help="GitHub repo (e.g. owner/myrepo)")
    parser.add_argument("--prs",     type=int, default=50,  help="Max PRs to index")
    parser.add_argument("--commits", type=int, default=100, help="Max commits to index")
    parser.add_argument("--issues",  type=int, default=200, help="Max issues to index")
    args = parser.parse_args()

    run_ingestion(
        repo_name   = args.repo,
        max_prs     = args.prs,
        max_commits = args.commits,
        max_issues  = args.issues,
    )
