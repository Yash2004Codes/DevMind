"""
setup_db.py — DevMind Database Initialisation Script

Run this ONCE before starting the MCP server for the first time.
Creates ALL tables (original agent_memory + 4 new DevMind tables).

Usage:
    python setup_db.py

Only requires COCKROACH_DB_URI — does NOT need AWS or GitHub credentials.
"""

import os
import sys
import pathlib
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box as rich_box

load_dotenv()
console = Console()


def get_engine(db_uri: str):
    import sqlalchemy as sa
    # Use cockroachdb dialect — fixes version string parsing for CockroachDB v22+
    if "cockroachdb+psycopg2" not in db_uri:
        db_uri = (
            db_uri
            .replace("postgresql+psycopg2://", "cockroachdb+psycopg2://", 1)
            .replace("postgresql://", "cockroachdb+psycopg2://", 1)
            .replace("postgres://",   "cockroachdb+psycopg2://", 1)
        )
    return sa.create_engine(db_uri, pool_pre_ping=True)


def run_schema(engine) -> list[tuple[str, str]]:
    """
    Execute db/schema.sql against CockroachDB.
    Returns list of (statement_preview, status) tuples.
    """
    import sqlalchemy as sa
    schema_path = pathlib.Path(__file__).parent / "db" / "schema.sql"
    schema_sql  = schema_path.read_text(encoding="utf-8")

    results = []
    statements = [s.strip() for s in schema_sql.split(";") if s.strip()]

    with engine.begin() as conn:
        for stmt in statements:
            # Skip comment-only lines
            lines = [l for l in stmt.splitlines() if not l.strip().startswith("--")]
            clean = " ".join(lines).strip()
            if not clean:
                continue
            preview = clean[:55] + "…"
            try:
                conn.execute(sa.text(clean))
                results.append((preview, "OK"))
            except Exception as e:
                err = str(e)[:60]
                # "already exists" errors are expected on re-runs — treat as OK
                if "already exists" in err.lower():
                    results.append((preview, "already exists"))
                else:
                    results.append((preview, f"! {err}"))

    return results


def print_results(results: list[tuple[str, str]]) -> None:
    tbl = Table(box=rich_box.ROUNDED, border_style="cyan", show_lines=False)
    tbl.add_column("Statement",  style="dim",   max_width=60)
    tbl.add_column("Status",     style="green", width=30)
    for stmt, status in results:
        color = "green" if "OK" in status or "exists" in status else "yellow"
        tbl.add_row(stmt, f"[{color}]{status}[/{color}]")
    console.print(tbl)


def main() -> None:
    db_uri = os.getenv("COCKROACH_DB_URI")
    if not db_uri:
        console.print(Panel(
            "[red]COCKROACH_DB_URI is not set.[/red]\n"
            "Copy [bold].env.example[/bold] → [bold].env[/bold] and fill it in.",
            title="[red]Error[/red]",
            border_style="red",
        ))
        sys.exit(1)

    console.print(Panel.fit(
        "[bold cyan]DevMind[/bold cyan] — Database Initialisation\n"
        "[dim]Creates all tables: agent_memory, code_chunks, gh_issues, arch_decisions, knowledge_edges[/dim]",
        border_style="cyan",
    ))
    console.print()

    with console.status("[bold green]Connecting to CockroachDB…", spinner="dots"):
        engine = get_engine(db_uri)

    with console.status("[bold green]Running schema migrations…", spinner="dots"):
        results = run_schema(engine)

    print_results(results)

    console.print(
        "\n[bold green]OK[/bold green] Database ready.\n"
        "Next steps:\n"
        "  [cyan]1.[/cyan] python mcp_server.py --ingest        [dim]# index your GitHub repo[/dim]\n"
        "  [cyan]2.[/cyan] Configure Claude Desktop             [dim]# see CLAUDE_DESKTOP_SETUP.md[/dim]\n"
        "  [cyan]3.[/cyan] Ask DevMind questions in Claude Desktop! 🚀"
    )


if __name__ == "__main__":
    main()
