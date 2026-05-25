"""Typer CLI — default opens Confluence demo in browser; use `pipeline` for terminal runs."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import webbrowser
from pathlib import Path

import typer
from langgraph.types import Command

from reqs_agent_demo.agent.graph import compile_demo_graph, initialise_state
from reqs_agent_demo.pipeline_runner import build_graph_deps_for_run, close_graph_clients

app = typer.Typer(invoke_without_command=True, help="Reqs-agent demo launcher.")


@app.callback(invoke_without_command=True)
def _default_hub(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        open_hub()


@app.command("open-hub")
def open_hub() -> None:
    """Browse the mocked Confluence PRD hub (needs `uv run mock-dev`)."""

    base = os.environ.get("MOCK_BASE_URL", "http://127.0.0.1:8877").rstrip("/")
    url = f"{base}/"
    typer.echo(f"Opening mocked Confluence page: {url}")
    typer.echo("Ensure `uv run mock-dev` runs in another shell.")
    webbrowser.open(url)


@app.command("pipeline")
def pipeline(
    page_id: str = typer.Option(
        "demo-prd",
        help="Confluence page ID served by the mock.",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Deterministic heuristic stories (no OpenAI billing).",
    ),
    ci: bool = typer.Option(
        False,
        "--ci",
        help="UNSAFE DEMO BYPASS — auto-approve (never for stakeholder recordings).",
    ),
    approve_with: Path | None = typer.Option(
        None,
        "--approve-with",
        readable=True,
        help='Stories JSON: list or {"stories": [...]}; validated against StoryPolicy.',
    ),
    fixture_mode: bool = typer.Option(
        False,
        "--fixture-mode",
        help="PRD+policy from fixtures; retrieval uses offline-pack; POST mock Jira only.",
    ),
    mock_base: str = typer.Option(
        "http://127.0.0.1:8877",
        "--mock-base",
        envvar="MOCK_BASE_URL",
        help="Unified FastAPI mocks origin.",
    ),
    max_repairs: int = typer.Option(3, "--max-repairs", min=1, max=10),
    model: str = typer.Option("gpt-4o-mini", "--model", envvar="OPENAI_MODEL"),
    stream_events: bool = typer.Option(
        False,
        "--stream-events",
        help="JSONL astream_events (requires --ci).",
    ),
) -> None:
    """Terminal LangGraph run (prefer the mocked Confluence `/` hub for stakeholder demos)."""
    ci_flag = ci or os.getenv("DEMO_AUTO_APPROVE", "").strip().lower() in {"1", "true", "yes"}

    if stream_events and not ci_flag:
        raise typer.BadParameter("`--stream-events` requires `--ci`.")

    mock_origin = mock_base.rstrip("/")
    approve_path_txt = approve_with.expanduser().as_posix() if approve_with else None

    deps = build_graph_deps_for_run(
        page_id=page_id,
        fixture_mode=fixture_mode,
        mock_origin=mock_origin,
        max_repairs=max_repairs,
        model=model,
    )

    runner_graph = compile_demo_graph(deps)
    initial_board = initialise_state(
        page_id=page_id,
        offline_llm=offline or not os.getenv("OPENAI_API_KEY", "").strip(),
        approve_with_path=approve_path_txt,
        auto_approve=ci_flag,
    )
    runnable_config = {"configurable": {"thread_id": initial_board["run_id"]}}

    async def stream_jsonl() -> None:
        async for event in runner_graph.astream_events(initial_board, runnable_config, version="v2"):
            typer.echo(
                json.dumps({"event": event.get("event"), "name": event.get("name")}, default=str)
            )

    try:
        if stream_events:
            asyncio.run(stream_jsonl())
            checkpoint = runner_graph.get_state(runnable_config)
            outcome_bundle: dict = {**initial_board, **checkpoint.values}
        else:
            outcome_bundle = runner_graph.invoke(initial_board, runnable_config)

        paused = outcome_bundle.get("__interrupt__") or []

        if paused:
            envelope = paused[0]
            val = getattr(envelope, "value", envelope)
            typer.echo(f"[human-gate interrupt] {val}", err=True)
            typer.echo(f"[proposal markdown] {outcome_bundle.get('proposal_md_path')}", err=True)

            if ci_flag:
                typer.echo("`--ci` should suppress interrupts.", err=True)
                raise typer.Exit(code=3)

            if approve_path_txt:
                typer.echo("Interrupted despite `--approve-with`; check JSON validity.", err=True)
                raise typer.Exit(code=4)

            if not sys.stdin.isatty():
                typer.echo("Non-interactive TTY — pass `--ci` or `--approve-with`.", err=True)
                raise typer.Exit(code=2)

            reviewer_ok = typer.confirm("Approve backlog for ledger + Jira mock?", default=False)
            outcome_bundle = runner_graph.invoke(Command(resume={"approve": reviewer_ok}), runnable_config)

        typer.echo(json.dumps({"transcript_path": outcome_bundle.get("transcript_path")}, indent=2))
        typer.echo(json.dumps({"jira_projection": outcome_bundle.get("jira_projection")}, indent=2))

    finally:
        close_graph_clients(deps)


def main() -> None:
    """Entry for `demo` script and `python -m reqs_agent_demo`."""

    app()
