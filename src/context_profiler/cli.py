"""CLI entry point for context-profiler."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from context_profiler.profiler import (
    ALL_ANALYZERS,
    load_multi_trace_session,
    load_request,
    load_session,
    profile_request,
    profile_session,
    try_load_langfuse,
)
from context_profiler.reporters.cli_reporter import render_report
from context_profiler.reporters.json_reporter import export_json


@click.group()
@click.version_option()
def main():
    """context-profiler — LLM agent context redundancy profiler."""
    pass


@main.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("--format", "fmt", type=click.Choice(["openai", "anthropic", "langfuse"]), default=None,
              help="Input format (auto-detected if not specified)")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Write JSON report to this file")
@click.option("--html", "html_output", type=click.Path(), default=None,
              help="Write interactive HTML report to this file")
@click.option("--only", multiple=True,
              help="Only run specific analyzers (token_counter)")
@click.option("--session", "session_mode", is_flag=True, default=False,
              help="Session mode: treat input as directory of requests or JSONL")
def analyze(paths: tuple, fmt: str | None, output: str | None, html_output: str | None,
            only: tuple, session_mode: bool):
    """Analyze an API request or session for context redundancy."""
    console = Console()

    if only:
        analyzers = [a for a in ALL_ANALYZERS if a.name in only]
        if not analyzers:
            console.print(f"[red]No matching analyzers for: {', '.join(only)}[/red]")
            raise SystemExit(1)
    else:
        analyzers = ALL_ANALYZERS

    # Multiple files → multi-trace session mode
    if len(paths) > 1:
        input_paths = [Path(p) for p in paths]
        try:
            session = load_multi_trace_session(input_paths, format_hint=fmt)
            result = profile_session(session, source=str(paths[0]), analyzers=analyzers)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise SystemExit(1)

        render_report(result, console=console)

        if output:
            output_path = Path(output)
            export_json(result, output_path)
            console.print(f"\n[green]JSON report saved to {output_path}[/green]")

        if html_output:
            from context_profiler.reporters.html_reporter import export_html
            html_path = Path(html_output)
            export_html(result, html_path, session=session)
            console.print(f"\n[green]HTML report saved to {html_path}[/green]")
        return

    # Single file — original logic
    input_path = Path(paths[0])

    is_session = (
        session_mode
        or input_path.is_dir()
        or input_path.suffix == ".jsonl"
        or fmt == "langfuse"
    )

    if not is_session and input_path.suffix == ".json":
        langfuse_session = try_load_langfuse(input_path)
        if langfuse_session is not None:
            is_session = True

    try:
        if is_session:
            session = load_session(input_path, format_hint=fmt)
            result = profile_session(session, source=str(input_path), analyzers=analyzers)
        else:
            request = load_request(input_path, format_hint=fmt)
            result = profile_request(request, source=str(input_path), analyzers=analyzers)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)

    render_report(result, console=console)

    if output:
        output_path = Path(output)
        export_json(result, output_path)
        console.print(f"\n[green]JSON report saved to {output_path}[/green]")

    if html_output:
        from context_profiler.reporters.html_reporter import export_html

        html_path = Path(html_output)
        session_data = None
        if is_session:
            session_data = session
        export_html(result, html_path, session=session_data)
        console.print(f"\n[green]HTML report saved to {html_path}[/green]")


if __name__ == "__main__":
    main()
