"""CLI entry point for context-profiler."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
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
    try_load_agent_trace,
    try_load_langfuse,
)
from context_profiler.reporters.cli_reporter import render_report
from context_profiler.reporters.json_reporter import export_json


@dataclass
class _ResolvedInput:
    path: Path
    source: str
    cleanup_path: Path | None = None

    def cleanup(self) -> None:
        if self.cleanup_path is None:
            return
        try:
            os.unlink(self.cleanup_path)
        except FileNotFoundError:
            pass


def _resolve_input_arg(arg: str) -> _ResolvedInput:
    """Resolve a CLI input argument, materializing stdin for existing loaders."""
    if arg != "-":
        return _ResolvedInput(path=Path(arg), source=arg)

    text = sys.stdin.read()
    stripped = text.lstrip()
    suffix = ".jsonl" if "\n" in text.rstrip() and not stripped.startswith(("{", "[")) else ".json"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=suffix, delete=False) as tmp:
        tmp.write(text)
        temp_path = Path(tmp.name)
    return _ResolvedInput(path=temp_path, source="-", cleanup_path=temp_path)


@click.group()
@click.version_option()
def main():
    """context-profiler — LLM agent context redundancy profiler."""
    pass


@main.group()
def formats():
    """Discover supported input formats."""
    pass


@formats.command("list")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
def formats_list(json_output: bool):
    """List supported and planned input formats."""
    from context_profiler.formats import list_formats

    data = {"formats": list_formats()}
    if json_output:
        click.echo(json.dumps(data, indent=2, ensure_ascii=False))
        return

    console = Console()
    for fmt in data["formats"]:
        console.print(f"[bold]{fmt['name']}[/bold] ({fmt['status']}) - {fmt['description']}")


@formats.command("describe")
@click.argument("name")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
def formats_describe(name: str, json_output: bool):
    """Describe a supported input format."""
    from context_profiler.formats import describe_format

    try:
        data = describe_format(name)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if json_output:
        click.echo(json.dumps(data, indent=2, ensure_ascii=False))
        return

    console = Console()
    console.print(f"[bold]{data['name']}[/bold] ({data['status']})")
    console.print(data["description"])
    console.print("Required signals:")
    for signal in data["required_signals"]:
        console.print(f"  - {signal}")


@main.command("schema")
@click.argument("name", type=click.Choice(["trace", "diagnosis"]))
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
def schema_command(name: str, json_output: bool):
    """Print JSON Schema for context-profiler contracts."""
    from context_profiler.schemas import get_schema

    data = get_schema(name)
    click.echo(json.dumps(data, indent=2, ensure_ascii=False))


@main.command("validate")
@click.argument("path", type=click.Path(exists=False), required=True)
@click.option("--format", "fmt", default=None, help="Input format hint")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
def validate_command(path: str, fmt: str | None, json_output: bool):
    """Validate an input trace or request."""
    from context_profiler.validation import validate_input

    data = validate_input(path, format_hint=fmt)
    if json_output:
        click.echo(json.dumps(data, indent=2, ensure_ascii=False))
        if not data["valid"]:
            raise SystemExit(1)
        return

    console = Console()
    color = "green" if data["valid"] else "red"
    console.print(f"[{color}]valid={data['valid']} format={data['detected_format']}[/{color}]")
    for error in data["errors"]:
        if isinstance(error, dict):
            console.print(f"[red]{error['code']}: {error['message']}[/red]")
        else:
            console.print(f"[red]{error}[/red]")
    if not data["valid"]:
        raise SystemExit(1)


@main.command("normalize")
@click.argument("path", type=click.Path(exists=False), required=True)
@click.option("--from", "from_format", default=None, help="Input format hint")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
def normalize_command(path: str, from_format: str | None, json_output: bool):
    """Normalize input into ContextTrace JSON."""
    from context_profiler.validation import normalize_input

    data = normalize_input(path, format_hint=from_format)
    click.echo(json.dumps(data, indent=2, ensure_ascii=False))


@main.command("diagnose")
@click.argument("path", type=click.Path(exists=False), required=True)
@click.option("--format", "fmt",
              type=click.Choice([
                  "auto", "openai", "anthropic", "langfuse",
                  "cursor-jsonl", "claude-code-jsonl", "agent-trace",
              ]),
              default=None,
              help="Input format hint")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
def diagnose_command(path: str, fmt: str | None, json_output: bool):
    """Diagnose context pathologies in an input trace or request."""
    from context_profiler.diagnostics import diagnose_result

    fmt = None if fmt == "auto" else fmt
    resolved = _resolve_input_arg(path)
    try:
        input_path = resolved.path
        session_for_diagnosis = try_load_langfuse(input_path)
        if session_for_diagnosis is None:
            session_for_diagnosis = try_load_agent_trace(input_path)
        if (
            session_for_diagnosis is not None
            or input_path.is_dir()
            or input_path.suffix == ".jsonl"
            or fmt in ("langfuse", "agent-trace")
        ):
            session_for_diagnosis = session_for_diagnosis or load_session(input_path, format_hint=fmt)
            result = profile_session(session_for_diagnosis, source=resolved.source)
        else:
            request = load_request(input_path, format_hint=fmt)
            result = profile_request(request, source=resolved.source)
            session_for_diagnosis = None
    finally:
        resolved.cleanup()

    data = diagnose_result(result, session=session_for_diagnosis)
    if json_output:
        click.echo(json.dumps(data, indent=2, ensure_ascii=False))
        return

    console = Console()
    console.print(f"[bold]Issues:[/bold] {data['summary']['issue_count']}")
    for issue in data["issues"]:
        console.print(f"[bold]{issue['severity']}[/bold] {issue['code']}: {issue['message']}")


@main.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=False), required=True)
@click.option("--format", "fmt",
              type=click.Choice([
                  "auto", "openai", "anthropic", "langfuse",
                  "cursor-jsonl", "claude-code-jsonl", "agent-trace",
              ]),
              default=None,
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
    fmt = None if fmt == "auto" else fmt
    if "-" in paths and len(paths) > 1:
        raise click.ClickException("stdin input '-' cannot be combined with other paths")
    resolved_inputs = [_resolve_input_arg(p) for p in paths]

    try:
        if only:
            analyzers = [a for a in ALL_ANALYZERS if a.name in only]
            if not analyzers:
                console.print(f"[red]No matching analyzers for: {', '.join(only)}[/red]")
                raise SystemExit(1)
        else:
            analyzers = ALL_ANALYZERS

        # Multiple files → multi-trace session mode
        if len(resolved_inputs) > 1:
            input_paths = [resolved.path for resolved in resolved_inputs]
            try:
                session = load_multi_trace_session(input_paths, format_hint=fmt)
                result = profile_session(session, source=resolved_inputs[0].source, analyzers=analyzers)
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
        resolved = resolved_inputs[0]
        input_path = resolved.path

        is_session = (
            session_mode
            or input_path.is_dir()
            or input_path.suffix == ".jsonl"
            or fmt in ("langfuse", "agent-trace")
        )

        if not is_session and input_path.suffix == ".json":
            langfuse_session = try_load_langfuse(input_path)
            if langfuse_session is not None:
                is_session = True
            elif try_load_agent_trace(input_path) is not None:
                is_session = True

        try:
            if is_session:
                session = load_session(input_path, format_hint=fmt)
                result = profile_session(session, source=resolved.source, analyzers=analyzers)
            else:
                request = load_request(input_path, format_hint=fmt)
                result = profile_request(request, source=resolved.source, analyzers=analyzers)
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
    finally:
        for resolved in resolved_inputs:
            resolved.cleanup()


if __name__ == "__main__":
    main()
