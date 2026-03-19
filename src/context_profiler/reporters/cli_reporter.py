"""Rich terminal reporter — beautiful CLI output."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from context_profiler.profiler import ProfileResult


def _format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _pct(part: float, whole: float) -> str:
    if whole == 0:
        return "0%"
    return f"{part / whole * 100:.1f}%"


def render_report(result: ProfileResult, console: Console | None = None) -> None:
    if console is None:
        console = Console()

    console.print()
    console.print(
        Panel.fit(
            f"[bold]context-profiler[/bold]  |  mode: {result.mode}  |  source: {result.source}",
            border_style="blue",
        )
    )

    # Warnings
    if result.all_warnings:
        console.print()
        console.print("[bold yellow]⚠ Warnings[/bold yellow]")
        for w in result.all_warnings:
            console.print(f"  [yellow]• {w}[/yellow]")

    # Token Counter
    tc = result.analyzer_results.get("token_counter")
    if tc:
        _render_token_summary(console, tc.summary)

    # Session Timeline
    if result.session_timeline:
        _render_timeline(console, result.session_timeline)

    console.print()


def _render_token_summary(console: Console, summary: dict) -> None:
    console.print()
    console.print("[bold]Token Distribution[/bold]")

    total = summary.get("total_input_tokens", 0)

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Category", style="cyan")
    table.add_column("Tokens", justify="right")
    table.add_column("% of Total", justify="right")

    table.add_row(
        "Total Input",
        f"[bold]{_format_tokens(total)}[/bold]",
        "100%",
    )
    table.add_row(
        "  System Prompt",
        _format_tokens(summary.get("system_prompt_tokens", 0)),
        _pct(summary.get("system_prompt_tokens", 0), total),
    )
    table.add_row(
        "  Tool Definitions",
        _format_tokens(summary.get("tool_definition_tokens", 0)),
        _pct(summary.get("tool_definition_tokens", 0), total),
    )

    by_role = summary.get("by_role", {})
    for role, tokens in sorted(by_role.items(), key=lambda x: x[1], reverse=True):
        if role == "system":
            continue
        table.add_row(f"  Messages ({role})", _format_tokens(tokens), _pct(tokens, total))

    console.print(table)

    by_type = summary.get("by_content_type", {})
    if by_type:
        console.print()
        console.print("[bold]  By Content Type[/bold]")
        type_table = Table(show_header=False, box=None, padding=(0, 2))
        type_table.add_column("Type", style="dim")
        type_table.add_column("Tokens", justify="right")
        type_table.add_column("%", justify="right")
        for ctype, tokens in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
            type_table.add_row(f"    {ctype}", _format_tokens(tokens), _pct(tokens, total))
        console.print(type_table)

    top_tools = summary.get("top_tools_by_tokens", [])
    if top_tools:
        console.print()
        console.print("[bold]  Top Tools by Token Usage[/bold]")
        tool_table = Table(show_header=False, box=None, padding=(0, 2))
        tool_table.add_column("Tool", style="green")
        tool_table.add_column("Tokens", justify="right")
        tool_table.add_column("%", justify="right")
        for tool_name, tokens in top_tools[:5]:
            tool_table.add_row(f"    {tool_name}", _format_tokens(tokens), _pct(tokens, total))
        console.print(tool_table)


def _render_timeline(console: Console, timeline: list[dict]) -> None:
    console.print()
    console.print("[bold]Context Growth Timeline[/bold]")

    if not timeline:
        return

    max_tokens = max(t["total_tokens"] for t in timeline) or 1
    bar_width = 40

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Tokens", justify="right", width=8)
    table.add_column("Bar", width=bar_width + 2)

    for t in timeline:
        idx = t["request_index"]
        total = t["total_tokens"]

        bar_len = int((total / max_tokens) * bar_width)
        bar_len = max(bar_len, 1) if total > 0 else 0

        bar = Text()
        bar.append("█" * bar_len, style="blue")

        table.add_row(str(idx), _format_tokens(total), bar)

    console.print(table)
