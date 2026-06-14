"""
Channel Intelligence Agent — CLI

Command-line interface for local development and testing.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from financial_intel.config import load_settings_from_yaml, get_settings
from financial_intel.graph.supervisor import run_research
from financial_intel.rag.pipeline import get_rag_pipeline
from financial_intel.agents.filings_agent import simple_filings_research
from financial_intel.agents.stakeholder_agent import run_stakeholder_agent
from financial_intel.state import VendorInfo, PartnerProfile, Opportunity

app = typer.Typer(
    name="channel-intel",
    help="Channel Intelligence Research Agent CLI",
    add_completion=False,
)
console = Console()


@app.command()
def research(
    query: str = typer.Argument(..., help="Research query"),
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID"),
    user_id: Optional[str] = typer.Option(None, "--user", "-u", help="User ID"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file (JSON)"),
    stream: bool = typer.Option(False, "--stream", help="Stream progress"),
):
    """Run a research query."""
    if config:
        load_settings_from_yaml(config)
    else:
        # Try default config location
        default_config = Path("configs/settings.yaml")
        if default_config.exists():
            load_settings_from_yaml(str(default_config))

    if not session_id:
        session_id = f"cli-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    console.print(Panel(f"[bold]Research Query:[/bold] {query}", title=f"Session: {session_id}"))

    if stream:
        console.print("[yellow]Streaming mode not yet implemented in CLI[/yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running research pipeline...", total=None)

        try:
            final_state = asyncio.run(run_research(
                user_query=query,
                session_id=session_id,
                user_id=user_id,
            ))

            progress.update(task, description="Research completed!")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    # Display results
    _display_results(final_state)

    # Save output if requested
    if output:
        _save_output(final_state, output)
        console.print(f"[green]Results saved to {output}[/green]")


@app.command()
def vendor(
    vendors: str = typer.Argument(..., help="Comma-separated vendor names"),
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file (JSON)"),
):
    """Research specific vendors."""
    vendor_list = [v.strip() for v in vendors.split(",")]
    if not session_id:
        session_id = f"vendor-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    console.print(f"Researching vendors: {', '.join(vendor_list)}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running vendor research...", total=None)

        results = asyncio.run(simple_filings_research(vendor_list, session_id))

        progress.update(task, description="Vendor research completed!")

    _display_vendors(results)

    if output:
        _save_vendors(results, output)


@app.command()
def partner(
    segments: str = typer.Argument(..., help="Comma-separated partner segments"),
    vendor_context: str = typer.Option("", "--vendor", "-v", help="Vendor context"),
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file (JSON)"),
):
    """Research partner segments."""
    segment_list = [s.strip() for s in segments.split(",")]
    if not session_id:
        session_id = f"partner-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    console.print(f"Researching partner segments: {', '.join(segment_list)}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running partner research...", total=None)

        results = asyncio.run(run_stakeholder_agent(segment_list, session_id, vendor_context))

        progress.update(task, description="Partner research completed!")

    _display_partners(results)

    if output:
        _save_partners(results, output)


@app.command()
def ingest(
    source: str = typer.Argument(..., help="Source type: vendor, partner, market"),
    name: str = typer.Argument(..., help="Source name"),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="File path (for vendor)"),
    data: Optional[str] = typer.Option(None, "--data", "-d", help="JSON data (for partner/market)"),
):
    """Ingest data into RAG pipeline."""
    rag = get_rag_pipeline()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Ingesting {source}...", total=None)

        if source == "vendor":
            if not path:
                console.print("[red]Vendor ingestion requires --path[/red]")
                raise typer.Exit(1)
            paths = [p.strip() for p in path.split(",")]
            result = asyncio.run(rag.ingest_vendor_docs(name, paths))

        elif source == "partner":
            if not data:
                console.print("[red]Partner ingestion requires --data[/red]")
                raise typer.Exit(1)
            result = asyncio.run(rag.ingest_partner_data(name, json.loads(data)))

        elif source == "market":
            if not data:
                console.print("[red]Market ingestion requires --data[/red]")
                raise typer.Exit(1)
            payload = json.loads(data)
            result = asyncio.run(rag.ingest_market_report(
                name, payload.get("content", ""), payload.get("metadata", {})
            ))

        else:
            console.print(f"[red]Unknown source type: {source}[/red]")
            raise typer.Exit(1)

        progress.update(task, description="Ingestion completed!")

    console.print(f"[green]Result:[/green] {json.dumps(result, indent=2)}")


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current config"),
    validate: bool = typer.Option(False, "--validate", help="Validate config"),
    config_file: Optional[str] = typer.Option(None, "--file", "-f", help="Config file path"),
):
    """Manage configuration."""
    if config_file:
        settings = load_settings_from_yaml(config_file)
    else:
        settings = get_settings()

    if show:
        console.print(Panel(
            settings.model_dump_json(indent=2),
            title="Current Configuration",
        ))

    if validate:
        try:
            # Validation happens in Pydantic
            console.print("[green]Configuration is valid![/green]")
        except Exception as e:
            console.print(f"[red]Validation failed:[/red] {e}")
            raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers"),
):
    """Start the API server."""
    import uvicorn
    uvicorn.run(
        "financial_intel.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
    )


# ============================================================================
# Display Helpers
# ============================================================================

def _display_results(state: dict):
    """Display research results in console."""
    briefing = state.get("briefing_final", "")
    opportunities = state.get("opportunities", [])
    citations = state.get("citations", [])
    token_usage = state.get("token_usage", {})

    # Briefing
    if briefing:
        console.print(Panel(
            Markdown(briefing),
            title="Executive Briefing",
            border_style="green",
        ))

    # Opportunities
    if opportunities:
        table = Table(title="Scored Opportunities")
        table.add_column("ID", style="cyan")
        table.add_column("Vendor", style="yellow")
        table.add_column("Partner", style="magenta")
        table.add_column("Type", style="blue")
        table.add_column("Score", style="green", justify="right")
        table.add_column("Rationale", style="white")

        for opp in opportunities:
            table.add_row(
                opp.id[:8] if hasattr(opp, 'id') else str(opp.get('id', ''))[:8],
                opp.vendor if hasattr(opp, 'vendor') else opp.get('vendor', ''),
                opp.partner if hasattr(opp, 'partner') else opp.get('partner', ''),
                opp.opportunity_type if hasattr(opp, 'opportunity_type') else opp.get('opportunity_type', ''),
                f"{opp.score:.2f}" if hasattr(opp, 'score') else f"{opp.get('score', 0):.2f}",
                (opp.rationale[:80] + "...") if hasattr(opp, 'rationale') else (opp.get('rationale', '')[:80] + "..."),
            )

        console.print(table)

    # Citations
    if citations:
        console.print("\n[bold]Citations:[/bold]")
        for cite in citations[:10]:
            console.print(f"  [{cite['id']}] {cite['source']}: {cite['title']} ({cite['url']})")

    # Token usage
    if token_usage:
        tu = token_usage
        console.print(f"\n[bold]Token Usage:[/bold] "
                     f"Prompt: {tu.get('prompt_tokens', 0):,} | "
                     f"Completion: {tu.get('completion_tokens', 0):,} | "
                     f"Total: {tu.get('total_tokens', 0):,} | "
                     f"Cost: ${tu.get('estimated_cost_usd', 0):.4f}")


def _display_vendors(results: Dict[str, VendorInfo]):
    """Display vendor research results."""
    for vendor_name, info in results.items():
        console.print(Panel(
            f"[bold]Description:[/bold] {info.description}\n\n"
            f"[bold]Categories:[/bold] {', '.join(info.product_categories) or 'N/A'}\n"
            f"[bold]Pricing:[/bold] {info.pricing_model or 'N/A'}\n"
            f"[bold]Partner Program:[/bold] {info.partner_program or 'N/A'}\n"
            f"[bold]Roadmap:[/bold] {', '.join(info.roadmap_highlights) or 'N/A'}\n"
            f"[bold]Confidence:[/bold] {info.confidence_score:.0%}",
            title=f"Vendor: {vendor_name}",
            border_style="blue",
        ))

        if info.source_urls:
            console.print("[dim]Sources:[/dim]")
            for url in info.source_urls[:5]:
                console.print(f"  • {url}")


def _display_partners(results: Dict[str, PartnerProfile]):
    """Display partner research results."""
    for partner_name, profile in results.items():
        console.print(Panel(
            f"[bold]Specializations:[/bold] {', '.join(profile.specializations) or 'N/A'}\n"
            f"[bold]Vendor Partnerships:[/bold] {', '.join(profile.vendor_partnerships) or 'N/A'}\n"
            f"[bold]Buying Signals:[/bold] {', '.join(profile.buying_signals) or 'N/A'}\n"
            f"[bold]Engagement Score:[/bold] {profile.engagement_score:.0%}\n"
            f"[bold]Opportunity Score:[/bold] {profile.opportunity_score:.0%}",
            title=f"Partner: {partner_name}",
            border_style="magenta",
        ))


def _save_output(state: dict, output_path: str):
    """Save research results to JSON file."""
    output = {
        "briefing": state.get("briefing_final", ""),
        "opportunities": [opp.model_dump() if hasattr(opp, 'model_dump') else opp for opp in state.get("opportunities", [])],
        "citations": state.get("citations", []),
        "token_usage": state.get("token_usage", {}).model_dump() if state.get("token_usage") else {},
        "vendor_research": {k: v.model_dump() for k, v in state.get("vendor_research", {}).items()},
        "market_signals": [s.model_dump() for s in state.get("market_signals", [])],
        "partner_profiles": {k: v.model_dump() for k, v in state.get("partner_profiles", {}).items()},
    }

    Path(output_path).write_text(json.dumps(output, indent=2))


def _save_vendors(results: Dict[str, VendorInfo], output_path: str):
    output = {k: v.model_dump() for k, v in results.items()}
    Path(output_path).write_text(json.dumps(output, indent=2))


def _save_partners(results: Dict[str, PartnerProfile], output_path: str):
    output = {k: v.model_dump() for k, v in results.items()}
    Path(output_path).write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    app()