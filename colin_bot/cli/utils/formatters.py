"""
Rich formatting utilities for the Colin Trading Bot CLI.

This module provides various formatting functions for creating beautiful
terminal output with consistent styling and branding.
"""

from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.align import Align
from rich.rule import Rule
from rich.columns import Columns
from rich.layout import Layout
import sys
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import asyncio

from .. import __version__


def format_branding(console: Optional[Console] = None) -> Panel:
    """Create the Colin Trading Bot branding panel."""
    if console is None:
        console = Console()

    banner_text = Text()
    banner_text.append("🚀 ", style="bold blue")
    banner_text.append("COLIN TRADING BOT", style="bold bright_blue")
    banner_text.append(f" v{__version__}", style="bold cyan")
    banner_text.append(" 🤖", style="bold blue")

    subtitle = Text("AI-Powered Institutional Trading System", style="italic dim")

    panel = Panel(
        Align.center(banner_text) + "\n" + Align.center(subtitle),
        border_style="blue",
        padding=(1, 2)
    )

    return panel


def format_version() -> Text:
    """Format version information."""
    version_text = Text()
    version_text.append("Colin Trading Bot ", style="bold blue")
    version_text.append(f"v{__version__}", style="bold cyan")
    version_text.append(f" (Python {sys.version.split()[0]})", style="dim")
    return version_text


def format_success(message: str) -> Text:
    """Format a success message."""
    return Text(f"✅ {message}", style="bold green")


def format_error(message: str) -> Text:
    """Format an error message."""
    return Text(f"❌ {message}", style="bold red")


def format_warning(message: str) -> Text:
    """Format a warning message."""
    return Text(f"⚠️  {message}", style="bold yellow")


def format_info(message: str) -> Text:
    """Format an info message."""
    return Text(f"ℹ️  {message}", style="bold blue")


def format_loading(description: str = "Loading...") -> Progress:
    """Create a loading progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    )


def format_table(data: List[Dict[str, Any]], title: str = "", show_headers: bool = True) -> Table:
    """Format data into a rich table."""
    if not data:
        return Table(title=title, show_header=show_headers)

    table = Table(title=title, show_header=show_headers, box=None)

    # Add columns based on first row keys
    for key in data[0].keys():
        table.add_column(str(key).replace("_", " ").title(), style="cyan")

    # Add rows
    for row in data:
        table.add_row(*[str(value) for value in row.values()])

    return table


def format_signal_strength(strength: float) -> Text:
    """Format signal strength with appropriate styling."""
    if strength >= 0.8:
        style = "bold green"
        emoji = "🟢"
    elif strength >= 0.6:
        style = "bold yellow"
        emoji = "🟡"
    elif strength >= 0.4:
        style = "bold orange1"
        emoji = "🟠"
    else:
        style = "bold red"
        emoji = "🔴"

    return Text(f"{emoji} {strength:.1%}", style=style)


def format_price(price: float, currency: str = "USD", precision: int = 2) -> Text:
    """Format price with appropriate styling."""
    formatted_price = f"${price:,.{precision}f}"
    return Text(formatted_price, style="bold green")


def format_percentage(value: float, precision: int = 2) -> Text:
    """Format percentage with appropriate styling."""
    formatted_value = f"{value:+.{precision}f}%"

    if value > 0:
        style = "bold green"
        emoji = "📈"
    elif value < 0:
        style = "bold red"
        emoji = "📉"
    else:
        style = "bold white"
        emoji = "➡️"

    return Text(f"{emoji} {formatted_value}", style=style)


def format_order_status(status: str) -> Text:
    """Format order status with appropriate styling."""
    status_colors = {
        "filled": "bold green",
        "open": "bold blue",
        "pending": "bold yellow",
        "cancelled": "bold red",
        "rejected": "bold red",
        "partially_filled": "bold orange1",
    }

    status_emojis = {
        "filled": "✅",
        "open": "🔵",
        "pending": "⏳",
        "cancelled": "❌",
        "rejected": "🚫",
        "partially_filled": "🔄",
    }

    style = status_colors.get(status.lower(), "bold white")
    emoji = status_emojis.get(status.lower(), "•")

    return Text(f"{emoji} {status.title()}", style=style)


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> Text:
    """Format datetime with consistent styling."""
    formatted = dt.strftime(format_str)
    return Text(formatted, style="dim")


def format_trading_pair(pair: str) -> Text:
    """Format trading pair with base/quote styling."""
    if "/" in pair:
        base, quote = pair.split("/", 1)
        return Text(f"{base}/{quote}", style="bold cyan")
    return Text(pair, style="bold cyan")


def format_portfolio_summary(data: Dict[str, Any]) -> Panel:
    """Format portfolio summary into a nice panel."""
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold")

    portfolio_items = [
        ("Total Value", format_price(data.get('total_value_usd', 0))),
        ("Available Balance", format_price(data.get('available_balance_usd', 0))),
        ("24h P&L", format_percentage(data.get('pnl_24h_percent', 0))),
        ("Open Positions", str(data.get('open_positions', 0))),
        ("Total Trades", str(data.get('total_trades', 0))),
    ]

    for metric, value in portfolio_items:
        table.add_row(metric, str(value))

    return Panel(table, title="[bold blue]Portfolio Summary[/bold blue]", border_style="cyan")


def format_signal_details(signal: Dict[str, Any]) -> Panel:
    """Format trading signal details into a panel."""
    table = Table(show_header=False, box=None)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="bold")

    signal_items = [
        ("Symbol", format_trading_pair(signal.get('symbol', 'N/A'))),
        ("Direction", signal.get('direction', 'N/A').upper()),
        ("Strength", format_signal_strength(signal.get('strength', 0))),
        ("Confidence", f"{signal.get('confidence', 0):.1%}"),
        ("Timeframe", signal.get('timeframe', 'N/A')),
        ("Timestamp", format_datetime(signal.get('timestamp', datetime.now()))),
    ]

    for field, value in signal_items:
        table.add_row(field, str(value))

    return Panel(table, title="[bold blue]Signal Details[/bold blue]", border_style="cyan")


def create_dashboard_layout() -> Layout:
    """Create a dashboard layout for multiple panels."""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )

    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )

    layout["left"].split_column(
        Layout(name="portfolio", ratio=1),
        Layout(name="signals", ratio=1),
    )

    layout["right"].split_column(
        Layout(name="orders", ratio=1),
        Layout(name="market_data", ratio=1),
    )

    return layout


def format_alert(alert: Dict[str, Any]) -> Panel:
    """Format alert message into a panel."""
    severity = alert.get('severity', 'info').lower()
    severity_colors = {
        'critical': 'bold red',
        'high': 'bold orange1',
        'medium': 'bold yellow',
        'low': 'bold blue',
        'info': 'bold cyan',
    }

    severity_emojis = {
        'critical': '🚨',
        'high': '⚠️',
        'medium': '⚡',
        'low': 'ℹ️',
        'info': '💡',
    }

    style = severity_colors.get(severity, 'bold white')
    emoji = severity_emojis.get(severity, '•')

    title = Text(f"{emoji} {alert.get('title', 'Alert')}", style=style)
    content = Text(alert.get('message', ''), style='white')

    panel = Panel(
        content,
        title=title,
        border_style=style.replace('bold', ''),
        padding=(1, 2)
    )

    return panel


def format_key_value_pairs(data: Dict[str, Any], title: str = "") -> Table:
    """Format key-value pairs into a table."""
    table = Table(title=title, show_header=False, box=None)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="white")

    for key, value in data.items():
        formatted_key = str(key).replace("_", " ").title()
        table.add_row(formatted_key, str(value))

    return table


def format_list(items: List[str], title: str = "", bullet: str = "•") -> Panel:
    """Format a list of items into a panel."""
    content = "\n".join(f"{bullet} {item}" for item in items)
    return Panel(content, title=title, border_style="cyan")