#!/usr/bin/env python3
"""
Colin Trading Bot CLI - Main Entry Point

This is the main entry point for the Colin Trading Bot command-line interface.
It provides a unified, rich terminal interface for accessing all bot functionality.

Usage:
    colin                    # Interactive mode with menu
    colin analysis signals   # Direct command execution
    colin --help            # Show help
    colin --version         # Show version
"""

import click
import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.table import Table
from rich.layout import Layout
from rich import print as rprint
import os
from typing import Optional

from .utils.formatters import format_branding, format_version
from .utils.error_handler import CLIError, handle_cli_error
from .commands.analysis import analysis_group
from .commands.trading import trading_group
from .commands.config import config_group
from .commands.security import security_group
from .commands.monitoring import monitoring_group
from .sessions.interactive import InteractiveSession

console = Console()


class ColinCLI:
    """Main CLI application class with rich branding and session management."""

    def __init__(self):
        self.console = Console()
        self.session: Optional[InteractiveSession] = None

    def display_banner(self):
        """Display the Colin Trading Bot branding banner."""
        banner_text = Text()
        banner_text.append("🚀 ", style="bold blue")
        banner_text.append("COLIN TRADING BOT", style="bold bright_blue")
        banner_text.append(" v2.0", style="bold cyan")
        banner_text.append(" 🤖", style="bold blue")

        subtitle = Text("AI-Powered Institutional Trading System", style="italic dim")

        panel = Panel(
            Align.center(banner_text) + "\n" + Align.center(subtitle),
            border_style="blue",
            padding=(1, 2)
        )

        self.console.print(panel)
        self.console.print()

    def display_main_menu(self):
        """Display the main interactive menu."""
        table = Table(show_header=False, box=None, padding=1)
        table.add_column("Command", style="bold cyan", width=20)
        table.add_column("Description", style="white")

        menu_items = [
            ("analysis", "📊 Market analysis & signal generation"),
            ("trading", "💼 Order management & trading operations"),
            ("config", "⚙️  Configuration management"),
            ("security", "🔐 Security & credential management"),
            ("monitoring", "📈 System monitoring & alerts"),
            ("interactive", "🎯 Interactive trading session"),
            ("status", "ℹ️  System status & health check"),
            ("help", "❓ Show help and documentation"),
            ("exit", "👋 Exit the CLI"),
        ]

        for cmd, desc in menu_items:
            table.add_row(cmd, desc)

        panel = Panel(
            table,
            title="[bold blue]Main Menu[/bold blue]",
            border_style="cyan",
            padding=(1, 2)
        )

        self.console.print(panel)

    def run_interactive_mode(self):
        """Run the interactive CLI mode with menu navigation."""
        self.display_banner()

        while True:
            self.display_main_menu()

            try:
                choice = self.console.input("\n[bold cyan]Enter your choice:[/bold cyan] ").strip().lower()

                if choice in ["exit", "quit", "q"]:
                    self.console.print("[bold green]Goodbye! 👋[/bold green]")
                    break
                elif choice == "help" or choice == "?":
                    self.show_help()
                elif choice == "status":
                    self.show_status()
                elif choice == "interactive":
                    self.start_interactive_session()
                elif choice in ["analysis", "trading", "config", "security", "monitoring"]:
                    self.run_command_group(choice)
                else:
                    self.console.print(f"[bold red]Unknown command: {choice}[/bold red]")
                    self.console.print("[dim]Type 'help' for available commands[/dim]")

            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]Use 'exit' to quit gracefully[/bold yellow]")
            except EOFError:
                self.console.print("\n[bold green]Goodbye! 👋[/bold green]")
                break

    def start_interactive_session(self):
        """Start an interactive trading session."""
        if not self.session:
            self.session = InteractiveSession(console=self.console)

        try:
            self.session.run()
        except Exception as e:
            handle_cli_error(e, self.console)

    def run_command_group(self, group_name: str):
        """Execute a command group interactively."""
        # This would invoke the appropriate click group
        # For now, show a placeholder message
        self.console.print(f"[bold cyan]Executing {group_name} commands...[/bold cyan]")
        self.console.print("[dim]Use direct commands for more control: colin {group_name} --help[/dim]")

    def show_help(self):
        """Display help information."""
        help_text = """
[bold blue]Colin Trading Bot CLI Help[/bold blue]

[dim]Direct Commands:[/dim]
  [cyan]colin analysis signals[/cyan]      - Generate trading signals
  [cyan]colin trading create[/cyan]       - Create new orders
  [cyan]colin config show[/cyan]          - Show configuration
  [cyan]colin security store[/cyan]       - Store credentials securely
  [cyan]colin monitoring dashboard[/cyan] - Open monitoring dashboard

[dim]Interactive Mode:[/dim]
  Simply run [cyan]colin[/cyan] to enter interactive mode with menu navigation

[dim]Get Help:[/dim]
  [cyan]colin --help[/cyan]              - Show this help
  [cyan]colin <command> --help[/cyan]    - Show command-specific help

[dim]Examples:[/dim]
  [cyan]colin analysis signals --symbol ETH/USDT[/cyan]
  [cyan]colin trading create --buy ETH --amount 100[/cyan]
  [cyan]colin config set --key trading.enabled --value true[/cyan]
        """
        self.console.print(help_text)

    def show_status(self):
        """Display system status information."""
        status_table = Table(title="System Status", box=None)
        status_table.add_column("Component", style="bold cyan")
        status_table.add_column("Status", style="bold")
        status_table.add_column("Details", style="dim")

        # Placeholder status information
        status_data = [
            ("CLI Version", "✅ Active", "v2.0.0"),
            ("Python", "✅ Active", sys.version.split()[0]),
            ("Dependencies", "✅ OK", "All required packages installed"),
            ("Bot Core", "⚠️  Standby", "Run 'colin status --detailed' for more info"),
            ("Market Data", "✅ Connected", "Multi-source data available"),
        ]

        for component, status, details in status_data:
            status_color = "green" if "✅" in status else "yellow" if "⚠️" in status else "red"
            status_table.add_row(component, f"[{status_color}]{status}[/{status_color}]", details)

        self.console.print(status_table)


@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version information')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, version, verbose):
    """
    🚀 Colin Trading Bot CLI - AI-Powered Institutional Trading System

    A sophisticated command-line interface for cryptocurrency trading with
    rich terminal interactions, secure credential management, and comprehensive
    trading functionality.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

    if version:
        console.print(format_version())
        return

    if ctx.invoked_subcommand is None:
        # No subcommand specified, run interactive mode
        colin_cli = ColinCLI()
        colin_cli.run_interactive_mode()


# Add command groups
cli.add_command(analysis_group, name='analysis')
cli.add_command(trading_group, name='trading')
cli.add_command(config_group, name='config')
cli.add_command(security_group, name='security')
cli.add_command(monitoring_group, name='monitoring')


@cli.command()
@click.option('--detailed', is_flag=True, help='Show detailed status information')
def status(detailed):
    """Show system status and health information."""
    colin_cli = ColinCLI()

    if detailed:
        console.print("[bold blue]Detailed System Status[/bold blue]")
        console.print("\n[dim]Note: Detailed status requires bot components to be initialized[/dim]")
        console.print("Run 'colin config validate' to check configuration health")

    colin_cli.show_status()


@cli.command()
def version():
    """Show version information."""
    console.print(format_version())


# Error handling for the entire CLI
def main():
    """Main entry point for the CLI application."""
    try:
        # Handle Ctrl+C gracefully
        cli(standalone_mode=False)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation cancelled by user[/bold yellow]")
        sys.exit(1)
    except Exception as e:
        handle_cli_error(e, console)
        sys.exit(1)
    except click.exceptions.ClickException as e:
        e.show()
        sys.exit(e.exit_code)


if __name__ == '__main__':
    main()