"""
Interactive trading session manager for the Colin Trading Bot CLI.

This module provides an interactive session with real-time updates,
market data streaming, and hands-on trading capabilities.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.prompt import Prompt, Confirm
from rich.gauge import Gauge
from rich.progress import Progress, BarColumn, TextColumn

from ..utils.formatters import (
    format_trading_pair, format_price, format_percentage, format_signal_strength,
    format_order_status, format_datetime, create_dashboard_layout
)
from ..utils.error_handler import InteractiveSessionError


class InteractiveSession:
    """Interactive trading session with real-time displays."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.running = False
        self.selected_symbols = ['ETH/USDT', 'BTC/USDT']
        self.refresh_rate = 2  # seconds
        self.session_start = datetime.now()
        self.layout = None

        # Session state
        self.portfolio_data = self._mock_portfolio_data()
        self.market_data = self._mock_market_data()
        self.orders_data = self._mock_orders_data()
        self.signals_data = self._mock_signals_data()
        self.alerts_data = []

    def _mock_portfolio_data(self) -> Dict[str, Any]:
        """Generate mock portfolio data."""
        return {
            'total_value_usd': 124580.50,
            'available_balance_usd': 45230.00,
            'unrealized_pnl': 2340.25,
            'pnl_percent': 1.92,
            'positions': [
                {
                    'symbol': 'ETH/USDT',
                    'side': 'long',
                    'size': 2.5,
                    'entry_price': 3850.00,
                    'current_price': 3980.50,
                    'unrealized_pnl': 326.25,
                    'pnl_percent': 3.39,
                },
                {
                    'symbol': 'BTC/USDT',
                    'side': 'long',
                    'size': 0.15,
                    'entry_price': 66500.00,
                    'current_price': 67200.00,
                    'unrealized_pnl': 105.00,
                    'pnl_percent': 1.05,
                }
            ]
        }

    def _mock_market_data(self) -> Dict[str, Any]:
        """Generate mock market data."""
        return {
            'ETH/USDT': {
                'price': 3980.50,
                'change_24h': 2.3,
                'volume_24h': 1250000000,
                'high_24h': 4050.00,
                'low_24h': 3850.00,
                'market_cap': 478000000000,
            },
            'BTC/USDT': {
                'price': 67200.00,
                'change_24h': 1.8,
                'volume_24h': 28000000000,
                'high_24h': 68500.00,
                'low_24h': 66000.00,
                'market_cap': 1310000000000,
            }
        }

    def _mock_orders_data(self) -> List[Dict[str, Any]]:
        """Generate mock orders data."""
        return [
            {
                'order_id': 'ORD-001',
                'symbol': 'ETH/USDT',
                'side': 'buy',
                'type': 'limit',
                'amount': 1.0,
                'price': 3850.00,
                'status': 'open',
                'created_at': datetime.now() - timedelta(minutes=15),
            },
            {
                'order_id': 'ORD-002',
                'symbol': 'BTC/USDT',
                'side': 'sell',
                'type': 'stop',
                'amount': 0.1,
                'price': 68000.00,
                'status': 'open',
                'created_at': datetime.now() - timedelta(minutes=30),
            }
        ]

    def _mock_signals_data(self) -> List[Dict[str, Any]]:
        """Generate mock signals data."""
        return [
            {
                'symbol': 'ETH/USDT',
                'direction': 'BUY',
                'strength': 0.78,
                'confidence': 0.85,
                'timeframe': '1h',
                'timestamp': datetime.now() - timedelta(minutes=5),
            },
            {
                'symbol': 'BTC/USDT',
                'direction': 'HOLD',
                'strength': 0.45,
                'confidence': 0.62,
                'timeframe': '4h',
                'timestamp': datetime.now() - timedelta(minutes=15),
            }
        ]

    def create_layout(self) -> Layout:
        """Create the interactive session layout."""
        layout = create_dashboard_layout()

        # Header with session info
        session_duration = datetime.now() - self.session_start
        header_text = Text()
        header_text.append("🚀 COLIN TRADING BOT - INTERACTIVE SESSION", style="bold blue")
        header_text.append(f" | Session: {session_duration}", style="dim")

        layout["header"].update(
            Panel(
                Align.center(header_text),
                border_style="blue"
            )
        )

        # Portfolio panel
        portfolio_panel = self._create_portfolio_panel()
        layout["portfolio"].update(portfolio_panel)

        # Signals panel
        signals_panel = self._create_signals_panel()
        layout["signals"].update(signals_panel)

        # Orders panel
        orders_panel = self._create_orders_panel()
        layout["orders"].update(orders_panel)

        # Market data panel
        market_panel = self._create_market_data_panel()
        layout["market_data"].update(market_panel)

        # Footer with controls
        footer_text = Text("[dim]Press 'q' to quit | 'o' for orders | 's' for symbols | 'h' for help[/dim]")
        layout["footer"].update(
            Panel(
                Align.center(footer_text),
                border_style="dim"
            )
        )

        return layout

    def _create_portfolio_panel(self) -> Panel:
        """Create portfolio summary panel."""
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="bold")

        pnl_style = "green" if self.portfolio_data['unrealized_pnl'] >= 0 else "red"

        portfolio_items = [
            ("Total Value", format_price(self.portfolio_data['total_value_usd']).plain),
            ("Available", format_price(self.portfolio_data['available_balance_usd']).plain),
            ("24h P&L", f"[{pnl_style}]{format_price(self.portfolio_data['unrealized_pnl']).plain}[/{pnl_style}]"),
            ("P&L %", f"[{pnl_style}]{format_percentage(self.portfolio_data['pnl_percent'] / 100).plain}[/{pnl_style}]"),
            ("Positions", str(len(self.portfolio_data['positions']))),
        ]

        for metric, value in portfolio_items:
            table.add_row(metric, value)

        return Panel(table, title="[bold blue]Portfolio[/bold blue]", border_style="cyan")

    def _create_signals_panel(self) -> Panel:
        """Create recent signals panel."""
        table = Table(title="Recent Signals", show_header=True, box=None)
        table.add_column("Symbol", style="bold cyan")
        table.add_column("Direction", style="bold")
        table.add_column("Strength", style="bold")
        table.add_column("Confidence", style="bold")
        table.add_column("Time", style="dim")

        for signal in self.signals_data[:3]:  # Show top 3 signals
            direction_style = "green" if signal['direction'] == 'BUY' else "red" if signal['direction'] == 'SELL' else "yellow"
            time_str = (datetime.now() - signal['timestamp']).seconds // 60

            table.add_row(
                format_trading_pair(signal['symbol']).plain,
                f"[{direction_style}]{signal['direction']}[/{direction_style}]",
                format_signal_strength(signal['strength']).plain,
                f"{signal['confidence']:.0%}",
                f"{time_str}m ago"
            )

        return Panel(table, title="[bold blue]Signals[/bold blue]", border_style="cyan")

    def _create_orders_panel(self) -> Panel:
        """Create active orders panel."""
        table = Table(title="Active Orders", show_header=True, box=None)
        table.add_column("ID", style="bold cyan")
        table.add_column("Symbol", style="bold")
        table.add_column("Side", style="bold")
        table.add_column("Type", style="white")
        table.add_column("Status", style="bold")

        for order in self.orders_data[:3]:  # Show top 3 orders
            side_style = "green" if order['side'] == 'buy' else "red"

            table.add_row(
                order['order_id'],
                format_trading_pair(order['symbol']).plain,
                f"[{side_style}]{order['side'].upper()}[/{side_style}]",
                order['type'].upper(),
                format_order_status(order['status'])
            )

        return Panel(table, title="[bold blue]Orders[/bold blue]", border_style="cyan")

    def _create_market_data_panel(self) -> Panel:
        """Create market data panel."""
        table = Table(title="Market Overview", show_header=True, box=None)
        table.add_column("Symbol", style="bold cyan")
        table.add_column("Price", style="bold")
        table.add_column("24h %", style="bold")
        table.add_column("Volume", style="dim")

        for symbol, data in self.market_data.items():
            change_style = "green" if data['change_24h'] >= 0 else "red"

            table.add_row(
                format_trading_pair(symbol).plain,
                format_price(data['price']).plain,
                f"[{change_style}]{format_percentage(data['change_24h'] / 100).plain}[/{change_style}]",
                f"${data['volume_24h']/1e9:.1f}B"
            )

        return Panel(table, title="[bold blue]Market Data[/bold blue]", border_style="cyan")

    def show_main_menu(self) -> str:
        """Show the interactive main menu."""
        console = Console()

        menu_table = Table(show_header=False, box=None, padding=1)
        menu_table.add_column("Option", style="bold cyan", width=25)
        menu_table.add_column("Description", style="white")

        menu_items = [
            ("1. Quick Buy", "Execute quick market buy order"),
            ("2. Quick Sell", "Execute quick market sell order"),
            ("3. Limit Order", "Place limit order with custom price"),
            ("4. Signals", "View and analyze trading signals"),
            ("5. Positions", "Manage open positions"),
            ("6. Market Data", "Detailed market analysis"),
            ("7. Settings", "Configure session settings"),
            ("8. Back to Dashboard", "Return to live dashboard"),
            ("q. Quit", "Exit interactive session"),
        ]

        for option, description in menu_items:
            menu_table.add_row(option, description)

        panel = Panel(
            menu_table,
            title="[bold blue]Interactive Trading Menu[/bold blue]",
            border_style="cyan",
            padding=(1, 2)
        )

        console.print(panel)
        return Prompt.ask("\n[bold cyan]Choose an option[/bold cyan]", choices=['1', '2', '3', '4', '5', '6', '7', '8', 'q'])

    def quick_buy(self):
        """Handle quick buy order."""
        console = Console()

        console.print("[bold blue]Quick Buy Order[/bold blue]\n")

        symbol = Prompt.ask("Symbol", default=self.selected_symbols[0])
        amount = float(Prompt.ask("Amount", default="0.1"))

        # Simulate order execution
        console.print(f"\n[yellow]Executing market buy: {amount} {symbol}[/yellow]")

        with Progress(
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Executing order...", total=100)
            for i in range(100):
                time.sleep(0.01)
                progress.update(task, advance=1)

        price = self.market_data.get(symbol, {}).get('price', 3980.50)
        total_cost = price * amount

        console.print(f"[bold green]✅ Order executed![/bold green]")
        console.print(f"Price: {format_price(price).plain}")
        console.print(f"Amount: {amount}")
        console.print(f"Total: {format_price(total_cost).plain}")

        # Update portfolio data (mock)
        self.portfolio_data['available_balance_usd'] -= total_cost

    def quick_sell(self):
        """Handle quick sell order."""
        console = Console()

        console.print("[bold blue]Quick Sell Order[/bold blue]\n")

        # Show available positions
        console.print("[cyan]Available positions:[/cyan]")
        for i, pos in enumerate(self.portfolio_data['positions'], 1):
            console.print(f"{i}. {pos['symbol']} - {pos['size']} @ {format_price(pos['entry_price']).plain}")

        choice = Prompt.ask("Select position (number)", choices=[str(i) for i in range(1, len(self.portfolio_data['positions']) + 1)])
        position = self.portfolio_data['positions'][int(choice) - 1]

        amount = float(Prompt.ask("Amount to sell", default=str(position['size'])))

        # Simulate order execution
        console.print(f"\n[yellow]Executing market sell: {amount} {position['symbol']}[/yellow]")

        with Progress(
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Executing order...", total=100)
            for i in range(100):
                time.sleep(0.01)
                progress.update(task, advance=1)

        current_price = self.market_data.get(position['symbol'], {}).get('price', position['current_price'])
        total_value = current_price * amount

        console.print(f"[bold green]✅ Order executed![/bold green]")
        console.print(f"Price: {format_price(current_price).plain}")
        console.print(f"Amount: {amount}")
        console.print(f"Total: {format_price(total_value).plain}")

        # Update portfolio data (mock)
        self.portfolio_data['available_balance_usd'] += total_value

    def limit_order(self):
        """Handle limit order placement."""
        console = Console()

        console.print("[bold blue]Limit Order[/bold blue]\n")

        symbol = Prompt.ask("Symbol", default=self.selected_symbols[0])
        side = Prompt.ask("Side", choices=['buy', 'sell'])
        amount = float(Prompt.ask("Amount", default="0.1"))
        price = float(Prompt.ask("Limit price", default="3950.00"))

        # Simulate order placement
        console.print(f"\n[yellow]Placing limit order: {side.upper()} {amount} {symbol} @ {format_price(price).plain}[/yellow]")

        with Progress(
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Placing order...", total=100)
            for i in range(100):
                time.sleep(0.01)
                progress.update(task, advance=1)

        # Add to orders data
        new_order = {
            'order_id': f'ORD-{datetime.now().strftime("%H%M%S")}',
            'symbol': symbol,
            'side': side,
            'type': 'limit',
            'amount': amount,
            'price': price,
            'status': 'open',
            'created_at': datetime.now(),
        }
        self.orders_data.insert(0, new_order)

        console.print(f"[bold green]✅ Limit order placed![/bold green]")
        console.print(f"Order ID: {new_order['order_id']}")
        console.print(f"{side.upper()} {amount} {symbol} @ {format_price(price).plain}")

    def view_signals(self):
        """View detailed signals analysis."""
        console = Console()

        console.print("[bold blue]Trading Signals Analysis[/bold blue]\n")

        # Detailed signals table
        table = Table(title="Active Signals", show_header=True)
        table.add_column("Symbol", style="bold cyan")
        table.add_column("Direction", style="bold")
        table.add_column("Strength", style="bold")
        table.add_column("Confidence", style="bold")
        table.add_column("Timeframe", style="white")
        table.add_column("Time", style="dim")

        for signal in self.signals_data:
            direction_style = "green" if signal['direction'] == 'BUY' else "red" if signal['direction'] == 'SELL' else "yellow"
            time_str = (datetime.now() - signal['timestamp']).seconds // 60

            table.add_row(
                format_trading_pair(signal['symbol']).plain,
                f"[{direction_style}]{signal['direction']}[/{direction_style}]",
                format_signal_strength(signal['strength']).plain,
                f"{signal['confidence']:.0%}",
                signal['timeframe'],
                f"{time_str}m ago"
            )

        console.print(table)

        # Action recommendations
        strong_signals = [s for s in self.signals_data if s['confidence'] > 0.75 and s['strength'] > 0.7]
        if strong_signals:
            console.print(f"\n[bold green]🎯 High-Confidence Opportunities:[/bold green]")
            for signal in strong_signals:
                action = "Consider BUY position" if signal['direction'] == 'BUY' else f"Consider {signal['direction']} position"
                console.print(f"• {signal['symbol']}: {action} (Conf: {signal['confidence']:.0%}, Str: {signal['strength']:.0%})")

    def configure_settings(self):
        """Configure session settings."""
        console = Console()

        console.print("[bold blue]Session Settings[/bold blue]\n")

        # Show current settings
        settings_table = Table(show_header=False, box=None)
        settings_table.add_column("Setting", style="bold cyan")
        settings_table.add_column("Value", style="white")

        current_settings = [
            ("Refresh Rate", f"{self.refresh_rate} seconds"),
            ("Selected Symbols", ", ".join(self.selected_symbols)),
            ("Auto-refresh", "Enabled"),
            ("Alert Level", "Medium"),
        ]

        for setting, value in current_settings:
            settings_table.add_row(setting, value)

        console.print(settings_table)

        # Allow changes
        console.print(f"\n[cyan]Available actions:[/cyan]")
        console.print("1. Change refresh rate")
        console.print("2. Add/remove symbols")
        console.print("3. Toggle auto-refresh")
        console.print("4. Back to menu")

        choice = Prompt.ask("\nChoose action", choices=['1', '2', '3', '4'])

        if choice == '1':
            new_rate = int(Prompt.ask("New refresh rate (seconds)", default=str(self.refresh_rate)))
            self.refresh_rate = max(1, min(30, new_rate))  # Limit between 1-30 seconds
            console.print(f"[green]Refresh rate set to {self.refresh_rate} seconds[/green]")

        elif choice == '2':
            available_symbols = ['ETH/USDT', 'BTC/USDT', 'ADA/USDT', 'SOL/USDT', 'DOT/USDT']
            console.print(f"Available symbols: {', '.join(available_symbols)}")
            symbol = Prompt.ask("Symbol to add/remove")
            if symbol in available_symbols:
                if symbol in self.selected_symbols:
                    self.selected_symbols.remove(symbol)
                    console.print(f"[yellow]Removed {symbol} from watchlist[/yellow]")
                else:
                    self.selected_symbols.append(symbol)
                    console.print(f"[green]Added {symbol} to watchlist[/green]")

        elif choice == '3':
            console.print("[yellow]Auto-refresh toggle not implemented yet[/yellow]")

    async def run_dashboard(self):
        """Run the live dashboard."""
        self.running = True

        try:
            with Live(self.create_layout(), refresh_per_second=1 / self.refresh_rate) as live:
                while self.running:
                    # Simulate data updates
                    self._update_market_data()
                    self._update_portfolio_data()

                    live.update(self.create_layout())
                    await asyncio.sleep(self.refresh_rate)

        except KeyboardInterrupt:
            self.running = False
            console = Console()
            console.print("\n[yellow]Dashboard stopped[/yellow]")

    def _update_market_data(self):
        """Update market data with simulated changes."""
        import random

        for symbol, data in self.market_data.items():
            # Simulate price movement
            change = random.uniform(-0.5, 0.5)  # ±0.5% change
            data['price'] *= (1 + change / 100)
            data['change_24h'] += random.uniform(-0.1, 0.1)

    def _update_portfolio_data(self):
        """Update portfolio data based on market changes."""
        import random

        # Update unrealized P&L based on market changes
        pnl_change = random.uniform(-50, 50)
        self.portfolio_data['unrealized_pnl'] += pnl_change
        self.portfolio_data['pnl_percent'] = (self.portfolio_data['unrealized_pnl'] /
                                               self.portfolio_data['total_value_usd']) * 100

    def run(self):
        """Run the interactive session."""
        console = Console()

        console.print("[bold blue]🚀 Starting Interactive Trading Session[/bold blue]")
        console.print("[dim]Press Ctrl+C at any time to exit[/dim]\n")

        while True:
            choice = self.show_main_menu()

            if choice == 'q':
                console.print("[bold green]Goodbye! 👋[/bold green]")
                break
            elif choice == '1':
                self.quick_buy()
                Prompt.ask("Press Enter to continue")
            elif choice == '2':
                self.quick_sell()
                Prompt.ask("Press Enter to continue")
            elif choice == '3':
                self.limit_order()
                Prompt.ask("Press Enter to continue")
            elif choice == '4':
                self.view_signals()
                Prompt.ask("Press Enter to continue")
            elif choice == '5':
                console.print("[yellow]Position management not implemented yet[/yellow]")
                Prompt.ask("Press Enter to continue")
            elif choice == '6':
                console.print("[yellow]Market data analysis not implemented yet[/yellow]")
                Prompt.ask("Press Enter to continue")
            elif choice == '7':
                self.configure_settings()
                Prompt.ask("Press Enter to continue")
            elif choice == '8':
                # Run live dashboard
                try:
                    asyncio.run(self.run_dashboard())
                except KeyboardInterrupt:
                    console.print("\n[yellow]Returning to menu[/yellow]")
                    continue