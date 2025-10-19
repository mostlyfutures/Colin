# PRP: ColinBot CLI Interface - Institutional Trading Command Line Tool

## Executive Summary

This PRP implements a sophisticated command-line interface (CLI) for the Colin Trading Bot, providing institutional-grade access to all trading, analysis, and system management functions through a unified `colin` command. The CLI will feature rich interactive interfaces, comprehensive error handling, secure credential management, and seamless integration with the existing v2 architecture while maintaining compatibility with v1 functionality.

## Current State Analysis

### Existing Command-Line Patterns

**Current Entry Points:**
- `src/main.py` - Original Colin Trading Bot with institutional signal scoring
- `src/v2/main.py` - Colin Trading Bot v2.0 with comprehensive institutional features
- `tools/analysis/colin_bot.py` - Simple wrapper script for the original bot
- Various validation and analysis scripts in `tools/` directory

**Current CLI Usage:**
```bash
# Original bot
python -m src.main --config config.yaml --time-horizon 4h ETHUSDT

# V2 system
python -m src.v2.main --mode development

# Analysis tools
python tools/analysis/demo_real_api.py
python tools/validation/validate_implementation.py
```

**Identified Issues:**
- ❌ Multiple disconnected entry points with different interfaces
- ❌ No unified command structure
- ❌ Poor user experience with basic command-line formatting
- ❌ No interactive capabilities for real-time operations
- ❌ Inconsistent parameter handling across components
- ❌ No secure credential management
- ❌ Limited error handling and user feedback
- ❌ No centralized configuration management via CLI

## Research Findings

### Modern CLI Libraries Analysis

**Recommended Stack: Click + Rich + Keyring**

**Click Benefits:**
- Mature, battle-tested with extensive documentation
- Excellent composability with nested commands and groups
- Rich ecosystem of extensions (click-plugins, click-completion)
- Strong type validation and custom parameter types
- Excellent error handling and help generation

**Rich Integration Benefits:**
- Beautiful terminal formatting and tables
- Interactive progress bars and status indicators
- Syntax highlighting for configuration display
- Rich panels and layouts for professional appearance
- Live updating displays for real-time data

**Keyring Benefits:**
- Secure storage of API credentials
- Cross-platform compatibility
- Integration with system keychain
- No plaintext credential storage

### CLI Design Patterns for Trading Systems

**Hierarchical Command Structure:**
```bash
colin                    # Main command
├── analyze              # Market analysis and signals
├── trade                # Trading operations
├── portfolio            # Portfolio management
├── risk                 # Risk management
├── config               # Configuration management
├── system               # System monitoring
├── validate             # Validation and testing
├── api                  # API server management
└── interactive          # Interactive trading session
```

**Best Practices Identified:**
- Subcommand groups for logical organization
- Rich error handling with specific trading error types
- Dry-run modes for safe testing
- Configuration validation and management
- Real-time data streaming capabilities
- Secure credential storage
- Comprehensive help documentation

## Technical Implementation Blueprint

### 1. Core CLI Architecture

```python
# src/cli/main.py
import click
import asyncio
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel

console = Console()

@click.group()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--dry-run', is_flag=True, help='Execute in simulation mode')
@click.option('--mode', type=click.Choice(['development', 'staging', 'production']),
              default='development', help='Operating mode')
@click.pass_context
def cli(ctx, config, verbose, dry_run, mode):
    """Colin Trading Bot - Institutional-Grade AI Trading System

    Provides command-line access to all trading, analysis, and system management
    functions of the Colin Trading Bot v2.0 with institutional-grade security
    and real-time capabilities.
    """
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose
    ctx.obj['dry_run'] = dry_run
    ctx.obj['mode'] = mode

    # Rich branding
    console.print(Panel.fit(
        "[bold blue]Colin Trading Bot v2.0[/bold blue]\n"
        "Institutional-Grade AI Trading System",
        border_style="blue"
    ))
```

### 2. Market Analysis Commands

```python
# src/cli/commands/analyze.py
import click
import asyncio
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

@cli.group()
def analyze():
    """Market analysis and signal generation commands"""
    pass

@analyze.command()
@click.option('--symbols', '-s', multiple=True, required=True,
              help='Trading symbols to analyze (e.g., ETH/USDT, BTC/USDT)')
@click.option('--confidence', type=click.FloatRange(0.0, 1.0),
              default=0.7, help='Minimum confidence threshold (0.0-1.0)')
@click.option('--time-horizon', type=click.Choice(['1m', '5m', '15m', '1h', '4h', '1d']),
              default='1h', help='Analysis time horizon')
@click.option('--models', multiple=True,
              type=click.Choice(['lstm', 'transformer', 'ensemble', 'all']),
              default=['all'], help='AI models to use')
@click.option('--live', is_flag=True, help='Stream signals in real-time')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.pass_context
def signals(ctx, symbols, confidence, time_horizon, models, live, output):
    """Generate trading signals using AI models

    Generates institutional-grade trading signals using the Colin Trading Bot's
    AI engine with multiple model support and real-time capabilities.
    """

    if live:
        asyncio.run(_stream_signals(ctx, symbols, confidence, time_horizon, models))
    else:
        asyncio.run(_generate_signals_once(ctx, symbols, confidence, time_horizon, models, output))

async def _generate_signals_once(ctx, symbols, confidence, time_horizon, models, output):
    """Generate signals once with rich display"""
    from ..integration import get_bot_instance

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        # Initialize bot
        task = progress.add_task("Initializing Colin Trading Bot...", total=None)
        bot = await get_bot_instance(ctx.obj['config'], ctx.obj['mode'])

        progress.update(task, description="Generating AI signals...")

        # Generate signals
        results = await bot.generate_signals(
            symbols=list(symbols),
            confidence_threshold=confidence,
            time_horizon=time_horizon,
            models=models
        )

        progress.update(task, description="Formatting results...")

        # Display results
        _display_signals_table(results)

        # Save to file if requested
        if output:
            _save_signals_results(results, output)
            console.print(f"[green]✅ Results saved to {output}[/green]")

def _display_signals_table(results):
    """Display signals in a rich table"""
    table = Table(title="AI Trading Signals", show_header=True, header_style="bold magenta")
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Direction", style="bold", width=10)
    table.add_column("Confidence", justify="right", style="green")
    table.add_column("Strength", justify="right", width=10)
    table.add_column("Price Target", justify="right", style="yellow")
    table.add_column("Time Horizon", style="blue")
    table.add_column("Model", style="dim")

    for signal in results:
        direction_emoji = "🟢" if signal.direction == "BUY" else "🔴"
        table.add_row(
            signal.symbol,
            f"{direction_emoji} {signal.direction}",
            f"{signal.confidence:.2%}",
            signal.strength,
            f"${signal.price_target:.2f}",
            signal.time_horizon,
            signal.model
        )

    console.print(table)

@analyze.command()
@click.argument('symbol', required=True)
@click.option('--indicators', multiple=True,
              type=click.Choice(['rsi', 'macd', 'bollinger', 'volume', 'orderbook', 'all']),
              default=['all'], help='Technical indicators to analyze')
@click.option('--period', type=int, default=100, help='Analysis period in candles')
@click.pass_context
def technical(ctx, symbol, indicators, period):
    """Perform technical analysis on a symbol

    Analyzes technical indicators and market structure for comprehensive
    trading insights using Colin Trading Bot's technical analysis engine.
    """
    asyncio.run(_perform_technical_analysis(ctx, symbol, indicators, period))

async def _perform_technical_analysis(ctx, symbol, indicators, period):
    """Execute technical analysis with rich output"""
    from ..integration import get_bot_instance

    with console.status(f"[bold green]Analyzing {symbol} technical indicators..."):
        bot = await get_bot_instance(ctx.obj['config'], ctx.obj['mode'])

        analysis = await bot.analyze_technical(
            symbol=symbol,
            indicators=list(indicators),
            period=period
        )

    # Display comprehensive technical analysis
    _display_technical_analysis(symbol, analysis)
```

### 3. Trading Operations Commands

```python
# src/cli/commands/trade.py
import click
import asyncio
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()

@cli.group()
def trade():
    """Trading operations and order management commands"""
    pass

@trade.command()
@click.option('--symbol', required=True, help='Trading symbol (e.g., ETH/USDT)')
@click.option('--side', type=click.Choice(['buy', 'sell']), required=True)
@click.option('--quantity', type=float, required=True, help='Order quantity')
@click.option('--order-type', type=click.Choice(['market', 'limit', 'stop']),
              default='market', help='Order type')
@click.option('--price', type=float, help='Price for limit/stop orders')
@click.option('--portfolio-percent', type=float, help='Quantity as portfolio percentage')
@click.option('--dry-run', is_flag=True, help='Simulate order without execution')
@click.pass_context
def create(ctx, symbol, side, quantity, order_type, price, portfolio_percent, dry_run):
    """Create and execute a trading order

    Creates and executes trading orders with comprehensive risk validation,
    smart order routing, and real-time monitoring. Supports multiple order
    types and risk management features.
    """

    # Safety confirmation for live trading
    if not ctx.obj['dry_run'] and not dry_run and not ctx.obj['mode'] == 'development':
        if not Confirm.ask(f"[bold red]Execute LIVE {side.upper()} order for {quantity} {symbol}?[/bold red]"):
            console.print("[yellow]Order cancelled[/yellow]")
            return

    asyncio.run(_execute_order(ctx, symbol, side, quantity, order_type,
                                price, portfolio_percent, dry_run))

async def _execute_order(ctx, symbol, side, quantity, order_type, price,
                         portfolio_percent, dry_run):
    """Execute order with comprehensive flow"""
    from ..integration import get_bot_instance
    from ..utils.validators import validate_order_parameters

    # Validate order parameters
    validate_order_parameters(symbol, side, quantity, order_type, price)

    with console.status(f"[bold green]Creating {side.upper()} order for {symbol}..."):
        bot = await get_bot_instance(ctx.obj['config'], ctx.obj['mode'])

        # Create order object
        from src.v2.execution_engine.smart_routing.router import Order, OrderSide, OrderType

        order = Order(
            symbol=symbol,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            order_type=_map_order_type(order_type),
            quantity=quantity,
            price=price or 0.0,
            portfolio_percent=portfolio_percent,
            dry_run=dry_run or ctx.obj['dry_run'],
            metadata={"source": "cli"}
        )

        # Execute order
        result = await bot.execute_order(order)

        # Display results
        _display_order_result(result, dry_run)

def _display_order_result(result, dry_run):
    """Display order execution results"""
    if result['success']:
        status = "🟢 SIMULATED" if dry_run else "🟢 EXECUTED"
        console.print(f"[bold green]{status} Order[/bold green]")

        table = Table(show_header=False, box=None)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Order ID", result['order_id'])
        table.add_row("Symbol", result['symbol'])
        table.add_row("Side", result['side'])
        table.add_row("Quantity", str(result['quantity']))
        table.add_row("Price", f"${result['price']:.2f}")
        table.add_row("Status", result['status'])

        if result.get('fills'):
            table.add_row("Fill Price", f"${result['fills'][0]['price']:.2f}")

        console.print(table)

        if result.get('risk_warnings'):
            for warning in result['risk_warnings']:
                console.print(f"[yellow]⚠️  {warning}[/yellow]")
    else:
        console.print(f"[bold red]❌ Order Failed: {result['error']}[/bold red]")

@trade.command()
@click.option('--status', type=click.Choice(['open', 'filled', 'cancelled', 'all']),
              default='open', help='Filter orders by status')
@click.option('--symbol', help='Filter by symbol')
@click.option('--limit', type=int, default=20, help='Maximum orders to display')
@click.pass_context
def list(ctx, status, symbol, limit):
    """List and monitor trading orders"""
    asyncio.run(_list_orders(ctx, status, symbol, limit))

async def _list_orders(ctx, status, symbol, limit):
    """List orders with rich display"""
    from ..integration import get_bot_instance

    with console.status("[bold green]Fetching orders..."):
        bot = await get_bot_instance(ctx.obj['config'], ctx.obj['mode'])

        orders = await bot.get_orders(
            status=status,
            symbol=symbol,
            limit=limit
        )

    if orders:
        _display_orders_table(orders)
    else:
        console.print("[yellow]No orders found[/yellow]")

def _display_orders_table(orders):
    """Display orders in table format"""
    table = Table(title="Trading Orders")
    table.add_column("Order ID", style="cyan")
    table.add_column("Symbol", style="white")
    table.add_column("Side", style="bold")
    table.add_column("Type", style="blue")
    table.add_column("Quantity", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Status", style="magenta")
    table.add_column("Time", style="dim")

    for order in orders:
        side_emoji = "🟢" if order['side'] == 'BUY' else "🔴"
        status_color = {
            'open': 'yellow',
            'filled': 'green',
            'cancelled': 'red'
        }.get(order['status'], 'white')

        table.add_row(
            order['id'][:8] + "...",
            order['symbol'],
            f"{side_emoji} {order['side']}",
            order['type'],
            str(order['quantity']),
            f"${order['price']:.2f}",
            f"[{status_color}]{order['status']}[/{status_color}]",
            order['timestamp']
        )

    console.print(table)
```

### 4. Configuration Management Commands

```python
# src/cli/commands/config.py
import click
import yaml
import json
from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()

@cli.group()
def config():
    """Configuration management commands"""
    pass

@config.command()
@click.option('--format', type=click.Choice(['yaml', 'json']),
              default='yaml', help='Output format')
@click.option('--section', help='Specific configuration section to show')
@click.pass_context
def show(ctx, format, section):
    """Display current configuration"""

    config_path = _get_config_path(ctx.obj['config'])

    if not config_path.exists():
        console.print(f"[red]❌ Configuration file not found: {config_path}[/red]")
        console.print("[yellow]💡 Run 'colin config init' to create configuration[/yellow]")
        return

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    if section:
        if section not in config_data:
            console.print(f"[red]❌ Configuration section '{section}' not found[/red]")
            return
        config_data = {section: config_data[section]}

    # Display configuration with syntax highlighting
    if format == 'yaml':
        syntax = Syntax(yaml.dump(config_data, default_flow_style=False),
                      "yaml", theme="monokai", line_numbers=True)
    else:
        syntax = Syntax(json.dumps(config_data, indent=2),
                      "json", theme="monokai", line_numbers=True)

    console.print(Panel(syntax, title=f"Configuration ({format.upper()})"))

@config.command()
@click.option('--key', help='Specific configuration key to validate')
@click.option('--strict', is_flag=True, help='Treat warnings as errors')
@click.pass_context
def validate(ctx, key, strict):
    """Validate configuration settings"""

    config_path = _get_config_path(ctx.obj['config'])

    with console.status("[bold green]Validating configuration..."):
        try:
            from colin_bot.v2.config.main_config import MainConfigManager

            config_manager = MainConfigManager(str(config_path))
            issues = config_manager.validate_configuration()

            if not issues:
                console.print("[green]✅ Configuration is valid[/green]")
            else:
                console.print("[yellow]⚠️  Configuration validation issues found:[/yellow]")
                for issue in issues:
                    console.print(f"  • {issue}")

                if strict:
                    console.print("[red]❌ Strict validation failed[/red]")
                    return

            if key:
                _validate_specific_key(config_manager.config, key)

        except Exception as e:
            console.print(f"[red]❌ Configuration validation failed: {e}[/red]")
            return 1

@config.command()
@click.option('--environment', type=click.Choice(['development', 'staging', 'production']),
              default='development', help='Target environment')
@click.option('--interactive', is_flag=True, help='Interactive configuration setup')
@click.option('--force', is_flag=True, help='Overwrite existing configuration')
@click.pass_context
def init(ctx, environment, interactive, force):
    """Initialize new configuration"""

    config_path = Path.cwd() / f"config.{environment}.yaml"

    if config_path.exists() and not force:
        if not Confirm.ask(f"Configuration {config_path} exists. Overwrite?"):
            return

    if interactive:
        _interactive_config_setup(config_path, environment)
    else:
        _generate_default_config(config_path, environment)

    console.print(f"[green]✅ Configuration initialized: {config_path}[/green]")

@config.command()
@click.argument('key_value', required=True)
@click.pass_context
def set(ctx, key_value):
    """Set configuration value

    Examples:
    colin config set trading.enabled=true
    colin config set risk.max_position_size=100000
    colin config set ai_models.lstm.enabled=true
    """

    try:
        key, value = key_value.split('=', 1)

        # Convert value to appropriate type
        if value.lower() in ('true', 'false'):
            value = value.lower() == 'true'
        elif value.replace('.', '').isdigit():
            value = float(value) if '.' in value else int(value)

        _set_config_value(key, value, ctx.obj['config'])
        console.print(f"[green]✅ Set {key} = {value}[/green]")

    except ValueError:
        console.print("[red]❌ Invalid key=value format. Use: key=value[/red]")
        return 1

def _set_config_value(key: str, value, config_path: str):
    """Set configuration value with nested key support"""
    config_path = Path(config_path) or Path.cwd() / "config.development.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Handle nested keys (e.g., "risk.max_position_size")
    keys = key.split('.')
    current = config

    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    current[keys[-1]] = value

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
```

### 5. Interactive Trading Session

```python
# src/cli/interactive.py
import click
import asyncio
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.prompt import Prompt, Confirm
from rich.text import Text

console = Console()

class InteractiveTradingSession:
    """Interactive trading session with real-time updates"""

    def __init__(self, bot, symbols=None):
        self.bot = bot
        self.symbols = symbols or []
        self.running = True

    async def start(self):
        """Start interactive trading session"""

        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )

        layout["main"].split_row(
            Layout(name="portfolio", ratio=1),
            Layout(name="signals", ratio=1),
            Layout(name="orders", ratio=1),
            Layout(name="risk", ratio=1)
        )

        # Update layout components
        self._update_header(layout)
        self._update_footer(layout)

        # Start live display
        with Live(layout, refresh_per_second=2) as live:

            # Command loop
            while self.running:
                try:
                    command = Prompt.ask(
                        "[bold green]colin>[/bold green]",
                        choices=["signals", "orders", "portfolio", "risk", "analyze", "quit", "help"]
                    )

                    if command == "quit":
                        self.running = False
                        break
                    elif command == "help":
                        self._show_help()
                    elif command == "signals":
                        await self._handle_signals_command(live)
                    elif command == "orders":
                        await self._handle_orders_command(live)
                    elif command == "portfolio":
                        await self._update_portfolio_panel(live)
                    elif command == "risk":
                        await self._update_risk_panel(live)
                    elif command == "analyze":
                        await self._handle_analyze_command(live)

                except KeyboardInterrupt:
                    self.running = False
                    break

    def _update_header(self, layout):
        """Update header panel"""
        header_text = Text.from_markup(
            "[bold blue]Colin Trading Bot - Interactive Session[/bold blue]\n"
            f"[green]Symbols: {', '.join(self.symbols) if self.symbols else 'None'}[/green]"
        )
        layout["header"].update(Panel(header_text, border_style="blue"))

    def _update_footer(self, layout):
        """Update footer panel"""
        footer_text = Text.from_markup(
            "[dim]Commands: signals | orders | portfolio | risk | analyze | help | quit[/dim]"
        )
        layout["footer"].update(Panel(footer_text, border_style="dim"))

    async def _handle_signals_command(self, live):
        """Handle signals-related commands"""
        action = Prompt.ask(
            "Signals action",
            choices=["generate", "analyze", "list", "back"]
        )

        if action == "generate":
            if not self.symbols:
                symbols = Prompt.ask("Enter symbols (comma-separated)").split(",")
            else:
                symbols = self.symbols

            confidence = Prompt.ask("Min confidence", default="0.7")

            with live:  # Type hint for live display
                # Generate signals
                results = await self.bot.generate_signals(
                    symbols=symbols,
                    confidence_threshold=float(confidence)
                )

                # Update signals panel
                self._update_signals_panel(live, results)

    def _update_signals_panel(self, live, signals):
        """Update signals display panel"""
        table = Table(title="Trading Signals", show_header=True)
        table.add_column("Symbol", style="cyan")
        table.add_column("Direction", style="bold")
        table.add_column("Confidence", justify="right", style="green")
        table.add_column("Price", justify="right")

        for signal in signals:
            direction_emoji = "🟢" if signal.direction == "BUY" else "🔴"
            table.add_row(
                signal.symbol,
                f"{direction_emoji} {signal.direction}",
                f"{signal.confidence:.2%}",
                f"${signal.price:.2f}"
            )

        live.layout["signals"].update(Panel(table, border_style="green"))

@cli.command()
@click.option('--symbols', multiple=True, help='Default symbols to monitor')
@click.option('--refresh-rate', type=int, default=2, help='Display refresh rate (seconds)')
@click.pass_context
def interactive(ctx, symbols, refresh_rate):
    """Start interactive trading session

    Launches an interactive terminal-based trading interface with real-time
    updates, command support, and comprehensive market monitoring.
    """

    asyncio.run(_start_interactive_session(ctx, symbols, refresh_rate))

async def _start_interactive_session(ctx, symbols, refresh_rate):
    """Start interactive session with bot initialization"""
    from ..integration import get_bot_instance

    with console.status("[bold green]Initializing Colin Trading Bot..."):
        bot = await get_bot_instance(ctx.obj['config'], ctx.obj['mode'])

    console.print("[green]✅ Bot initialized successfully[/green]")

    # Start interactive session
    session = InteractiveTradingSession(bot, list(symbols))
    await session.start()
```

### 6. Security and Credential Management

```python
# src/cli/commands/security.py
import click
import keyring
from cryptography.fernet import Fernet
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

console = Console()

@cli.group()
def security():
    """Security and credential management commands"""
    pass

@security.command()
@click.option('--exchange', required=True,
              type=click.Choice(['binance', 'bybit', 'okx', 'deribit', 'kraken']),
              help='Exchange name')
@click.option('--testnet', is_flag=True, help='Store testnet credentials')
@click.pass_context
def store_credentials(ctx, exchange, testnet):
    """Securely store API credentials

    Stores API credentials in the system keychain with encryption for
    secure access during trading operations.
    """

    service_name = f"colin_bot_{exchange}_{'testnet' if testnet else 'main'}"

    # Get credentials securely
    api_key = Prompt.ask(f"Enter {exchange} API Key", password=True)
    api_secret = Prompt.ask(f"Enter {exchange} API Secret", password=True)

    # Validate format
    if not _validate_api_credentials(exchange, api_key, api_secret):
        console.print("[red]❌ Invalid API credential format[/red]")
        return 1

    # Store securely
    try:
        secure_manager = SecureCredentialManager()
        secure_manager.store_credentials(service_name, api_key, api_secret)

        console.print(f"[green]✅ Credentials securely stored for {exchange}[/green]")

        if testnet:
            console.print("[blue]🧪 Testnet credentials stored[/blue]")

    except Exception as e:
        console.print(f"[red]❌ Failed to store credentials: {e}[/red]")
        return 1

@security.command()
@click.option('--exchange', required=True,
              type=click.Choice(['binance', 'bybit', 'okx', 'deribit', 'kraken']),
              help='Exchange name')
@click.option('--testnet', is_flag=True, help='Use testnet credentials')
def test_credentials(exchange, testnet):
    """Test API credentials connectivity"""

    service_name = f"colin_bot_{exchange}_{'testnet' if testnet else 'main'}"

    try:
        secure_manager = SecureCredentialManager()
        api_key, api_secret = secure_manager.get_credentials(service_name)

        with console.status(f"[bold green]Testing {exchange} connectivity..."):
            # Test API connection
            test_result = await _test_exchange_connection(exchange, api_key, api_secret, testnet)

        if test_result['success']:
            console.print(f"[green]✅ {exchange} credentials are valid[/green]")
            _display_connection_info(test_result)
        else:
            console.print(f"[red]❌ {exchange} credentials test failed: {test_result['error']}[/red]")
            return 1

    except Exception as e:
        console.print(f"[red]❌ Failed to retrieve or test credentials: {e}[/red]")
        return 1

class SecureCredentialManager:
    """Secure credential management with encryption"""

    def __init__(self):
        self.cipher_suite = Fernet(self._get_or_create_key())

    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key from keyring"""
        service_name = "colin_trading_bot"
        key = keyring.get_password(service_name, "encryption_key")

        if not key:
            key = Fernet.generate_key().decode()
            keyring.set_password(service_name, "encryption_key", key)

        return key.encode()

    def store_credentials(self, service_name: str, api_key: str, api_secret: str):
        """Store encrypted credentials in keyring"""
        encrypted_key = self.cipher_suite.encrypt(api_key.encode())
        encrypted_secret = self.cipher_suite.encrypt(api_secret.encode())

        keyring.set_password(service_name, "api_key", encrypted_key.decode())
        keyring.set_password(service_name, "api_secret", encrypted_secret.decode())

    def get_credentials(self, service_name: str) -> tuple[str, str]:
        """Retrieve and decrypt credentials from keyring"""
        encrypted_key = keyring.get_password(service_name, "api_key")
        encrypted_secret = keyring.get_password(service_name, "api_secret")

        if not encrypted_key or not encrypted_secret:
            raise ValueError(f"No credentials found for {service_name}")

        api_key = self.cipher_suite.decrypt(encrypted_key.encode()).decode()
        api_secret = self.cipher_suite.decrypt(encrypted_secret.encode()).decode()

        return api_key, api_secret

def _validate_api_credentials(exchange: str, api_key: str, api_secret: str) -> bool:
    """Validate API credential format for specific exchange"""
    validation_patterns = {
        'binance': r'^[A-Za-z0-9]{64}$',
        'bybit': r'^[A-Za-z0-9]{64}$',
        'okx': r'^[A-Za-z0-9-]{32,64}$',
        'deribit': r'^[A-Za-z0-9-]{32}$',
        'kraken': r'^[A-Za-z0-9+/]{64,87}==$'
    }

    import re
    pattern = validation_patterns.get(exchange.lower())

    if pattern:
        key_valid = bool(re.match(pattern, api_key))
        secret_valid = len(api_secret) >= 32
        return key_valid and secret_valid

    # Generic validation
    return len(api_key) >= 32 and len(api_secret) >= 32
```

## Implementation Tasks

### Task 1: Core CLI Infrastructure
**File**: `src/cli/main.py`
- Implement main Click application with rich branding
- Add global options (config, verbose, dry-run, mode)
- Create command group structure
- Implement error handling decorators
- Add context management for bot instances

### Task 2: Analysis Commands Module
**File**: `src/cli/commands/analyze.py`
- Implement `analyze signals` command with AI model selection
- Add `analyze technical` command for technical analysis
- Create `analyze sentiment` command for market sentiment
- Implement real-time signal streaming capabilities
- Add rich table formatting for signal display

### Task 3: Trading Operations Module
**File**: `src/cli/commands/trade.py`
- Implement `trade create` command with order types
- Add `trade list` command for order monitoring
- Create `trade cancel` command for order management
- Implement risk validation for all trading operations
- Add portfolio percentage sizing

### Task 4: Configuration Management
**File**: `src/cli/commands/config.py`
- Implement `config show` command with syntax highlighting
- Add `config validate` command with strict validation
- Create `config init` command for setup wizard
- Implement `config set` command for runtime updates
- Add configuration backup and restore

### Task 5: Interactive Session Manager
**File**: `src/cli/interactive.py`
- Implement InteractiveTradingSession class
- Create rich layout management for real-time displays
- Add command processing and panel updates
- Implement real-time signal streaming in interactive mode
- Add portfolio and risk monitoring panels

### Task 6: Security and Credential Management
**File**: `src/cli/commands/security.py`
- Implement SecureCredentialManager class
- Add `security store-credentials` command
- Create `security test-credentials` command
- Implement encrypted keyring storage
- Add credential validation and testing

### Task 7: Bot Integration Layer
**File**: `src/cli/integration.py`
- Implement bot context manager for lifecycle management
- Create bot instance factory with configuration
- Add async command execution wrapper
- Implement error handling and user feedback
- Add mode-specific behavior (development/staging/production)

### Task 8: Utilities and Formatters
**Files**: `src/cli/utils/`
- `error_handling.py`: Comprehensive error handling decorators
- `validators.py`: Input validation for trading parameters
- `formatters.py`: Rich formatting utilities for tables and panels
- `security.py`: Security utilities and validation functions

### Task 9: Configuration Integration
**File**: `setup.py` and `pyproject.toml`
- Add CLI entry point configuration
- Update dependencies to include Click, Rich, Keyring
- Configure script installation
- Add console script: `colin = src.cli.main:cli`

### Task 10: Testing Infrastructure
**Files**: `tests/cli/`
- Create unit tests for all CLI commands
- Add integration tests for bot interaction
- Implement CLI testing utilities
- Add security testing for credential management
- Create end-to-end testing scenarios

## Code Patterns to Follow

### Error Handling Pattern
```python
from src.cli.utils.error_handling import handle_cli_errors, TradingBotError

@handle_cli_errors
def my_command():
    try:
        # Command logic
        pass
    except ConfigurationError as e:
        console.print(f"[red]❌ Configuration Error: {e}[/red]")
        console.print("[yellow]💡 Try: colin config validate[/yellow]")
```

### Rich Display Pattern
```python
from rich.console import Console
from rich.table import Table

console = Console()

def display_results_table(data):
    table = Table(title="Results")
    table.add_column("Column 1", style="cyan")
    table.add_column("Column 2", style="green")
    # Add rows...
    console.print(table)
```

### Async Command Pattern
```python
import asyncio

@click.command()
def my_command():
    """Command with async operations"""
    asyncio.run(_async_command_logic())

async def _async_command_logic():
    # Async implementation
    pass
```

## Critical Considerations

### Security Requirements
- API credentials must be encrypted and stored securely
- No plaintext credential storage in configuration files
- Keyring integration for cross-platform security
- Testnet and production credential separation
- Audit logging for all credential access

### Error Handling Strategy
- Specific error types for trading operations
- Rich error formatting with helpful suggestions
- Graceful degradation for non-critical failures
- Comprehensive logging for debugging
- User-friendly error messages with actionable advice

### Performance Considerations
- Lazy loading of bot components only when needed
- Efficient real-time display updates
- Background task management for long operations
- Memory usage optimization for interactive sessions
- Responsive CLI interface during operations

### User Experience Requirements
- Rich terminal formatting with colors and tables
- Progress indicators for long operations
- Interactive confirmation for critical operations
- Comprehensive help documentation
- Command auto-completion support

## Validation Gates

### Pre-Implementation Validation
```bash
# Check dependencies
cd "/Users/gdove/Desktop/DEEPs_Colin_TradingBot copy"
source venv_colin_bot/bin/activate

# Install CLI dependencies
pip install click rich keyring cryptography

# Validate existing code structure
python -c "
import colin_bot.v2.config.main_config
print('✅ V2 configuration accessible')
"

# Validate existing bot integration
python -c "
import sys
sys.path.append('src')
from v2.main import ColinTradingBotV2
print('✅ V2 bot accessible')
"
```

### Implementation Validation
```bash
# Syntax and import checking
cd "/Users/gdove/Desktop/DEEPs_Colin_TradingBot copy"
source venv_colin_bot/bin/activate

# Check CLI syntax
python -m py_compile src/cli/main.py
python -m py_compile src/cli/commands/*.py
python -m py_compile src/cli/interactive.py

# Type checking
python -m mypy src/cli/ --ignore-missing-imports

# CLI functionality testing
python -m src.cli.main --help
python -m src.cli.main config --help
python -m src.cli.main analyze --help
```

### Integration Testing
```bash
# Test CLI integration with bot
source venv_colin_bot/bin/activate

# Test configuration commands
python -m src.cli.main config init --environment development
python -m src.cli.main config validate

# Test analysis commands
python -m src.cli.main analyze signals --symbols ETH/USDT --dry-run

# Test security commands
python -m src.cli.main security --help

# Test interactive mode
python -m src.cli.main interactive --help
```

### End-to-End Validation
```bash
# Complete CLI installation test
cd "/Users/gdove/Desktop/DEEPs_Colin_TradingBot copy"
source venv_colin_bot/bin/activate

# Install package in development mode
pip install -e .

# Test global command availability
colin --help
colin config --help
colin analyze --help

# Test full workflow
colin config init --environment development
colin config validate
colin analyze signals --symbols ETH/USDT --confidence 0.8 --dry-run
colin security store-credentials --exchange binance --testnet
colin security test-credentials --exchange binance --testnet
```

## Success Metrics

### Functional Requirements
- ✅ Unified `colin` command entry point
- ✅ Hierarchical command structure with logical grouping
- ✅ Rich terminal formatting with tables and progress indicators
- ✅ Interactive trading session with real-time updates
- ✅ Secure credential management with encryption
- ✅ Comprehensive configuration management via CLI
- ✅ Integration with existing v2 bot architecture
- ✅ Error handling with specific trading error types
- ✅ Dry-run mode for safe testing
- ✅ Command auto-completion and help documentation

### Quality Metrics
- ✅ All commands have comprehensive help documentation
- ✅ Error messages provide actionable feedback
- ✅ Configuration validation and management
- ✅ Security best practices for credential handling
- ✅ Rich user experience with professional formatting
- ✅ Async command execution for bot integration
- ✅ Unit test coverage >90% for CLI components
- ✅ Integration tests for all major workflows

### Performance Requirements
- ✅ CLI startup time <2 seconds
- ✅ Command response time <1 second for non-network operations
- ✅ Interactive session updates at 2-second intervals
- ✅ Memory usage <50MB for interactive sessions
- ✅ Concurrent command support where applicable

## Risk Mitigation

### Technical Risks
1. **Bot Integration Complexity**: Implement thorough integration testing and error handling
2. **CLI Dependency Management**: Pin specific versions and provide fallback options
3. **Security Implementation**: Use well-vetted libraries (keyring, cryptography) with proven track records
4. **Performance Impact**: Implement lazy loading and background task management

### Operational Risks
1. **User Experience**: Provide comprehensive help documentation and error messages
2. **Credential Security**: Implement secure defaults and validation for all credential operations
3. **Configuration Management**: Provide configuration validation and backup/restore functionality
4. **Interactive Session Stability**: Implement robust error handling and session recovery

## PRP Quality Assessment

**Confidence Score: 9/10** for one-pass implementation success

**Strengths:**
- ✅ Comprehensive research of CLI best practices and libraries
- ✅ Detailed integration plan with existing Colin Trading Bot architecture
- ✅ Well-defined hierarchical command structure
- ✅ Rich user experience design with interactive capabilities
- ✅ Security-first approach with encrypted credential management
- ✅ Complete error handling and validation strategy
- ✅ Executable validation gates for quality assurance
- ✅ Professional terminal interface design with Rich formatting

**Potential Challenges:**
- Bot integration complexity requiring careful async handling
- Security implementation details requiring thorough testing
- Rich display management performance optimization
- Cross-platform compatibility for keyring integration

This PRP provides sufficient context and detail for successful one-pass implementation while leveraging modern CLI best practices and the existing Colin Trading Bot architecture. The design prioritizes user experience, security, and integration with the sophisticated v2 institutional trading platform.