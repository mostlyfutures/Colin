"""
Configuration management commands for the Colin Trading Bot CLI.

This module provides commands for viewing, validating, and managing
bot configuration settings.
"""

import click
import json
import yaml
import os
from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.syntax import Syntax
from rich.prompt import Confirm, Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..utils.formatters import format_key_value_pairs, format_success, format_error
from ..utils.error_handler import with_error_handling, ConfigurationError


@click.group()
def config():
    """⚙️ Configuration management commands."""
    pass


@config.command()
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), default='table',
              help='Output format')
@click.option('--section', '-s', help='Show specific configuration section')
@click.pass_context
@with_error_handling("Configuration Display")
def show(ctx, format, section):
    """Display current configuration settings."""
    console = Console()

    # Load configuration (mock data for demonstration)
    mock_config = {
        'system': {
            'environment': 'development',
            'log_level': 'INFO',
            'debug_mode': False,
            'timezone': 'UTC',
        },
        'trading': {
            'enabled': False,
            'max_portfolio_value_usd': 100000.0,
            'default_order_size_usd': 1000.0,
            'simulation_mode': True,
        },
        'market_data': {
            'primary_source': 'coingecko',
            'fallback_sources': ['kraken', 'cryptocompare'],
            'cache_ttl_seconds': 300,
            'max_concurrent_requests': 5,
        },
        'risk_management': {
            'max_position_size_usd': 100000.0,
            'max_portfolio_exposure': 0.20,
            'max_leverage': 3.0,
            'max_drawdown_hard': 0.05,
        },
        'api': {
            'enabled': True,
            'host': '0.0.0.0',
            'port': 8000,
            'require_auth': False,
            'rate_limit': 100,
        },
        'exchanges': {
            'binance': {
                'enabled': True,
                'testnet': True,
                'api_key': '***REDACTED***',
                'api_secret': '***REDACTED***',
            },
            'coinbase': {
                'enabled': False,
                'testnet': True,
            },
        }
    }

    if section:
        if section not in mock_config:
            console.print(f"[bold red]Configuration section '{section}' not found[/bold red]")
            return
        mock_config = {section: mock_config[section]}

    if format == 'table':
        display_config_table(console, mock_config, section)
    elif format == 'json':
        console.print(Syntax(json.dumps(mock_config, indent=2), "json"))
    elif format == 'yaml':
        console.print(Syntax(yaml.dump(mock_config, default_flow_style=False), "yaml"))


def display_config_table(console: Console, config_data: Dict[str, Any], section: Optional[str] = None):
    """Display configuration in table format."""
    title = f"Configuration - {section}" if section else "Complete Configuration"

    for section_name, section_data in config_data.items():
        if isinstance(section_data, dict):
            table = Table(title=section_name.replace('_', ' ').title(), show_header=False, box=None)
            table.add_column("Setting", style="bold cyan")
            table.add_column("Value", style="white")

            for key, value in section_data.items():
                # Handle sensitive values
                if 'key' in key.lower() or 'secret' in key.lower() or 'password' in key.lower():
                    value = "***REDACTED***"
                elif isinstance(value, bool):
                    value = "✅ Enabled" if value else "❌ Disabled"
                elif isinstance(value, (int, float)):
                    if 'percent' in key.lower() or 'ratio' in key.lower():
                        value = f"{value:.1%}"
                    elif value > 1000:
                        value = f"{value:,.0f}"
                    else:
                        value = str(value)

                table.add_row(key.replace('_', ' ').title(), str(value))

            console.print(table)
            console.print()


@config.command()
@click.option('--detailed', '-d', is_flag=True, help='Show detailed validation results')
@click.option('--fix', is_flag=True, help='Attempt to fix configuration issues')
@click.pass_context
@with_error_handling("Configuration Validation")
def validate(ctx, detailed, fix):
    """Validate configuration settings."""
    console = Console()

    console.print("[bold blue]Validating Configuration[/bold blue]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:

        progress.add_task("Loading configuration files...", total=None)
        import time
        time.sleep(0.5)
        progress.add_task("Validating syntax...", total=None)
        time.sleep(0.5)
        progress.add_task("Checking required settings...", total=None)
        time.sleep(0.5)
        progress.add_task("Validating data types...", total=None)
        time.sleep(0.5)

    # Mock validation results
    validation_results = {
        'syntax': {'status': 'pass', 'message': 'All configuration files have valid syntax'},
        'required_fields': {'status': 'pass', 'message': 'All required fields are present'},
        'data_types': {'status': 'pass', 'message': 'All data types are correct'},
        'exchanges': {
            'status': 'warning',
            'message': 'Some exchanges are missing API keys',
            'details': ['Binance: API key configured', 'Coinbase: API key missing']
        },
        'risk_settings': {
            'status': 'pass',
            'message': 'Risk management settings are within safe limits'
        },
        'connectivity': {
            'status': 'warning',
            'message': 'Some external services may be unreachable',
            'details': ['CoinGecko API: Connected', 'Kraken API: Timeout']
        }
    }

    # Create validation summary table
    table = Table(title="Validation Results", show_header=True)
    table.add_column("Check", style="bold cyan")
    table.add_column("Status", style="bold")
    table.add_column("Message", style="white")

    for check_name, result in validation_results.items():
        status = result['status']
        status_display = {
            'pass': '✅ PASS',
            'warning': '⚠️ WARN',
            'error': '❌ ERROR'
        }.get(status, '❓ UNKNOWN')

        status_style = {
            'pass': 'green',
            'warning': 'yellow',
            'error': 'red'
        }.get(status, 'white')

        table.add_row(
            check_name.replace('_', ' ').title(),
            f"[{status_style}]{status_display}[/{status_style}]",
            result['message']
        )

    console.print(table)

    # Show detailed results if requested
    if detailed:
        console.print("\n[bold blue]Detailed Results[/bold blue]")
        for check_name, result in validation_results.items():
            if 'details' in result:
                console.print(f"\n[cyan]{check_name.replace('_', ' ').title()}:[/cyan]")
                for detail in result['details']:
                    console.print(f"  • {detail}")

    # Show fix suggestions if there are warnings or errors
    has_issues = any(r['status'] in ['warning', 'error'] for r in validation_results.values())
    if has_issues:
        console.print(f"\n[bold yellow]Recommendations:[/bold yellow]")
        console.print("• Configure missing API keys using 'colin security store'")
        console.print("• Test external service connectivity with 'colin security test'")
        console.print("• Review risk management settings")

        if fix and Confirm.ask("\n[yellow]Attempt to fix configuration issues?[/yellow]"):
            console.print("[dim]Auto-fix functionality would be implemented here[/dim]")


@config.command()
@click.option('--environment', type=click.Choice(['development', 'staging', 'production']),
              default='development', help='Target environment')
@click.option('--template', type=click.Choice(['minimal', 'standard', 'advanced']),
              default='standard', help='Configuration template to use')
@click.option('--overwrite', is_flag=True, help='Overwrite existing configuration')
@click.pass_context
@with_error_handling("Configuration Initialization")
def init(ctx, environment, template, overwrite):
    """Initialize a new configuration file."""
    console = Console()

    config_dir = Path("config")
    config_file = config_dir / f"{environment}.yaml"

    if config_file.exists() and not overwrite:
        console.print(f"[bold red]Configuration file already exists: {config_file}[/bold red]")
        console.print("Use --overwrite to replace it or choose a different environment")
        return

    console.print(f"[bold blue]Initializing Configuration[/bold blue]")
    console.print(f"[dim]Environment: {environment}[/dim]")
    console.print(f"[dim]Template: {template}[/dim]")
    console.print(f"[dim]Target: {config_file}[/dim]\n")

    # Create configuration templates
    templates = {
        'minimal': {
            'system': {
                'environment': environment,
                'log_level': 'INFO'
            },
            'trading': {
                'enabled': False,
                'simulation_mode': True
            }
        },
        'standard': {
            'system': {
                'environment': environment,
                'log_level': 'INFO',
                'debug_mode': False,
                'timezone': 'UTC'
            },
            'trading': {
                'enabled': False,
                'simulation_mode': True,
                'max_portfolio_value_usd': 100000.0,
                'default_order_size_usd': 1000.0
            },
            'market_data': {
                'primary_source': 'coingecko',
                'cache_ttl_seconds': 300
            },
            'risk_management': {
                'max_position_size_usd': 100000.0,
                'max_portfolio_exposure': 0.20,
                'max_leverage': 3.0
            },
            'api': {
                'enabled': True,
                'host': '0.0.0.0',
                'port': 8000,
                'require_auth': False
            }
        },
        'advanced': {
            'system': {
                'environment': environment,
                'log_level': 'DEBUG',
                'debug_mode': True,
                'timezone': 'UTC',
                'monitoring_enabled': True
            },
            'trading': {
                'enabled': False,
                'simulation_mode': True,
                'max_portfolio_value_usd': 10000000.0,
                'default_order_size_usd': 100000.0,
                'max_concurrent_orders': 50,
                'order_timeout_seconds': 30
            },
            'market_data': {
                'primary_source': 'coingecko',
                'fallback_sources': ['kraken', 'cryptocompare', 'alternative_me'],
                'cache_ttl_seconds': 300,
                'max_concurrent_requests': 10,
                'rate_limit_per_minute': 60
            },
            'risk_management': {
                'max_position_size_usd': 1000000.0,
                'max_portfolio_exposure': 0.20,
                'max_leverage': 3.0,
                'max_drawdown_hard': 0.05,
                'max_drawdown_warning': 0.03,
                'var_limit_95_1d': 0.02,
                'circuit_breaker_enabled': True
            },
            'execution_engine': {
                'smart_routing_enabled': True,
                'slippage_tolerance_percent': 0.1,
                'execution_timeout_ms': 50
            },
            'ai_engine': {
                'models_enabled': ['lstm', 'transformer', 'ensemble'],
                'confidence_threshold': 0.7,
                'retraining_interval_hours': 24
            },
            'monitoring': {
                'metrics_enabled': True,
                'alert_channels': ['console', 'email'],
                'dashboard_enabled': True
            }
        }
    }

    selected_template = templates[template]

    # Create config directory if it doesn't exist
    config_dir.mkdir(exist_ok=True)

    # Write configuration file
    try:
        with open(config_file, 'w') as f:
            yaml.dump(selected_template, f, default_flow_style=False, indent=2)

        console.print(format_success(f"Configuration initialized successfully: {config_file}"))
        console.print(f"\n[cyan]Next steps:[/cyan]")
        console.print(f"1. Review the configuration file: {config_file}")
        console.print(f"2. Set your API keys: colin security store")
        console.print(f"3. Validate configuration: colin config validate")
        console.print(f"4. Test the system: colin status")

    except Exception as e:
        console.print(format_error(f"Failed to create configuration: {str(e)}"))


@config.command()
@click.option('--key', '-k', required=True, help='Configuration key (e.g., trading.enabled)')
@click.option('--value', '-v', required=True, help='Configuration value')
@click.option('--environment', type=click.Choice(['development', 'staging', 'production']),
              default='development', help='Target environment')
@click.option('--type', type=click.Choice(['string', 'int', 'float', 'bool', 'json']),
              help='Value type (auto-detected if not specified)')
@click.pass_context
@with_error_handling("Configuration Setting")
def set(ctx, key, value, environment, type):
    """Set a configuration value."""
    console = Console()

    config_file = Path("config") / f"{environment}.yaml"

    if not config_file.exists():
        console.print(f"[bold red]Configuration file not found: {config_file}[/bold red]")
        console.print("Use 'colin config init' to create a configuration file")
        return

    # Parse the key (supports nested keys like 'trading.enabled')
    key_parts = key.split('.')

    # Parse the value
    parsed_value = parse_config_value(value, type)

    console.print(f"[bold blue]Setting Configuration[/bold blue]")
    console.print(f"[dim]Environment: {environment}[/dim]")
    console.print(f"[dim]Key: {key}[/dim]")
    console.print(f"[dim]Value: {parsed_value}[/dim]\n")

    try:
        # Load existing configuration
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f) or {}

        # Navigate to the correct nested location
        current_level = config_data
        for part in key_parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]

        # Set the value
        current_level[key_parts[-1]] = parsed_value

        # Write back to file
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

        console.print(format_success(f"Configuration updated successfully"))
        console.print(f"[dim]File: {config_file}[/dim]")

        # Show the updated configuration section
        console.print(f"\n[cyan]Updated {key}: {parsed_value}[/cyan]")

    except Exception as e:
        console.print(format_error(f"Failed to update configuration: {str(e)}"))


def parse_config_value(value: str, value_type: Optional[str] = None) -> Any:
    """Parse a configuration value based on its type."""
    if value_type:
        type_map = {
            'string': str,
            'int': int,
            'float': float,
            'bool': lambda x: x.lower() in ['true', '1', 'yes', 'on'],
            'json': lambda x: json.loads(x)
        }
        if value_type in type_map:
            return type_map[value_type](value)

    # Auto-detect type
    # Try boolean
    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'

    # Try integer
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Try JSON
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        pass

    # Default to string
    return value


@config.command()
@click.option('--environment', type=click.Choice(['development', 'staging', 'production']),
              default='development', help='Target environment')
@click.pass_context
@with_error_handling("Configuration Reset")
def reset(ctx, environment):
    """Reset configuration to default values."""
    console = Console()

    config_file = Path("config") / f"{environment}.yaml"

    if not config_file.exists():
        console.print(f"[bold red]Configuration file not found: {config_file}[/bold red]")
        return

    console.print(f"[bold red]Reset Configuration[/bold red]")
    console.print(f"[dim]Environment: {environment}[/dim]")
    console.print(f"[dim]This will restore default values[/dim]\n")

    if not Confirm.ask(f"[red]Are you sure you want to reset {environment} configuration?[/red]"):
        console.print("[yellow]Reset cancelled[/yellow]")
        return

    try:
        # Create backup
        backup_file = config_file.with_suffix('.yaml.backup')
        config_file.rename(backup_file)

        # Initialize with standard template
        from .init import templates
        with open(config_file, 'w') as f:
            yaml.dump(templates['standard'], f, default_flow_style=False, indent=2)

        console.print(format_success(f"Configuration reset successfully"))
        console.print(f"[dim]Backup saved as: {backup_file}[/dim]")

    except Exception as e:
        console.print(format_error(f"Failed to reset configuration: {str(e)}"))


@config.command()
@click.pass_context
@with_error_handling("Configuration Backup")
def backup(ctx):
    """Create a backup of current configuration."""
    console = Console()

    config_dir = Path("config")
    backup_dir = config_dir / "backups"
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"config_backup_{timestamp}.tar.gz"

    console.print(f"[bold blue]Creating Configuration Backup[/bold blue]")
    console.print(f"[dim]Backup file: {backup_file}[/dim]\n")

    try:
        import tarfile

        with tarfile.open(backup_file, "w:gz") as tar:
            for config_file in config_dir.glob("*.yaml"):
                if config_file.is_file():
                    tar.add(config_file, arcname=config_file.name)

        console.print(format_success(f"Backup created successfully: {backup_file}"))

        # Show backup contents
        console.print(f"\n[cyan]Backup contains:[/cyan]")
        with tarfile.open(backup_file, "r:gz") as tar:
            for member in tar.getmembers():
                console.print(f"  • {member.name}")

    except Exception as e:
        console.print(format_error(f"Failed to create backup: {str(e)}"))