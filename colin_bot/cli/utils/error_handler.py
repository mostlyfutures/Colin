"""
Error handling utilities for the Colin Trading Bot CLI.

This module provides comprehensive error handling with user-friendly
messages and proper logging for debugging.
"""

import sys
import traceback
from typing import Optional, Type
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

# Bot-specific exceptions (these would be imported from the main bot modules)
class BotError(Exception):
    """Base exception for Colin Trading Bot errors."""
    pass

class ConfigurationError(BotError):
    """Raised when there's a configuration problem."""
    pass

class AuthenticationError(BotError):
    """Raised when authentication fails."""
    pass

class APIError(BotError):
    """Raised when API calls fail."""
    pass

class DataError(BotError):
    """Raised when there's a problem with market data."""
    pass

class OrderError(BotError):
    """Raised when order operations fail."""
    pass

class RiskError(BotError):
    """Raised when risk management blocks an operation."""
    pass


class CLIError(Exception):
    """Base exception for CLI-specific errors."""

    def __init__(self, message: str, suggestion: Optional[str] = None):
        super().__init__(message)
        self.suggestion = suggestion


class ConfigurationValidationError(CLIError):
    """Raised when configuration validation fails."""
    pass


class CredentialError(CLIError):
    """Raised when credential operations fail."""
    pass


class NetworkError(CLIError):
    """Raised when network operations fail."""
    pass


class InteractiveSessionError(CLIError):
    """Raised when interactive sessions fail."""
    pass


def format_error_message(error: Exception, console: Console) -> Panel:
    """Format an error message into a rich panel."""

    # Determine error type and styling
    if isinstance(error, KeyboardInterrupt):
        title = Text("Operation Cancelled", style="bold yellow")
        message = "Operation was cancelled by user."
        suggestion = "Use Ctrl+C to cancel operations gracefully."
        style = "yellow"

    elif isinstance(error, (AuthenticationError, CredentialError)):
        title = Text("Authentication Error", style="bold red")
        message = str(error)
        suggestion = "Check your API keys and credentials using 'colin security test'."
        style = "red"

    elif isinstance(error, (ConfigurationError, ConfigurationValidationError)):
        title = Text("Configuration Error", style="bold red")
        message = str(error)
        suggestion = "Run 'colin config validate' to check your configuration."
        style = "red"

    elif isinstance(error, (APIError, NetworkError)):
        title = Text("Network/API Error", style="bold orange1")
        message = str(error)
        suggestion = "Check your internet connection and API status."
        style = "orange1"

    elif isinstance(error, (DataError, OrderError, RiskError)):
        title = Text("Trading System Error", style="bold red")
        message = str(error)
        suggestion = "Check system status and try again later."
        style = "red"

    elif isinstance(error, CLIError):
        title = Text("CLI Error", style="bold red")
        message = str(error)
        suggestion = getattr(error, 'suggestion', "Use 'colin --help' for assistance.")
        style = "red"

    elif isinstance(error, FileNotFoundError):
        title = Text("File Not Found", style="bold red")
        message = str(error)
        suggestion = "Check that the file exists and you have permission to access it."
        style = "red"

    elif isinstance(error, PermissionError):
        title = Text("Permission Denied", style="bold red")
        message = str(error)
        suggestion = "Check file permissions and try running with appropriate privileges."
        style = "red"

    elif isinstance(error, ImportError):
        title = Text("Import Error", style="bold red")
        message = str(error)
        suggestion = "Run 'pip install -r requirements_v2.txt' to install dependencies."
        style = "red"

    else:
        title = Text("Unexpected Error", style="bold red")
        message = str(error)
        suggestion = "Run with --verbose flag for more details or check logs."
        style = "red"

    # Create error panel
    content = Text(message, style="white")

    if suggestion:
        content.append(f"\n\n💡 Suggestion: {suggestion}", style="cyan")

    panel = Panel(
        content,
        title=title,
        border_style=style,
        padding=(1, 2)
    )

    return panel


def create_debug_info(error: Exception) -> Table:
    """Create debug information table for developers."""
    table = Table(title="Debug Information", show_header=True, box=None)
    table.add_column("Property", style="bold cyan")
    table.add_column("Value", style="dim")

    # Basic error info
    table.add_row("Error Type", type(error).__name__)
    table.add_row("Error Message", str(error))

    # Add traceback info if available
    if hasattr(error, '__traceback__') and error.__traceback__:
        tb_info = traceback.format_exception(type(error), error, error.__traceback__)
        tb_text = ''.join(tb_info)
        # Show just the last few lines of traceback
        tb_lines = tb_text.strip().split('\n')
        if len(tb_lines) > 6:
            table.add_row("Traceback", "..." + "\n".join(tb_lines[-4:]))
        else:
            table.add_row("Traceback", tb_text)

    return table


def handle_cli_error(error: Exception, console: Console, verbose: bool = False) -> None:
    """
    Handle CLI errors with appropriate formatting and logging.

    Args:
        error: The exception that occurred
        console: Rich console instance for output
        verbose: Whether to show detailed debug information
    """
    # Don't show tracebacks for Ctrl+C
    if isinstance(error, KeyboardInterrupt):
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        return

    # Display formatted error message
    error_panel = format_error_message(error, console)
    console.print("\n")
    console.print(error_panel)

    # Show debug information in verbose mode
    if verbose:
        console.print("\n")
        debug_table = create_debug_info(error)
        console.print(debug_table)

    # Log the error for debugging (would go to log file in production)
    log_error(error, verbose)


def log_error(error: Exception, verbose: bool = False) -> None:
    """Log error information for debugging purposes."""
    import logging
    from datetime import datetime

    # Configure logging (in production, this would be centralized)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='colin_cli_errors.log'
    )

    logger = logging.getLogger('colin_cli')

    logger.error(f"CLI Error: {type(error).__name__}: {str(error)}")

    if verbose and hasattr(error, '__traceback__') and error.__traceback__:
        logger.debug("Traceback:", exc_info=True)


def safe_execute(func, *args, **kwargs):
    """
    Safely execute a function and handle any errors.

    Returns a tuple of (success: bool, result: Any, error: Optional[Exception])
    """
    try:
        result = func(*args, **kwargs)
        return True, result, None
    except Exception as e:
        return False, None, e


class ErrorContext:
    """Context manager for handling errors in CLI operations."""

    def __init__(self, console: Console, operation_name: str = "operation"):
        self.console = console
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Operation completed successfully
            duration = datetime.now() - self.start_time
            self.console.print(
                f"✅ {self.operation_name} completed successfully "
                f"(took {duration.total_seconds():.2f}s)",
                style="green"
            )
        else:
            # Handle the error
            handle_cli_error(exc_val, self.console)

        # Don't suppress exceptions
        return False


def with_error_handling(operation_name: str = "operation"):
    """Decorator for adding error handling to CLI functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract console from kwargs or create new one
            console = kwargs.get('console', Console())

            with ErrorContext(console, operation_name):
                return func(*args, **kwargs)

        return wrapper
    return decorator


def validate_required_deps():
    """Validate that required dependencies are available."""
    missing_deps = []

    try:
        import click
    except ImportError:
        missing_deps.append("click")

    try:
        import rich
    except ImportError:
        missing_deps.append("rich")

    try:
        import keyring
    except ImportError:
        missing_deps.append("keyring")

    try:
        import cryptography
    except ImportError:
        missing_deps.append("cryptography")

    if missing_deps:
        raise CLIError(
            f"Missing required dependencies: {', '.join(missing_deps)}",
            suggestion="Run 'pip install -r requirements_v2.txt' to install missing packages."
        )


def handle_import_error(module_name: str, error: ImportError) -> CLIError:
    """Handle import errors with helpful suggestions."""

    suggestions = {
        'src': "Make sure you're running from the project root directory.",
        'src.v2': "Check that the v2 system is properly installed and configured.",
        'click': "Run 'pip install click' to install Click.",
        'rich': "Run 'pip install rich' to install Rich.",
        'keyring': "Run 'pip install keyring' to install Keyring.",
        'cryptography': "Run 'pip install cryptography' to install Cryptography.",
    }

    suggestion = suggestions.get(module_name, "Check that all dependencies are installed.")

    return CLIError(
        f"Failed to import module '{module_name}': {str(error)}",
        suggestion=suggestion
    )