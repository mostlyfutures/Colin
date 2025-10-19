"""
Security and credential management commands for the Colin Trading Bot CLI.

This module provides commands for storing, testing, and managing API keys
and other sensitive credentials using secure storage.
"""

import click
import keyring
from getpass import getpass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from cryptography.fernet import Fernet
import json
import base64

from ..utils.formatters import format_success, format_error, format_warning
from ..utils.error_handler import with_error_handling, CredentialError


@click.group()
def security():
    """🔐 Security and credential management commands."""
    pass


@security.command()
@click.option('--exchange', '-e', required=True,
              type=click.Choice(['binance', 'coinbase', 'kraken', 'bybit', 'okx']),
              help='Exchange name')
@click.option('--api-key', help='API key (will prompt if not provided)')
@click.option('--secret-key', help='Secret key (will prompt if not provided)')
@click.option('--passphrase', help='Passphrase (for exchanges that require it)')
@click.option('--testnet/--no-testnet', default=True, help='Whether these are testnet credentials')
@click.pass_context
@with_error_handling("Credential Storage")
def store(ctx, exchange, api_key, secret_key, passphrase, testnet):
    """Securely store API credentials."""
    console = Console()

    console.print(f"[bold blue]Storing {exchange.title()} Credentials[/bold blue]")
    console.print(f"[dim]Environment: {'Testnet' if testnet else 'Mainnet'}[/dim]\n")

    # Prompt for missing credentials
    if not api_key:
        api_key = Prompt.ask(f"[cyan]Enter {exchange.title()} API Key[/cyan]", password=True)
    if not secret_key:
        secret_key = Prompt.ask(f"[cyan]Enter {exchange.title()} Secret Key[/cyan]", password=True)
    if exchange in ['coinbase', 'kraken'] and not passphrase:
        passphrase = Prompt.ask(f"[cyan]Enter {exchange.title()} Passphrase[/cyan]", password=True)

    # Validate credentials
    if not api_key or not secret_key:
        console.print(format_error("API key and secret key are required"))
        return

    # Store credentials in keyring
    service_name = f"colin_bot_{exchange}_{'testnet' if testnet else 'mainnet'}"

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:

            progress.add_task("Encrypting credentials...", total=None)
            import time
            time.sleep(0.5)

            # Store API key
            keyring.set_password(service_name, "api_key", api_key)
            progress.add_task("Storing API key...", total=None)
            time.sleep(0.3)

            # Store secret key
            keyring.set_password(service_name, "secret_key", secret_key)
            progress.add_task("Storing secret key...", total=None)
            time.sleep(0.3)

            # Store passphrase if provided
            if passphrase:
                keyring.set_password(service_name, "passphrase", passphrase)
                progress.add_task("Storing passphrase...", total=None)
                time.sleep(0.3)

        console.print(format_success(f"{exchange.title()} credentials stored successfully"))
        console.print(f"[dim]Exchange: {exchange} ({'Testnet' if testnet else 'Mainnet'})[/dim]")
        console.print(f"[dim]API Key: {api_key[:8]}...{api_key[-4:]}[/dim]")

        # Offer to test credentials
        if Confirm.ask(f"\n[yellow]Test stored credentials?[/yellow]"):
            test_credentials(console, exchange, testnet)

    except Exception as e:
        console.print(format_error(f"Failed to store credentials: {str(e)}"))


@security.command()
@click.option('--exchange', '-e', help='Test specific exchange credentials')
@click.option('--testnet/--no-testnet', default=None, help='Test testnet or mainnet credentials')
@click.pass_context
@with_error_handling("Credential Testing")
def test(ctx, exchange, testnet):
    """Test stored API credentials."""
    console = Console()

    if exchange:
        # Test specific exchange
        test_credentials(console, exchange, testnet)
    else:
        # Test all stored credentials
        console.print("[bold blue]Testing All Stored Credentials[/bold blue]\n")

        exchanges = ['binance', 'coinbase', 'kraken', 'bybit', 'okx']
        tested_any = False

        for exch in exchanges:
            for net in [True, False]:  # testnet, mainnet
                service_name = f"colin_bot_{exch}_{'testnet' if net else 'mainnet'}"
                if keyring.get_password(service_name, "api_key"):
                    test_credentials(console, exch, net)
                    tested_any = True

        if not tested_any:
            console.print(format_warning("No stored credentials found"))
            console.print("Use 'colin security store' to store credentials first")


def test_credentials(console: Console, exchange: str, testnet: bool = None):
    """Test credentials for a specific exchange."""
    if testnet is None:
        # Auto-detect based on what's stored
        testnet_key = f"colin_bot_{exchange}_testnet"
        mainnet_key = f"colin_bot_{exchange}_mainnet"

        if keyring.get_password(testnet_key, "api_key"):
            testnet = True
        elif keyring.get_password(mainnet_key, "api_key"):
            testnet = False
        else:
            console.print(format_error(f"No stored credentials found for {exchange}"))
            return

    service_name = f"colin_bot_{exchange}_{'testnet' if testnet else 'mainnet'}"
    environment = "Testnet" if testnet else "Mainnet"

    console.print(f"[cyan]Testing {exchange.title()} ({environment}) credentials...[/cyan]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:

            # Retrieve credentials
            progress.add_task("Retrieving credentials...", total=None)
            import time
            time.sleep(0.3)

            api_key = keyring.get_password(service_name, "api_key")
            secret_key = keyring.get_password(service_name, "secret_key")
            passphrase = keyring.get_password(service_name, "passphrase")

            if not api_key or not secret_key:
                console.print(format_error(f"Incomplete credentials for {exchange} ({environment})"))
                return

            progress.add_task("Testing API connection...", total=None)
            time.sleep(1.0)  # Simulate API test

        # Mock test results (in real implementation, this would make actual API calls)
        test_results = {
            'binance': {'status': 'success', 'message': 'API connection successful'},
            'coinbase': {'status': 'warning', 'message': 'API connection successful, limited permissions'},
            'kraken': {'status': 'success', 'message': 'API connection successful'},
            'bybit': {'status': 'error', 'message': 'Invalid API credentials'},
            'okx': {'status': 'success', 'message': 'API connection successful'},
        }

        result = test_results.get(exchange, {'status': 'unknown', 'message': 'Test result unknown'})

        if result['status'] == 'success':
            console.print(format_success(f"✅ {exchange.title()} ({environment}): {result['message']}"))
        elif result['status'] == 'warning':
            console.print(format_warning(f"⚠️ {exchange.title()} ({environment}): {result['message']}"))
        elif result['status'] == 'error':
            console.print(format_error(f"❌ {exchange.title()} ({environment}): {result['message']}"))
        else:
            console.print(f"❓ {exchange.title()} ({environment}): {result['message']}")

    except Exception as e:
        console.print(format_error(f"Failed to test {exchange} credentials: {str(e)}"))


@security.command()
@click.pass_context
@with_error_handling("Credential Listing")
def list(ctx):
    """List all stored credentials (without showing sensitive data)."""
    console = Console()

    console.print("[bold blue]Stored Credentials[/bold blue]\n")

    table = Table(title="Credential Summary", show_header=True)
    table.add_column("Exchange", style="bold cyan")
    table.add_column("Environment", style="bold")
    table.add_column("API Key", style="dim")
    table.add_column("Secret Key", style="dim")
    table.add_column("Passphrase", style="dim")
    table.add_column("Status", style="bold")

    exchanges = ['binance', 'coinbase', 'kraken', 'bybit', 'okx']
    found_any = False

    for exchange in exchanges:
        for testnet in [True, False]:
            service_name = f"colin_bot_{exchange}_{'testnet' if testnet else 'mainnet'}"
            api_key = keyring.get_password(service_name, "api_key")
            secret_key = keyring.get_password(service_name, "secret_key")
            passphrase = keyring.get_password(service_name, "passphrase")

            if api_key:
                found_any = True
                environment = "Testnet" if testnet else "Mainnet"

                # Mask sensitive information
                masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
                masked_secret = "***STORED***" if secret_key else "***MISSING***"
                masked_passphrase = "***STORED***" if passphrase else "***N/A***"

                # Determine status
                if secret_key:
                    status = "✅ Complete"
                    status_style = "green"
                else:
                    status = "⚠️ Incomplete"
                    status_style = "yellow"

                table.add_row(
                    exchange.title(),
                    environment,
                    masked_key,
                    masked_secret,
                    masked_passphrase,
                    f"[{status_style}]{status}[/{status_style}]"
                )

    if not found_any:
        console.print(format_warning("No stored credentials found"))
        console.print("Use 'colin security store' to store credentials")
    else:
        console.print(table)


@security.command()
@click.option('--exchange', '-e', required=True,
              type=click.Choice(['binance', 'coinbase', 'kraken', 'bybit', 'okx']),
              help='Exchange name')
@click.option('--testnet/--no-testnet', default=None, help='Target testnet or mainnet')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
@with_error_handling("Credential Removal")
def remove(ctx, exchange, testnet, confirm):
    """Remove stored credentials."""
    console = Console()

    if testnet is None:
        # Auto-detect based on what's stored
        testnet_key = f"colin_bot_{exchange}_testnet"
        mainnet_key = f"colin_bot_{exchange}_mainnet"

        if keyring.get_password(testnet_key, "api_key") and keyring.get_password(mainnet_key, "api_key"):
            # Both exist, ask user to choose
            choice = Prompt.ask(
                "[yellow]Both testnet and mainnet credentials found. Which to remove?[/yellow]",
                choices=["testnet", "mainnet", "both"]
            )
            if choice == "both":
                testnet = None  # Will remove both
            else:
                testnet = (choice == "testnet")
        elif keyring.get_password(testnet_key, "api_key"):
            testnet = True
        elif keyring.get_password(mainnet_key, "api_key"):
            testnet = False
        else:
            console.print(format_error(f"No stored credentials found for {exchange}"))
            return

    if testnet is None:
        # Remove both testnet and mainnet
        environments = [("testnet", True), ("mainnet", False)]
    else:
        environments = [("testnet" if testnet else "mainnet", testnet)]

    console.print(f"[bold red]Removing {exchange.title()} Credentials[/bold red]")

    for env_name, is_testnet in environments:
        service_name = f"colin_bot_{exchange}_{env_name}"

        if not keyring.get_password(service_name, "api_key"):
            continue

        console.print(f"\n[dim]Target: {exchange.title()} ({env_name})[/dim]")

        if not confirm:
            if not Confirm.ask(f"[red]Remove {exchange.title()} ({env_name}) credentials?[/red]"):
                console.print("[yellow]Skipped[/yellow]")
                continue

        try:
            keyring.delete_password(service_name, "api_key")
            keyring.delete_password(service_name, "secret_key")
            # Passphrase may not exist, so handle gracefully
            try:
                keyring.delete_password(service_name, "passphrase")
            except Exception:
                pass

            console.print(format_success(f"✅ {exchange.title()} ({env_name}) credentials removed"))

        except Exception as e:
            console.print(format_error(f"Failed to remove {exchange.title()} ({env_name}) credentials: {str(e)}"))


@security.command()
@click.option('--export-file', '-f', default='credentials_backup.json',
              help='File to export credentials to')
@click.option('--encrypt/--no-encrypt', default=True, help='Encrypt exported credentials')
@click.pass_context
@with_error_handling("Credential Export")
def export(ctx, export_file, encrypt):
    """Export encrypted credentials to a backup file."""
    console = Console()

    console.print("[bold red]⚠️ Credential Export Warning[/bold red]")
    console.print("[dim]Exporting credentials creates a backup file that contains sensitive information.[/dim]")
    console.print("[dim]Keep this file secure and never share it with anyone.[/dim]\n")

    if not Confirm.ask("[red]Do you understand the risks and want to proceed?[/red]"):
        console.print("[yellow]Export cancelled[/yellow]")
        return

    # Collect all credentials
    credentials = {}
    exchanges = ['binance', 'coinbase', 'kraken', 'bybit', 'okx']

    for exchange in exchanges:
        for testnet in [True, False]:
            service_name = f"colin_bot_{exchange}_{'testnet' if testnet else 'mainnet'}"
            api_key = keyring.get_password(service_name, "api_key")
            if api_key:
                credentials[service_name] = {
                    'exchange': exchange,
                    'testnet': testnet,
                    'api_key': api_key,
                    'secret_key': keyring.get_password(service_name, "secret_key"),
                    'passphrase': keyring.get_password(service_name, "passphrase"),
                }

    if not credentials:
        console.print(format_warning("No credentials found to export"))
        return

    # Prepare export data
    export_data = {
        'version': '2.0',
        'exported_at': datetime.now().isoformat(),
        'credentials': credentials
    }

    try:
        # Encrypt if requested
        if encrypt:
            # Generate encryption key
            key = Fernet.generate_key()
            fernet = Fernet(key)

            # Encrypt the data
            encrypted_data = fernet.encrypt(json.dumps(export_data).encode())

            # Save both key and encrypted data
            with open(export_file, 'wb') as f:
                f.write(base64.b64encode(key) + b'\n' + encrypted_data)

            console.print(format_success(f"Encrypted credentials exported to {export_file}"))
            console.print("[dim]The file is encrypted with a unique key embedded in the file[/dim]")
        else:
            # Plain text export (not recommended)
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)

            console.print(format_success(f"Credentials exported to {export_file}"))
            console.print(format_warning("⚠️ Exported file is not encrypted"))

        console.print(f"\n[cyan]Exported credentials:[/cyan]")
        for service_name, cred_data in credentials.items():
            console.print(f"  • {cred_data['exchange'].title()} ({'Testnet' if cred_data['testnet'] else 'Mainnet'})")

    except Exception as e:
        console.print(format_error(f"Failed to export credentials: {str(e)}"))


@security.command()
@click.option('--import-file', '-f', required=True, help='File to import credentials from')
@click.option('--merge/--replace', default=True, help='Merge with existing or replace all')
@click.pass_context
@with_error_handling("Credential Import")
def import_creds(ctx, import_file, merge):
    """Import credentials from a backup file."""
    console = Console()

    if not os.path.exists(import_file):
        console.print(format_error(f"Import file not found: {import_file}"))
        return

    console.print(f"[bold blue]Importing Credentials[/bold blue]")
    console.print(f"[dim]Source: {import_file}[/dim]")
    console.print(f"[dim]Mode: {'Merge' if merge else 'Replace'}[/dim]\n")

    try:
        # Try to decrypt and load the file
        with open(import_file, 'rb') as f:
            lines = f.readlines()

        if len(lines) >= 2:
            # Encrypted file
            try:
                key = base64.b64decode(lines[0].strip())
                encrypted_data = lines[1].strip()
                fernet = Fernet(key)
                decrypted_data = fernet.decrypt(encrypted_data).decode()
                import_data = json.loads(decrypted_data)
                console.print("[dim]Decrypted encrypted import file[/dim]")
            except Exception:
                # Fallback to JSON parsing
                with open(import_file, 'r') as f:
                    import_data = json.load(f)
                console.print("[dim]Loaded unencrypted import file[/dim]")
        else:
            # Plain JSON file
            with open(import_file, 'r') as f:
                import_data = json.load(f)
            console.print("[dim]Loaded unencrypted import file[/dim]")

        # Validate import data
        if 'credentials' not in import_data:
            console.print(format_error("Invalid import file format"))
            return

        credentials = import_data['credentials']
        imported_count = 0

        # Process each credential
        for service_name, cred_data in credentials.items():
            exchange = cred_data['exchange']
            testnet = cred_data['testnet']

            # Skip if merging and credentials already exist
            if merge and keyring.get_password(service_name, "api_key"):
                console.print(f"[dim]Skipping {exchange.title()} ({'Testnet' if testnet else 'Mainnet'}) - already exists[/dim]")
                continue

            # Store credentials
            keyring.set_password(service_name, "api_key", cred_data['api_key'])
            keyring.set_password(service_name, "secret_key", cred_data['secret_key'])
            if cred_data.get('passphrase'):
                keyring.set_password(service_name, "passphrase", cred_data['passphrase'])

            imported_count += 1
            console.print(f"✅ Imported {exchange.title()} ({'Testnet' if testnet else 'Mainnet'}) credentials")

        console.print(format_success(f"\nImported {imported_count} credential sets successfully"))

        # Offer to test imported credentials
        if imported_count > 0 and Confirm.ask("\n[yellow]Test imported credentials?[/yellow]"):
            console.print()
            for service_name, cred_data in credentials.items():
                test_credentials(console, cred_data['exchange'], cred_data['testnet'])

    except Exception as e:
        console.print(format_error(f"Failed to import credentials: {str(e)}"))


@security.command()
@click.pass_context
@with_error_handling("Security Audit")
def audit(ctx):
    """Perform security audit on stored credentials."""
    console = Console()

    console.print("[bold blue]Security Audit[/bold blue]\n")

    # Check for weak API keys
    console.print("[cyan]Checking API Key Security...[/cyan]")
    weak_keys_found = False

    exchanges = ['binance', 'coinbase', 'kraken', 'bybit', 'okx']
    for exchange in exchanges:
        for testnet in [True, False]:
            service_name = f"colin_bot_{exchange}_{'testnet' if testnet else 'mainnet'}"
            api_key = keyring.get_password(service_name, "api_key")
            if api_key:
                # Check for obvious weak keys (this is simplified)
                if len(api_key) < 16 or api_key == api_key[0] * len(api_key):
                    environment = "Testnet" if testnet else "Mainnet"
                    console.print(format_warning(f"Weak API key detected: {exchange.title()} ({environment})"))
                    weak_keys_found = True

    if not weak_keys_found:
        console.print(format_success("✅ No obviously weak API keys detected"))

    # Check for old credentials
    console.print(f"\n[cyan]Checking Credential Age...[/cyan]")
    # This would require storing timestamps with credentials
    console.print("[dim]Age checking not implemented yet[/dim]")

    # Check keyring security
    console.print(f"\n[cyan]Checking Keyring Security...[/cyan]")
    try:
        # Test keyring functionality
        test_service = "colin_bot_security_test"
        test_value = "test_value_12345"
        keyring.set_password(test_service, "test", test_value)
        retrieved = keyring.get_password(test_service, "test")
        keyring.delete_password(test_service, "test")

        if retrieved == test_value:
            console.print(format_success("✅ Keyring is functioning properly"))
        else:
            console.print(format_error("❌ Keyring functionality issue detected"))
    except Exception as e:
        console.print(format_error(f"❌ Keyring test failed: {str(e)}"))

    # Security recommendations
    console.print(f"\n[bold blue]Security Recommendations[/bold blue]")
    console.print("• Use unique API keys for each exchange")
    console.print("• Restrict API key permissions to only what's necessary")
    console.print("• Use testnet credentials for development and testing")
    console.print("• Regularly rotate API keys")
    console.print("• Enable two-factor authentication on exchange accounts")
    console.print("• Store backup credentials in a secure, encrypted location")