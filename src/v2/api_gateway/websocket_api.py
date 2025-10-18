"""
WebSocket API for Colin Trading Bot v2.0

This module implements WebSocket API endpoints for real-time data streaming.

Key Features:
- Real-time signal streaming: ws://localhost:8000/ws/signals
- Live order updates: ws://localhost:8000/ws/orders
- Portfolio updates: ws://localhost:8000/ws/portfolio
- System metrics: ws://localhost:8000/ws/metrics
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from fastapi import WebSocket, WebSocketDisconnect, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger
import websockets
from dataclasses import dataclass, asdict

# Import v2 components
from ..main import ColinTradingBotV2
from ..config.main_config import get_main_config


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    channel: str


class WebSocketConnection:
    """WebSocket connection wrapper."""

    def __init__(self, websocket: WebSocket, client_id: str):
        self.websocket = websocket
        self.client_id = client_id
        self.subscriptions: Set[str] = set()
        self.last_ping = time.time()
        self.is_authenticated = False

    async def send_message(self, message: WebSocketMessage):
        """Send message to WebSocket client."""
        try:
            message_dict = asdict(message)
            message_dict["timestamp"] = message.timestamp.isoformat()
            await self.websocket.send_text(json.dumps(message_dict))
        except Exception as e:
            logger.error(f"Error sending message to client {self.client_id}: {e}")

    async def ping(self):
        """Send ping to client."""
        try:
            await self.websocket.send_text(json.dumps({"type": "ping", "timestamp": datetime.now().isoformat()}))
            self.last_ping = time.time()
        except Exception as e:
            logger.error(f"Error sending ping to client {self.client_id}: {e}")


class WebSocketAPI:
    """
    WebSocket API for Colin Trading Bot v2.0.

    This class provides real-time data streaming through WebSocket connections
    with authentication, subscription management, and connection monitoring.
    """

    def __init__(self, trading_bot: ColinTradingBotV2):
        """
        Initialize WebSocket API.

        Args:
            trading_bot: Colin Trading Bot v2.0 instance
        """
        self.trading_bot = trading_bot
        self.config = get_main_config()

        # Connection management
        self.connections: Dict[str, WebSocketConnection] = {}
        self.channel_subscribers: Dict[str, Set[str]] = {
            "signals": set(),
            "orders": set(),
            "portfolio": set(),
            "metrics": set(),
            "risk": set()
        }

        # Background tasks
        self.broadcast_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False

        # Security
        self.security = HTTPBearer()

        # Message queues
        self.message_queues: Dict[str, asyncio.Queue] = {
            "signals": asyncio.Queue(),
            "orders": asyncio.Queue(),
            "portfolio": asyncio.Queue(),
            "metrics": asyncio.Queue(),
            "risk": asyncio.Queue()
        }

        logger.info("WebSocket API initialized")

    async def handle_websocket_connection(
        self,
        websocket: WebSocket,
        client_id: str,
        token: Optional[str] = None
    ):
        """
        Handle WebSocket connection and lifecycle.

        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier
            token: Authentication token
        """
        await websocket.accept()

        # Create connection wrapper
        connection = WebSocketConnection(websocket, client_id)
        self.connections[client_id] = connection

        try:
            # Authenticate connection
            if token:
                await self._authenticate_connection(connection, token)
            else:
                # Send authentication request
                await connection.send_message(WebSocketMessage(
                    type="auth_required",
                    data={"message": "Authentication required"},
                    timestamp=datetime.now(),
                    channel="system"
                ))

            # Send welcome message
            await connection.send_message(WebSocketMessage(
                type="connected",
                data={
                    "client_id": client_id,
                    "server_time": datetime.now().isoformat(),
                    "available_channels": list(self.channel_subscribers.keys())
                },
                timestamp=datetime.now(),
                channel="system"
            ))

            # Start background tasks if not already running
            if not self.is_running:
                await self._start_background_tasks()

            # Handle messages
            await self._handle_client_messages(connection)

        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling WebSocket connection for {client_id}: {e}")
        finally:
            # Cleanup connection
            await self._cleanup_connection(client_id)

    async def _authenticate_connection(self, connection: WebSocketConnection, token: str):
        """Authenticate WebSocket connection."""
        if not self.config.api_key_required:
            connection.is_authenticated = True
            return

        # Simple token validation (in production, use proper JWT)
        valid_tokens = os.getenv("VALID_API_KEYS", "test-api-key").split(",")
        if token in valid_tokens:
            connection.is_authenticated = True
            await connection.send_message(WebSocketMessage(
                type="auth_success",
                data={"message": "Authentication successful"},
                timestamp=datetime.now(),
                channel="system"
            ))
        else:
            await connection.send_message(WebSocketMessage(
                type="auth_error",
                data={"message": "Invalid authentication token"},
                timestamp=datetime.now(),
                channel="system"
            ))

    async def _handle_client_messages(self, connection: WebSocketConnection):
        """Handle incoming messages from client."""
        try:
            while True:
                # Receive message with timeout
                data = await asyncio.wait_for(
                    connection.websocket.receive_text(),
                    timeout=30.0  # 30 second timeout
                )

                try:
                    message = json.loads(data)
                    await self._process_client_message(connection, message)
                except json.JSONDecodeError:
                    await connection.send_message(WebSocketMessage(
                        type="error",
                        data={"message": "Invalid JSON format"},
                        timestamp=datetime.now(),
                        channel="system"
                    ))

        except asyncio.TimeoutError:
            logger.warning(f"Client {connection.client_id} timed out")
            await connection.send_message(WebSocketMessage(
                type="timeout",
                data={"message": "Connection timed out"},
                timestamp=datetime.now(),
                channel="system"
            ))
        except WebSocketDisconnect:
            raise
        except Exception as e:
            logger.error(f"Error handling message from {connection.client_id}: {e}")

    async def _process_client_message(self, connection: WebSocketConnection, message: Dict[str, Any]):
        """Process incoming client message."""
        message_type = message.get("type")

        if message_type == "subscribe":
            await self._handle_subscription(connection, message.get("channels", []))
        elif message_type == "unsubscribe":
            await self._handle_unsubscription(connection, message.get("channels", []))
        elif message_type == "ping":
            await connection.send_message(WebSocketMessage(
                type="pong",
                data={"timestamp": datetime.now().isoformat()},
                timestamp=datetime.now(),
                channel="system"
            ))
        elif message_type == "get_status":
            await self._send_status_update(connection)
        else:
            await connection.send_message(WebSocketMessage(
                type="error",
                data={"message": f"Unknown message type: {message_type}"},
                timestamp=datetime.now(),
                channel="system"
            ))

    async def _handle_subscription(self, connection: WebSocketConnection, channels: List[str]):
        """Handle channel subscription request."""
        if not connection.is_authenticated:
            await connection.send_message(WebSocketMessage(
                type="error",
                data={"message": "Authentication required for subscriptions"},
                timestamp=datetime.now(),
                channel="system"
            ))
            return

        for channel in channels:
            if channel in self.channel_subscribers:
                connection.subscriptions.add(channel)
                self.channel_subscribers[channel].add(connection.client_id)

                # Send current data for subscribed channels
                await self._send_current_channel_data(connection, channel)

        await connection.send_message(WebSocketMessage(
            type="subscription_success",
            data={"channels": list(connection.subscriptions)},
            timestamp=datetime.now(),
            channel="system"
        ))

    async def _handle_unsubscription(self, connection: WebSocketConnection, channels: List[str]):
        """Handle channel unsubscription request."""
        for channel in channels:
            if channel in connection.subscriptions:
                connection.subscriptions.remove(channel)
                self.channel_subscribers[channel].discard(connection.client_id)

        await connection.send_message(WebSocketMessage(
            type="unsubscription_success",
            data={"channels": channels},
            timestamp=datetime.now(),
            channel="system"
        ))

    async def _send_current_channel_data(self, connection: WebSocketConnection, channel: str):
        """Send current data for a channel when client subscribes."""
        try:
            if channel == "portfolio":
                data = await self._get_portfolio_data()
            elif channel == "metrics":
                data = await self._get_metrics_data()
            elif channel == "risk":
                data = await self._get_risk_data()
            elif channel == "signals":
                data = {"recent_signals": []}  # Would get recent signals
            elif channel == "orders":
                data = {"recent_orders": []}  # Would get recent orders
            else:
                return

            await connection.send_message(WebSocketMessage(
                type="data",
                data=data,
                timestamp=datetime.now(),
                channel=channel
            ))

        except Exception as e:
            logger.error(f"Error sending current data for channel {channel}: {e}")

    async def _send_status_update(self, connection: WebSocketConnection):
        """Send status update to client."""
        status = {
            "client_id": connection.client_id,
            "subscriptions": list(connection.subscriptions),
            "server_uptime": time.time() - (self.trading_bot.start_time if hasattr(self.trading_bot, 'start_time') else time.time()),
            "active_connections": len(self.connections),
            "authenticated": connection.is_authenticated
        }

        await connection.send_message(WebSocketMessage(
            type="status",
            data=status,
            timestamp=datetime.now(),
            channel="system"
        ))

    async def _start_background_tasks(self):
        """Start background broadcasting tasks."""
        if self.is_running:
            return

        self.is_running = True

        # Start broadcasting tasks for each channel
        for channel in self.channel_subscribers.keys():
            task = asyncio.create_task(self._broadcast_channel_updates(channel))
            self.broadcast_tasks[channel] = task

        # Start connection monitoring task
        asyncio.create_task(self._monitor_connections())

        logger.info("WebSocket background tasks started")

    async def _broadcast_channel_updates(self, channel: str):
        """Broadcast updates for a specific channel."""
        while self.is_running:
            try:
                # Check if there are subscribers
                if not self.channel_subscribers[channel]:
                    await asyncio.sleep(1)
                    continue

                # Get channel data
                data = await self._get_channel_data(channel)

                if data:
                    # Create message
                    message = WebSocketMessage(
                        type="data",
                        data=data,
                        timestamp=datetime.now(),
                        channel=channel
                    )

                    # Broadcast to all subscribers
                    await self._broadcast_to_channel(channel, message)

                # Sleep based on channel update frequency
                if channel == "metrics":
                    await asyncio.sleep(5)  # Metrics every 5 seconds
                elif channel == "portfolio":
                    await asyncio.sleep(10)  # Portfolio every 10 seconds
                elif channel == "risk":
                    await asyncio.sleep(15)  # Risk every 15 seconds
                else:
                    await asyncio.sleep(2)  # Other channels every 2 seconds

            except Exception as e:
                logger.error(f"Error broadcasting updates for channel {channel}: {e}")
                await asyncio.sleep(5)

    async def _get_channel_data(self, channel: str) -> Optional[Dict[str, Any]]:
        """Get data for a specific channel."""
        try:
            if channel == "portfolio":
                return await self._get_portfolio_data()
            elif channel == "metrics":
                return await self._get_metrics_data()
            elif channel == "risk":
                return await self._get_risk_data()
            elif channel == "signals":
                return await self._get_signals_data()
            elif channel == "orders":
                return await self._get_orders_data()
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting data for channel {channel}: {e}")
            return None

    async def _get_portfolio_data(self) -> Dict[str, Any]:
        """Get portfolio data for broadcasting."""
        portfolio_value = sum(pos["value_usd"] for pos in self.trading_bot.active_positions.values())
        total_pnl = self.trading_bot.total_pnl

        return {
            "total_value": portfolio_value,
            "total_pnl": total_pnl,
            "pnl_percentage": (total_pnl / (portfolio_value - total_pnl)) * 100 if portfolio_value > 0 else 0,
            "position_count": len(self.trading_bot.active_positions),
            "positions": [
                {
                    "symbol": symbol,
                    "value_usd": pos["value_usd"],
                    "quantity": pos["quantity"],
                    "side": pos["side"]
                }
                for symbol, pos in self.trading_bot.active_positions.items()
            ],
            "last_updated": datetime.now().isoformat()
        }

    async def _get_metrics_data(self) -> Dict[str, Any]:
        """Get metrics data for broadcasting."""
        return {
            "signals_generated": self.trading_bot.signals_generated,
            "executions_completed": self.trading_bot.executions_completed,
            "success_rate": self.trading_bot.executions_completed / max(1, self.trading_bot.signals_generated),
            "average_latency_ms": self.trading_bot.average_latency_ms,
            "active_connections": len(self.connections),
            "uptime_seconds": time.time() - (self.trading_bot.start_time if hasattr(self.trading_bot, 'start_time') else time.time()),
            "last_updated": datetime.now().isoformat()
        }

    async def _get_risk_data(self) -> Dict[str, Any]:
        """Get risk data for broadcasting."""
        risk_data = {
            "circuit_breaker_active": False,
            "current_drawdown": 0.0,
            "risk_score": 0.0
        }

        if self.trading_bot.risk_controller:
            risk_metrics = self.trading_bot.risk_controller.get_risk_metrics()
            risk_data.update({
                "circuit_breaker_active": risk_metrics.get("circuit_breaker_active", False),
                "current_drawdown": risk_metrics.get("current_drawdown", 0.0),
                "risk_score": risk_metrics.get("rejection_rate", 0.0) * 100
            })

        if self.trading_bot.drawdown_controller:
            risk_data["current_drawdown"] = self.trading_bot.drawdown_controller.current_drawdown

        risk_data["last_updated"] = datetime.now().isoformat()
        return risk_data

    async def _get_signals_data(self) -> Dict[str, Any]:
        """Get recent signals data for broadcasting."""
        # Mock implementation - would get from actual signal generator
        return {
            "recent_signals": [],
            "signal_accuracy": 0.68,  # Mock
            "last_signal_time": datetime.now().isoformat()
        }

    async def _get_orders_data(self) -> Dict[str, Any]:
        """Get recent orders data for broadcasting."""
        # Mock implementation - would get from actual order manager
        return {
            "recent_orders": [],
            "execution_rate": 0.95,  # Mock
            "average_fill_time_ms": 45.0,  # Mock
            "last_order_time": datetime.now().isoformat()
        }

    async def _broadcast_to_channel(self, channel: str, message: WebSocketMessage):
        """Broadcast message to all subscribers of a channel."""
        if not self.channel_subscribers[channel]:
            return

        # Get copy of subscribers to avoid modification during iteration
        subscribers = self.channel_subscribers[channel].copy()
        disconnected_clients = []

        for client_id in subscribers:
            if client_id in self.connections:
                connection = self.connections[client_id]
                try:
                    await connection.send_message(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to client {client_id}: {e}")
                    disconnected_clients.append(client_id)
            else:
                disconnected_clients.append(client_id)

        # Remove disconnected clients
        for client_id in disconnected_clients:
            self.channel_subscribers[channel].discard(client_id)

    async def _monitor_connections(self):
        """Monitor WebSocket connections and cleanup inactive ones."""
        while self.is_running:
            try:
                current_time = time.time()
                inactive_clients = []

                for client_id, connection in self.connections.items():
                    # Check if connection is inactive (no ping for 60 seconds)
                    if current_time - connection.last_ping > 60:
                        inactive_clients.append(client_id)
                        try:
                            await connection.websocket.close()
                        except:
                            pass

                # Cleanup inactive connections
                for client_id in inactive_clients:
                    await self._cleanup_connection(client_id)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in connection monitoring: {e}")
                await asyncio.sleep(30)

    async def _cleanup_connection(self, client_id: str):
        """Clean up WebSocket connection."""
        if client_id in self.connections:
            connection = self.connections[client_id]

            # Remove from all channel subscriptions
            for channel in connection.subscriptions:
                self.channel_subscribers[channel].discard(client_id)

            # Remove connection
            del self.connections[client_id]

            logger.debug(f"Cleaned up connection for client {client_id}")

    async def broadcast_signal(self, signal_data: Dict[str, Any]):
        """Broadcast trading signal to all subscribers."""
        message = WebSocketMessage(
            type="signal",
            data=signal_data,
            timestamp=datetime.now(),
            channel="signals"
        )
        await self._broadcast_to_channel("signals", message)

    async def broadcast_order_update(self, order_data: Dict[str, Any]):
        """Broadcast order update to all subscribers."""
        message = WebSocketMessage(
            type="order_update",
            data=order_data,
            timestamp=datetime.now(),
            channel="orders"
        )
        await self._broadcast_to_channel("orders", message)

    async def broadcast_alert(self, alert_data: Dict[str, Any]):
        """Broadcast alert to all subscribers."""
        message = WebSocketMessage(
            type="alert",
            data=alert_data,
            timestamp=datetime.now(),
            channel="system"
        )

        # Send to all connected clients (alerts are system-wide)
        for client_id, connection in self.connections.items():
            if connection.is_authenticated:
                try:
                    await connection.send_message(message)
                except Exception as e:
                    logger.error(f"Error sending alert to client {client_id}: {e}")

    async def shutdown(self):
        """Shutdown WebSocket API."""
        logger.info("Shutting down WebSocket API...")

        self.is_running = False

        # Cancel background tasks
        for task in self.broadcast_tasks.values():
            task.cancel()

        # Close all connections
        for client_id, connection in self.connections.items():
            try:
                await connection.websocket.close()
            except:
                pass

        self.connections.clear()
        self.channel_subscribers.clear()

        logger.info("WebSocket API shutdown completed")

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            "total_connections": len(self.connections),
            "authenticated_connections": sum(1 for c in self.connections.values() if c.is_authenticated),
            "channel_subscribers": {
                channel: len(subscribers) for channel, subscribers in self.channel_subscribers.items()
            },
            "is_running": self.is_running
        }


# Helper function to create WebSocket API
def create_websocket_api(trading_bot: ColinTradingBotV2) -> WebSocketAPI:
    """Create WebSocket API instance."""
    return WebSocketAPI(trading_bot)


if __name__ == "__main__":
    # For testing
    import os
    from unittest.mock import Mock

    mock_bot = Mock(spec=ColinTradingBotV2)
    mock_bot.active_positions = {}
    mock_bot.signals_generated = 0
    mock_bot.executions_completed = 0
    mock_bot.total_pnl = 0.0
    mock_bot.average_latency_ms = 25.0

    ws_api = WebSocketAPI(mock_bot)
    print(f"WebSocket API created: {ws_api.get_connection_stats()}")