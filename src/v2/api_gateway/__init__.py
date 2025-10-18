"""
API Gateway for Colin Trading Bot v2.0

This module provides REST API and WebSocket endpoints for external integration.

Key Features:
- Signal generation endpoints: POST /api/v2/signals/generate, GET /api/v2/signals/{symbol}
- Order management endpoints: POST /api/v2/orders, GET /api/v2/orders/{order_id}
- Portfolio management: GET /api/v2/portfolio, GET /api/v2/portfolio/performance
- System status: GET /api/v2/health, GET /api/v2/metrics
- Authentication and rate limiting
"""

__version__ = "2.0.0"
__author__ = "Colin Trading Bot Team"

from .rest_api import RestAPI
from .websocket_api import WebSocketAPI

__all__ = [
    "RestAPI",
    "WebSocketAPI"
]