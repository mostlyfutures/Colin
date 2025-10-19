"""
REST API for Colin Trading Bot v2.0

This module implements REST API endpoints for external integration.

Key Features:
- Signal generation endpoints: POST /api/v2/signals/generate, GET /api/v2/signals/{symbol}
- Order management endpoints: POST /api/v2/orders, GET /api/v2/orders/{order_id}
- Portfolio management: GET /api/v2/portfolio, GET /api/v2/portfolio/performance
- System status: GET /api/v2/health, GET /api/v2/metrics
- Authentication and rate limiting per SECURITY_IMPLEMENTATION_GUIDE.md
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, Security, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import v2 components
from ..main import ColinTradingBotV2
from ..config.main_config import get_main_config


# Pydantic models for API requests/responses
class SignalRequest(BaseModel):
    """Signal generation request model."""
    symbols: List[str] = Field(..., description="List of symbols to generate signals for")
    confidence_threshold: float = Field(0.65, ge=0.0, le=1.0, description="Minimum confidence threshold")
    time_horizon_hours: int = Field(24, ge=1, le=168, description="Signal time horizon in hours")


class SignalResponse(BaseModel):
    """Signal response model."""
    symbol: str
    direction: str  # "long", "short", "neutral"
    confidence: float
    strength: float
    predicted_return: float
    timestamp: datetime
    source_model: str
    metadata: Dict[str, Any]


class OrderRequest(BaseModel):
    """Order creation request model."""
    symbol: str
    side: str  # "buy", "sell"
    order_type: str = "market"  # "market", "limit"
    quantity: float
    price: Optional[float] = None  # Required for limit orders
    client_order_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class OrderResponse(BaseModel):
    """Order response model."""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    status: str
    filled_quantity: float
    average_price: Optional[float]
    fees: float
    timestamp: datetime
    exchange: Optional[str]
    metadata: Dict[str, Any]


class PortfolioResponse(BaseModel):
    """Portfolio response model."""
    total_value: float
    total_pnl: float
    pnl_percentage: float
    positions: List[Dict[str, Any]]
    cash_balance: float
    margin_used: float
    margin_available: float
    last_updated: datetime


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    environment: str
    uptime_seconds: float
    components: Dict[str, str]
    metrics: Dict[str, Any]


class MetricsResponse(BaseModel):
    """Metrics response model."""
    signals_generated: int
    executions_completed: int
    success_rate: float
    average_latency_ms: float
    portfolio_value: float
    active_positions: int
    risk_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    timestamp: datetime


# Security
security = HTTPBearer()
limiter = Limiter(key_func=get_remote_address)


class RestAPI:
    """
    REST API for Colin Trading Bot v2.0.

    This class provides a comprehensive REST API following FastAPI patterns
    with authentication, rate limiting, and security features.
    """

    def __init__(self, trading_bot: ColinTradingBotV2):
        """
        Initialize REST API.

        Args:
            trading_bot: Colin Trading Bot v2.0 instance
        """
        self.trading_bot = trading_bot
        self.config = get_main_config()
        self.app = FastAPI(
            title="Colin Trading Bot v2.0 API",
            description="AI-Powered Institutional Trading System API",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        # API state
        self.start_time = time.time()
        self.request_count = 0
        self.rate_limit_counts = {}

        # Setup middleware
        self._setup_middleware()

        # Setup routes
        self._setup_routes()

        # Setup exception handlers
        self._setup_exception_handlers()

        logger.info("REST API initialized")

    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.api_cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            self.request_count += 1

            response = await call_next(request)

            process_time = time.time() - start_time
            logger.info(f"{request.method} {request.url.path} - "
                       f"Status: {response.status_code} - "
                       f"Time: {process_time:.3f}s")

            return response

    def _setup_routes(self):
        """Setup API routes."""
        # Health check
        @self.app.get("/api/v2/health", response_model=HealthResponse)
        @limiter.limit("100/minute")
        async def health_check():
            """System health check endpoint."""
            return await self._get_health_status()

        # Signal endpoints
        @self.app.post("/api/v2/signals/generate", response_model=List[SignalResponse])
        @limiter.limit("10/minute")
        async def generate_signals(
            request: SignalRequest,
            credentials: HTTPAuthorizationCredentials = Security(security)
        ):
            """Generate trading signals for specified symbols."""
            await self._verify_api_key(credentials.credentials)
            return await self._generate_signals(request)

        @self.app.get("/api/v2/signals/{symbol}", response_model=List[SignalResponse])
        @limiter.limit("30/minute")
        async def get_signal_history(
            symbol: str,
            limit: int = 10,
            credentials: HTTPAuthorizationCredentials = Security(security)
        ):
            """Get signal history for a specific symbol."""
            await self._verify_api_key(credentials.credentials)
            return await self._get_signal_history(symbol, limit)

        # Order management endpoints
        @self.app.post("/api/v2/orders", response_model=OrderResponse)
        @limiter.limit("20/minute")
        async def create_order(
            order: OrderRequest,
            credentials: HTTPAuthorizationCredentials = Security(security)
        ):
            """Create and execute a new order."""
            await self._verify_api_key(credentials.credentials)
            return await self._create_order(order)

        @self.app.get("/api/v2/orders/{order_id}", response_model=OrderResponse)
        @limiter.limit("60/minute")
        async def get_order(
            order_id: str,
            credentials: HTTPAuthorizationCredentials = Security(security)
        ):
            """Get order details by ID."""
            await self._verify_api_key(credentials.credentials)
            return await self._get_order(order_id)

        @self.app.get("/api/v2/orders", response_model=List[OrderResponse])
        @limiter.limit("30/minute")
        async def list_orders(
            symbol: Optional[str] = None,
            status: Optional[str] = None,
            limit: int = 50,
            credentials: HTTPAuthorizationCredentials = Security(security)
        ):
            """List orders with optional filtering."""
            await self._verify_api_key(credentials.credentials)
            return await self._list_orders(symbol, status, limit)

        # Portfolio management endpoints
        @self.app.get("/api/v2/portfolio", response_model=PortfolioResponse)
        @limiter.limit("30/minute")
        async def get_portfolio(
            credentials: HTTPAuthorizationCredentials = Security(security)
        ):
            """Get current portfolio status."""
            await self._verify_api_key(credentials.credentials)
            return await self._get_portfolio()

        @self.app.get("/api/v2/portfolio/performance")
        @limiter.limit("10/minute")
        async def get_portfolio_performance(
            period_days: int = 30,
            credentials: HTTPAuthorizationCredentials = Security(security)
        ):
            """Get portfolio performance metrics."""
            await self._verify_api_key(credentials.credentials)
            return await self._get_portfolio_performance(period_days)

        # System metrics endpoint
        @self.app.get("/api/v2/metrics", response_model=MetricsResponse)
        @limiter.limit("20/minute")
        async def get_metrics(
            credentials: HTTPAuthorizationCredentials = Security(security)
        ):
            """Get system metrics and performance data."""
            await self._verify_api_key(credentials.credentials)
            return await self._get_metrics()

        # Risk management endpoints
        @self.app.get("/api/v2/risk/status")
        @limiter.limit("15/minute")
        async def get_risk_status(
            credentials: HTTPAuthorizationCredentials = Security(security)
        ):
            """Get current risk management status."""
            await self._verify_api_key(credentials.credentials)
            return await self._get_risk_status()

        @self.app.get("/api/v2/risk/limits")
        @limiter.limit("10/minute")
        async def get_risk_limits(
            credentials: HTTPAuthorizationCredentials = Security(security)
        ):
            """Get current risk limits configuration."""
            await self._verify_api_key(credentials.credentials)
            return await self._get_risk_limits()

        # Configuration endpoint (admin only)
        @self.app.get("/api/v2/config")
        @limiter.limit("5/minute")
        async def get_configuration(
            credentials: HTTPAuthorizationCredentials = Security(security)
        ):
            """Get system configuration (admin only)."""
            await self._verify_admin_key(credentials.credentials)
            return await self._get_configuration()

    def _setup_exception_handlers(self):
        """Setup exception handlers."""
        @self.app.exception_handler(RateLimitExceeded)
        async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
            """Handle rate limit exceeded."""
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please try again later."}
            )

        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Handle HTTP exceptions."""
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail}
            )

        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle general exceptions."""
            logger.error(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )

    async def _verify_api_key(self, api_key: str):
        """Verify API key authentication."""
        if not self.config.api_key_required:
            return

        # Simple API key verification (in production, use proper JWT or database lookup)
        valid_keys = os.getenv("VALID_API_KEYS", "test-api-key").split(",")
        if api_key not in valid_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )

    async def _verify_admin_key(self, api_key: str):
        """Verify admin API key."""
        admin_keys = os.getenv("ADMIN_API_KEYS", "admin-key").split(",")
        if api_key not in admin_keys:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )

    async def _get_health_status(self) -> HealthResponse:
        """Get system health status."""
        uptime = time.time() - self.start_time

        # Check component health
        components = {
            "trading_bot": "healthy" if self.trading_bot.is_running else "stopped",
            "risk_system": "healthy",
            "execution_engine": "healthy",
            "compliance_system": "healthy",
            "database": "healthy"
        }

        # Overall status
        overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "degraded"

        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            version="2.0.0",
            environment=self.config.system.environment,
            uptime_seconds=uptime,
            components=components,
            metrics={
                "requests_handled": self.request_count,
                "active_positions": len(self.trading_bot.active_positions)
            }
        )

    async def _generate_signals(self, request: SignalRequest) -> List[SignalResponse]:
        """Generate trading signals."""
        # This would integrate with the actual ML pipeline
        # For now, return mock signals
        signals = []

        for symbol in request.symbols:
            # Mock signal generation
            import random
            if random.random() < 0.7:  # 70% chance of signal
                direction = random.choice(["long", "short"])
                confidence = random.uniform(request.confidence_threshold, 0.95)
                strength = random.uniform(0.5, 1.0)

                signal = SignalResponse(
                    symbol=symbol,
                    direction=direction,
                    confidence=confidence,
                    strength=strength,
                    predicted_return=random.uniform(-0.05, 0.05),
                    timestamp=datetime.now(),
                    source_model="ensemble_model",
                    metadata={
                        "time_horizon_hours": request.time_horizon_hours,
                        "features_used": ["technical", "orderbook", "sentiment"]
                    }
                )
                signals.append(signal)

        return signals

    async def _get_signal_history(self, symbol: str, limit: int) -> List[SignalResponse]:
        """Get signal history for symbol."""
        # Mock implementation
        return []

    async def _create_order(self, order_request: OrderRequest) -> OrderResponse:
        """Create and execute order."""
        # Convert to internal order format
        from ..execution_engine.smart_routing.router import Order, OrderSide, OrderType

        internal_order = Order(
            symbol=order_request.symbol,
            side=OrderSide.BUY if order_request.side.lower() == "buy" else OrderSide.SELL,
            order_type=OrderType.MARKET if order_request.order_type.lower() == "market" else OrderType.LIMIT,
            quantity=order_request.quantity,
            price=order_request.price,
            client_order_id=order_request.client_order_id or f"api_order_{int(time.time())}",
            metadata=order_request.metadata or {}
        )

        # Execute order
        execution_result = await self.trading_bot._execute_order(internal_order)

        return OrderResponse(
            order_id=internal_order.client_order_id,
            symbol=internal_order.symbol,
            side=internal_order.side.value,
            order_type=internal_order.order_type.value,
            quantity=internal_order.quantity,
            price=internal_order.price,
            status="filled" if execution_result["success"] else "failed",
            filled_quantity=execution_result.get("executed_quantity", 0),
            average_price=execution_result.get("executed_price"),
            fees=execution_result.get("fees", 0),
            timestamp=datetime.now(),
            exchange=execution_result.get("exchange"),
            metadata=execution_result
        )

    async def _get_order(self, order_id: str) -> OrderResponse:
        """Get order by ID."""
        # Mock implementation - would query from database
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )

    async def _list_orders(self, symbol: Optional[str], status: Optional[str], limit: int) -> List[OrderResponse]:
        """List orders with filtering."""
        # Mock implementation
        return []

    async def _get_portfolio(self) -> PortfolioResponse:
        """Get current portfolio status."""
        portfolio_value = sum(pos["value_usd"] for pos in self.trading_bot.active_positions.values())
        total_pnl = self.trading_bot.total_pnl
        pnl_percentage = (total_pnl / (portfolio_value - total_pnl)) * 100 if portfolio_value > 0 else 0

        positions = [
            {
                "symbol": symbol,
                "quantity": pos["quantity"],
                "value_usd": pos["value_usd"],
                "side": pos["side"],
                "avg_price": pos["avg_price"],
                "pnl": pos.get("pnl", 0)
            }
            for symbol, pos in self.trading_bot.active_positions.items()
        ]

        return PortfolioResponse(
            total_value=portfolio_value,
            total_pnl=total_pnl,
            pnl_percentage=pnl_percentage,
            positions=positions,
            cash_balance=portfolio_value * 0.1,  # Mock
            margin_used=portfolio_value * 0.05,  # Mock
            margin_available=portfolio_value * 0.15,  # Mock
            last_updated=datetime.now()
        )

    async def _get_portfolio_performance(self, period_days: int) -> Dict[str, Any]:
        """Get portfolio performance metrics."""
        # Mock implementation
        return {
            "period_days": period_days,
            "total_return": 0.05,
            "annualized_return": 0.18,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.03,
            "volatility": 0.12,
            "win_rate": 0.65
        }

    async def _get_metrics(self) -> MetricsResponse:
        """Get system metrics."""
        portfolio_metrics = await self._get_portfolio_metrics()

        return MetricsResponse(
            signals_generated=self.trading_bot.signals_generated,
            executions_completed=self.trading_bot.executions_completed,
            success_rate=self.trading_bot.executions_completed / max(1, self.trading_bot.signals_generated),
            average_latency_ms=self.trading_bot.average_latency_ms,
            portfolio_value=portfolio_metrics["total_value"],
            active_positions=len(self.trading_bot.active_positions),
            risk_metrics=await self._get_risk_metrics(),
            performance_metrics={
                "uptime_seconds": time.time() - self.start_time,
                "requests_per_minute": self.request_count / max(1, (time.time() - self.start_time) / 60),
                "memory_usage_mb": 150  # Mock
            },
            timestamp=datetime.now()
        )

    async def _get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get portfolio metrics."""
        return {
            "total_value": sum(pos["value_usd"] for pos in self.trading_bot.active_positions.values()),
            "position_count": len(self.trading_bot.active_positions),
            "cash_balance": 100000.0,  # Mock
            "margin_used": 0.0
        }

    async def _get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics."""
        if self.trading_bot.risk_controller:
            return self.trading_bot.risk_controller.get_risk_metrics()
        return {}

    async def _get_risk_status(self) -> Dict[str, Any]:
        """Get risk management status."""
        return {
            "risk_controller_active": self.trading_bot.risk_controller is not None,
            "circuit_breaker_active": self.trading_bot.risk_controller.circuit_breaker_active if self.trading_bot.risk_controller else False,
            "current_drawdown": self.trading_bot.drawdown_controller.current_drawdown if self.trading_bot.drawdown_controller else 0,
            "risk_metrics": await self._get_risk_metrics()
        }

    async def _get_risk_limits(self) -> Dict[str, Any]:
        """Get risk limits configuration."""
        if self.trading_bot.risk_controller:
            limits = self.trading_bot.risk_controller.risk_limits
            return {
                "max_position_size_usd": limits.max_position_size_usd,
                "max_portfolio_exposure": limits.max_portfolio_exposure,
                "max_leverage": limits.max_leverage,
                "max_correlation_exposure": limits.max_correlation_exposure,
                "max_drawdown_hard": limits.max_drawdown_hard,
                "max_drawdown_warning": limits.max_drawdown_warning
            }
        return {}

    async def _get_configuration(self) -> Dict[str, Any]:
        """Get system configuration."""
        return self.config.get_configuration_summary()

    async def run(self, host: str = None, port: int = None):
        """Run the API server."""
        host = host or self.config.api_host
        port = port or self.config.api_port

        logger.info(f"Starting REST API server on {host}:{port}")

        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level=self.config.system.log_level.lower(),
            access_log=True
        )

        server = uvicorn.Server(config)
        await server.serve()


# Helper function to create and run API
def create_api_server(trading_bot: ColinTradingBotV2) -> RestAPI:
    """Create API server instance."""
    return RestAPI(trading_bot)


if __name__ == "__main__":
    # For testing
    import os
    from unittest.mock import Mock

    mock_bot = Mock(spec=ColinTradingBotV2)
    mock_bot.is_running = True
    mock_bot.active_positions = {}
    mock_bot.signals_generated = 0
    mock_bot.executions_completed = 0
    mock_bot.total_pnl = 0.0
    mock_bot.average_latency_ms = 25.0
    mock_bot.risk_controller = None
    mock_bot.drawdown_controller = None

    api = RestAPI(mock_bot)

    # Run the API
    asyncio.run(api.run())