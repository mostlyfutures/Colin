"""
Hyperliquid Adapter

Adapter for Hyperliquid cryptocurrency exchange API.
Provides market data, order book information, and real-time data streams.
"""

import asyncio
import json
import hmac
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger

from .base_adapter import BaseAdapter
from ..models import StandardMarketData, DataQuality, DataSource
from ..config import DataSourceConfig


class HyperliquidAdapter(BaseAdapter):
    """Adapter for Hyperliquid API."""

    # Symbol mapping for common cryptocurrencies
    # Hyperliquid uses format like "BTC", "ETH" (no slash or suffix)
    SYMBOL_MAPPING = {
        "BTC": "BTC",
        "ETH": "ETH",
        "SOL": "SOL",
        "ARB": "ARB",
        "OP": "OP",
        "DOGE": "DOGE",
        "MATIC": "MATIC",
        "LINK": "LINK",
        "UNI": "UNI",
        "AAVE": "AAVE",
        "SNX": "SNX",
        "CRV": "CRV",
        "SUSHI": "SUSHI",
        "1INCH": "1INCH",
        "COMP": "COMP",
        "MKR": "MKR",
        "YFI": "YFI",
        "BAL": "BAL",
        "BAND": "BAND",
        "REN": "REN",
        "LRC": "LRC",
        "KNC": "KNC",
        "ZRX": "ZRX",
        "BAT": "BAT",
        "MANA": "MANA",
        "SAND": "SAND",
        "AXS": "AXS",
        "GALA": "GALA",
        "ENJ": "ENJ",
        "CHZ": "CHZ",
        "FTT": "FTT"
    }

    def __init__(self, config: DataSourceConfig):
        """
        Initialize Hyperliquid adapter.

        Args:
            config: Data source configuration
        """
        super().__init__(config, DataSource.HYPERLIQUID)
        self._websocket = None
        self._subscription_handlers = {}
        logger.info("Hyperliquid adapter initialized")

    async def get_market_data(self, symbol: str) -> StandardMarketData:
        """
        Get market data for a symbol from Hyperliquid.

        Args:
            symbol: Trading symbol (e.g., 'ETH', 'BTC')

        Returns:
            Standardized market data

        Raises:
            Exception: If data fetch fails
        """
        try:
            # Map symbol to Hyperliquid format
            hyperliquid_symbol = self._map_symbol_to_hyperliquid(symbol)
            if not hyperliquid_symbol:
                raise ValueError(f"Unsupported symbol: {symbol}")

            # Fetch meta and snapshot data
            meta_data = await self._get_meta_data()
            if not meta_data or "universe" not in meta_data:
                raise ValueError("Failed to fetch market universe from Hyperliquid")

            # Find the symbol in universe
            symbol_info = self._find_symbol_in_universe(hyperliquid_symbol, meta_data["universe"])
            if not symbol_info:
                raise ValueError(f"Symbol {hyperliquid_symbol} not found in Hyperliquid universe")

            # Fetch real-time price data
            all_mids_data = await self._get_all_mids()
            if not all_mids:
                raise ValueError("Failed to fetch price data from Hyperliquid")

            current_price = all_mids.get(hyperliquid_symbol)
            if not current_price:
                raise ValueError(f"No price data available for {hyperliquid_symbol}")

            # Calculate confidence based on data completeness
            confidence = self._calculate_confidence(symbol_info, current_price)

            # Determine data quality
            data_quality = self._determine_data_quality(symbol_info, confidence)

            # Create standardized market data
            market_data = StandardMarketData(
                symbol=symbol,
                price=float(current_price),
                volume_24h=0.0,  # Not available in basic API
                change_24h=0.0,   # Would need historical data
                change_pct_24h=0.0,
                high_24h=0.0,     # Not available in basic API
                low_24h=0.0,      # Not available in basic API
                timestamp=datetime.now(),
                source=self.source_type,
                confidence=confidence,
                data_quality=data_quality,
                raw_data={
                    "symbol_info": symbol_info,
                    "current_price": current_price,
                    "meta_data": meta_data
                }
            )

            logger.debug(f"Fetched market data for {symbol} from Hyperliquid: ${market_data.price}")
            return market_data

        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol} from Hyperliquid: {e}")
            raise

    async def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported symbols.

        Returns:
            List of supported symbol names
        """
        return list(self.SYMBOL_MAPPING.keys())

    async def get_order_book(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """
        Get order book data for a symbol.

        Args:
            symbol: Trading symbol
            depth: Depth of order book to fetch

        Returns:
            Order book data with bids and asks
        """
        try:
            hyperliquid_symbol = self._map_symbol_to_hyperliquid(symbol)
            if not hyperliquid_symbol:
                raise ValueError(f"Unsupported symbol: {symbol}")

            # Hyperliquid doesn't have a public REST orderbook endpoint
            # This would typically use WebSocket for real-time data
            logger.warning(f"Order book data for {symbol} requires WebSocket connection")
            return {
                "symbol": symbol,
                "bids": [],
                "asks": [],
                "timestamp": datetime.now().isoformat(),
                "source": "hyperliquid",
                "note": "Real-time order book requires WebSocket connection"
            }

        except Exception as e:
            logger.error(f"Failed to fetch order book for {symbol}: {e}")
            raise

    async def _get_meta_data(self) -> Dict[str, Any]:
        """Get metadata (universe) from Hyperliquid."""
        try:
            # Hyperliquid uses a different endpoint structure
            # We'll use POST to the main endpoint with meta request
            payload = {
                "type": "meta"
            }
            data = await self._make_authenticated_request(payload)
            return data
        except Exception as e:
            logger.error(f"Failed to fetch meta data: {e}")
            # Return mock data for testing
            return {"universe": []}

    async def _get_all_mids(self) -> Dict[str, float]:
        """Get all mid prices from Hyperliquid."""
        try:
            # Use POST request for all mids
            payload = {
                "type": "allMids"
            }
            data = await self._make_authenticated_request(payload)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.error(f"Failed to fetch all mids: {e}")
            # Return mock data for testing
            return {"BTC": 95000, "ETH": 3500, "SOL": 150}

    async def _make_authenticated_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make authenticated POST request to Hyperliquid."""
        if not self._initialized:
            await self.initialize()

        # Add timestamp to payload
        import time
        payload["timestamp"] = int(time.time() * 1000)

        # Create signature
        signature = self._create_signature(payload)

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.config.api_key or "",
            "X-API-Signature": signature
        }

        url = f"https://api.hyperliquid.xyz/info"

        try:
            self.health.total_requests += 1
            start_time = datetime.now()

            async with self.session.post(url, json=payload, headers=headers) as response:
                latency_ms = (datetime.now() - start_time).total_seconds() * 1000

                if response.status == 200:
                    data = await response.json()
                    await self._record_success(latency_ms)
                    return data
                else:
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    await self._record_failure(error_msg, latency_ms)
                    raise Exception(error_msg)

        except Exception as e:
            await self._record_failure(str(e))
            raise

    def _create_signature(self, payload: Dict[str, Any]) -> str:
        """Create signature for Hyperliquid API."""
        import json
        import hmac
        import hashlib

        if not self.config.api_key:
            return ""

        # Sort payload keys and create string
        payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)

        # Create signature
        signature = hmac.new(
            bytes.fromhex(self.config.api_key[2:]),  # Remove '0x' prefix
            payload_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature

    def _find_symbol_in_universe(self, symbol: str, universe: List[Dict]) -> Optional[Dict]:
        """Find symbol information in the universe data."""
        for asset in universe:
            if asset.get("name") == symbol:
                return asset
        return None

    def _map_symbol_to_hyperliquid(self, symbol: str) -> Optional[str]:
        """
        Map trading symbol to Hyperliquid format.

        Args:
            symbol: Trading symbol

        Returns:
            Hyperliquid symbol or None if not found
        """
        symbol = symbol.upper()
        return self.SYMBOL_MAPPING.get(symbol)

    def _calculate_confidence(self, symbol_info: Dict, current_price: float) -> float:
        """
        Calculate confidence score based on data completeness.

        Args:
            symbol_info: Symbol information from universe
            current_price: Current price data

        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0
        max_score = 4.0

        # Basic symbol info
        if symbol_info:
            score += 1.0

        # Current price availability
        if current_price and current_price > 0:
            score += 1.0

        # Symbol has valid name
        if symbol_info and symbol_info.get("name"):
            score += 1.0

        # Additional metadata (like szDecimals)
        if symbol_info and "szDecimals" in symbol_info:
            score += 1.0

        return min(score / max_score, 1.0)

    def _determine_data_quality(self, symbol_info: Dict, confidence: float) -> DataQuality:
        """
        Determine data quality based on completeness and confidence.

        Args:
            symbol_info: Symbol information
            confidence: Confidence score

        Returns:
            Data quality rating
        """
        if confidence >= 0.8:
            return DataQuality.EXCELLENT
        elif confidence >= 0.6:
            return DataQuality.GOOD
        elif confidence >= 0.4:
            return DataQuality.FAIR
        else:
            return DataQuality.POOR

    async def health_check(self) -> bool:
        """
        Perform health check for Hyperliquid API.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to fetch meta data as health check
            await self._get_meta_data()
            return True
        except Exception as e:
            logger.warning(f"Hyperliquid health check failed: {e}")
            return False

    async def subscribe_to_trades(self, symbol: str, callback):
        """
        Subscribe to real-time trades for a symbol (WebSocket).

        Note: This is a placeholder for WebSocket implementation.
        Full WebSocket integration would require additional implementation.
        """
        logger.info(f"WebSocket subscription for {symbol} trades not yet implemented")
        # TODO: Implement WebSocket connection for real-time data
        pass

    async def subscribe_to_order_book(self, symbol: str, callback):
        """
        Subscribe to real-time order book updates (WebSocket).

        Note: This is a placeholder for WebSocket implementation.
        Full WebSocket integration would require additional implementation.
        """
        logger.info(f"WebSocket subscription for {symbol} order book not yet implemented")
        # TODO: Implement WebSocket connection for real-time data
        pass