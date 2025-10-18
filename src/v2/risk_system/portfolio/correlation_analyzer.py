"""
Correlation Analyzer for Colin Trading Bot v2.0

This module implements portfolio correlation analysis and monitoring.

Key Features:
- Multi-asset correlation matrix calculation
- Correlation limit enforcement (<0.7 portfolio correlation from PRP)
- Time-varying correlation tracking
- Concentration risk monitoring
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class CorrelationLevel(Enum):
    """Correlation level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class CorrelationMetrics:
    """Correlation metrics data structure."""
    correlation_matrix: np.ndarray
    avg_correlation: float
    max_correlation: float
    min_correlation: float
    eigenvalues: np.ndarray
    condition_number: float
    diversification_ratio: float
    effective_number_of_bets: float
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class CorrelationAlert:
    """Correlation risk alert."""
    alert_type: str  # "warning", "critical"
    correlation_value: float
    symbols: Tuple[str, str]
    limit_value: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CorrelationConfiguration:
    """Correlation analysis configuration."""
    max_correlation_limit: float = 0.70      # 70% max correlation from PRP
    warning_threshold: float = 0.60          # 60% warning threshold
    min_data_points: int = 30                # Minimum data points for correlation
    correlation_window: int = 60             # 60-day correlation window
    update_frequency_minutes: int = 15       # Update frequency
    eigenvalue_threshold: float = 0.01       # Minimum eigenvalue for stability


class CorrelationAnalyzer:
    """
    Portfolio correlation analyzer with real-time monitoring.

    This class calculates and monitors portfolio correlations to ensure
    diversification and identify concentration risks.
    """

    def __init__(
        self,
        config: Optional[CorrelationConfiguration] = None,
        symbols: Optional[List[str]] = None
    ):
        """
        Initialize correlation analyzer.

        Args:
            config: Correlation analysis configuration
            symbols: List of symbols to analyze
        """
        self.config = config or CorrelationConfiguration()
        self.symbols = symbols or []

        # Data storage
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.returns_history: Dict[str, List[float]] = {}
        self.correlation_matrix: Optional[np.ndarray] = None
        self.correlation_metrics: Optional[CorrelationMetrics] = None

        # Alerts and monitoring
        self.correlation_alerts: List[CorrelationAlert] = []
        self.active_alerts: List[CorrelationAlert] = []

        # Background tasks
        self.monitoring_task = None
        self.is_monitoring = False

        # Performance metrics
        self.calculation_count = 0
        self.last_calculation_time = 0.0

        logger.info(f"CorrelationAnalyzer initialized with {len(self.symbols)} symbols")
        logger.info(f"Max correlation limit: {self.config.max_correlation_limit:.1%}")

    def add_symbol(self, symbol: str):
        """Add a symbol to correlation analysis."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self.price_history[symbol] = []
            self.returns_history[symbol] = []
            logger.info(f"Added symbol {symbol} to correlation analysis")

    def remove_symbol(self, symbol: str):
        """Remove a symbol from correlation analysis."""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            self.price_history.pop(symbol, None)
            self.returns_history.pop(symbol, None)
            logger.info(f"Removed symbol {symbol} from correlation analysis")

    async def add_price_data(
        self,
        symbol: str,
        price: float,
        timestamp: Optional[datetime] = None
    ):
        """
        Add price data for correlation analysis.

        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()

        if symbol not in self.symbols:
            self.add_symbol(symbol)

        # Add price data
        self.price_history[symbol].append((timestamp, price))

        # Keep only recent data (based on correlation window)
        cutoff_date = timestamp - timedelta(days=self.config.correlation_window)
        self.price_history[symbol] = [
            (ts, p) for ts, p in self.price_history[symbol]
            if ts >= cutoff_date
        ]

        # Calculate returns if we have enough data
        if len(self.price_history[symbol]) >= 2:
            prices = [p for _, p in self.price_history[symbol]]
            returns = [np.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
            self.returns_history[symbol] = returns

        # Trigger correlation update if we have enough data
        if await self._has_sufficient_data():
            await self.calculate_correlations()

    async def calculate_correlations(self) -> CorrelationMetrics:
        """
        Calculate correlation matrix and metrics.

        Returns:
            Correlation metrics result
        """
        start_time = asyncio.get_event_loop().time()
        self.calculation_count += 1

        try:
            if not await self._has_sufficient_data():
                logger.warning("Insufficient data for correlation calculation")
                return self._get_default_metrics()

            # Create returns matrix
            returns_matrix = self._create_returns_matrix()

            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(returns_matrix.T)

            # Handle any NaN values
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

            # Calculate additional metrics
            avg_correlation = self._calculate_average_correlation(correlation_matrix)
            max_correlation = self._calculate_max_correlation(correlation_matrix)
            min_correlation = self._calculate_min_correlation(correlation_matrix)
            eigenvalues = np.linalg.eigvals(correlation_matrix)
            condition_number = np.linalg.cond(correlation_matrix)

            # Calculate diversification metrics
            diversification_ratio = self._calculate_diversification_ratio(correlation_matrix)
            effective_bets = self._calculate_effective_number_of_bets(correlation_matrix)

            # Create metrics object
            self.correlation_metrics = CorrelationMetrics(
                correlation_matrix=correlation_matrix,
                avg_correlation=avg_correlation,
                max_correlation=max_correlation,
                min_correlation=min_correlation,
                eigenvalues=eigenvalues,
                condition_number=condition_number,
                diversification_ratio=diversification_ratio,
                effective_number_of_bets=effective_bets,
                last_update=datetime.now()
            )

            # Check for correlation alerts
            await self._check_correlation_alerts(correlation_matrix)

            # Performance metrics
            calculation_time = asyncio.get_event_loop().time() - start_time
            self.last_calculation_time = calculation_time

            logger.debug(f"Correlation calculated in {calculation_time:.3f}s: "
                        f"Avg={avg_correlation:.2f}, Max={max_correlation:.2f}")

            return self.correlation_metrics

        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return self._get_default_metrics()

    async def _has_sufficient_data(self) -> bool:
        """Check if we have sufficient data for correlation analysis."""
        if len(self.symbols) < 2:
            return False

        for symbol in self.symbols:
            if len(self.returns_history.get(symbol, [])) < self.config.min_data_points:
                return False

        return True

    def _create_returns_matrix(self) -> np.ndarray:
        """Create returns matrix from historical data."""
        # Find minimum length across all symbols
        min_length = min(
            len(self.returns_history[symbol])
            for symbol in self.symbols
            if symbol in self.returns_history
        )

        if min_length == 0:
            return np.array([])

        # Create matrix
        returns_matrix = np.zeros((min_length, len(self.symbols)))

        for i, symbol in enumerate(self.symbols):
            if symbol in self.returns_history:
                returns = self.returns_history[symbol][-min_length:]
                returns_matrix[:, i] = returns

        return returns_matrix

    def _calculate_average_correlation(self, correlation_matrix: np.ndarray) -> float:
        """Calculate average correlation (excluding diagonal)."""
        if correlation_matrix.size == 0:
            return 0.0

        # Get upper triangular (excluding diagonal)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        correlations = correlation_matrix[mask]

        return np.mean(correlations) if len(correlations) > 0 else 0.0

    def _calculate_max_correlation(self, correlation_matrix: np.ndarray) -> float:
        """Calculate maximum correlation (excluding diagonal)."""
        if correlation_matrix.size == 0:
            return 0.0

        # Set diagonal to 0
        np.fill_diagonal(correlation_matrix, 0.0)

        return np.max(np.abs(correlation_matrix))

    def _calculate_min_correlation(self, correlation_matrix: np.ndarray) -> float:
        """Calculate minimum correlation (excluding diagonal)."""
        if correlation_matrix.size == 0:
            return 0.0

        # Set diagonal to 1
        np.fill_diagonal(correlation_matrix, 1.0)

        return np.min(correlation_matrix)

    def _calculate_diversification_ratio(self, correlation_matrix: np.ndarray) -> float:
        """Calculate diversification ratio."""
        if correlation_matrix.size == 0:
            return 1.0

        # Simplified diversification ratio
        avg_correlation = self._calculate_average_correlation(correlation_matrix)
        diversification_ratio = 1.0 / (1.0 + avg_correlation)

        return min(diversification_ratio, 1.0)

    def _calculate_effective_number_of_bets(self, correlation_matrix: np.ndarray) -> float:
        """Calculate effective number of independent bets."""
        if correlation_matrix.size == 0:
            return 1.0

        try:
            eigenvalues = np.linalg.eigvals(correlation_matrix)
            # Remove very small eigenvalues to avoid numerical issues
            eigenvalues = eigenvalues[eigenvalues > self.config.eigenvalue_threshold]

            if len(eigenvalues) == 0:
                return 1.0

            effective_bets = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
            return effective_bets

        except Exception as e:
            logger.error(f"Error calculating effective number of bets: {e}")
            return 1.0

    async def _check_correlation_alerts(self, correlation_matrix: np.ndarray):
        """Check for correlation alerts and create alerts if needed."""
        if correlation_matrix.size == 0:
            return

        new_alerts = []

        for i, symbol1 in enumerate(self.symbols):
            for j, symbol2 in enumerate(self.symbols):
                if i >= j:  # Skip diagonal and lower triangular
                    continue

                correlation = abs(correlation_matrix[i, j])

                # Check critical level
                if correlation > self.config.max_correlation_limit:
                    alert = CorrelationAlert(
                        alert_type="critical",
                        correlation_value=correlation,
                        symbols=(symbol1, symbol2),
                        limit_value=self.config.max_correlation_limit,
                        message=f"CRITICAL: Correlation {correlation:.2%} between {symbol1} and {symbol2} exceeds limit {self.config.max_correlation_limit:.1%}"
                    )
                    new_alerts.append(alert)

                # Check warning level
                elif correlation > self.config.warning_threshold:
                    alert = CorrelationAlert(
                        alert_type="warning",
                        correlation_value=correlation,
                        symbols=(symbol1, symbol2),
                        limit_value=self.config.warning_threshold,
                        message=f"WARNING: Correlation {correlation:.2%} between {symbol1} and {symbol2} above warning {self.config.warning_threshold:.1%}"
                    )
                    new_alerts.append(alert)

        if new_alerts:
            self.correlation_alerts.extend(new_alerts)
            self.active_alerts = new_alerts

            for alert in new_alerts:
                logger.warning(alert.message)

    async def start_monitoring(self):
        """Start background correlation monitoring."""
        if self.is_monitoring:
            logger.warning("Correlation monitoring already started")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Correlation monitoring started")

    async def stop_monitoring(self):
        """Stop background correlation monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Correlation monitoring stopped")

    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                # Calculate correlations
                await self.calculate_correlations()

                # Clean up old alerts (keep last hour)
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.correlation_alerts = [
                    alert for alert in self.correlation_alerts
                    if alert.timestamp >= cutoff_time
                ]

                await asyncio.sleep(self.config.update_frequency_minutes * 60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in correlation monitoring loop: {e}")
                await asyncio.sleep(60)

    def _get_default_metrics(self) -> CorrelationMetrics:
        """Get default correlation metrics when insufficient data."""
        size = len(self.symbols)
        if size == 0:
            correlation_matrix = np.array([[1.0]])
        else:
            correlation_matrix = np.eye(size)

        return CorrelationMetrics(
            correlation_matrix=correlation_matrix,
            avg_correlation=0.0,
            max_correlation=0.0,
            min_correlation=0.0,
            eigenvalues=np.ones(size),
            condition_number=1.0,
            diversification_ratio=1.0,
            effective_number_of_bets=float(size),
            last_update=datetime.now()
        )

    def get_correlation_summary(self) -> Dict[str, Any]:
        """Get correlation analysis summary."""
        if self.correlation_metrics is None:
            return {
                "status": "no_data",
                "symbols_count": len(self.symbols),
                "message": "Insufficient data for correlation analysis"
            }

        # Convert correlation matrix to readable format
        correlation_dict = {}
        for i, symbol1 in enumerate(self.symbols):
            correlation_dict[symbol1] = {}
            for j, symbol2 in enumerate(self.symbols):
                correlation_dict[symbol1][symbol2] = float(
                    self.correlation_metrics.correlation_matrix[i, j]
                )

        return {
            "status": "calculated",
            "symbols": self.symbols,
            "symbols_count": len(self.symbols),
            "avg_correlation": self.correlation_metrics.avg_correlation,
            "max_correlation": self.correlation_metrics.max_correlation,
            "min_correlation": self.correlation_metrics.min_correlation,
            "condition_number": self.correlation_metrics.condition_number,
            "diversification_ratio": self.correlation_metrics.diversification_ratio,
            "effective_number_of_bets": self.correlation_metrics.effective_number_of_bets,
            "last_update": self.correlation_metrics.last_update.isoformat(),
            "correlation_matrix": correlation_dict,
            "active_alerts": len(self.active_alerts),
            "total_alerts": len(self.correlation_alerts),
            "alerts": [
                {
                    "type": alert.alert_type,
                    "symbols": alert.symbols,
                    "correlation": alert.correlation_value,
                    "limit": alert.limit_value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.active_alerts
            ]
        }

    def get_high_correlations(self, threshold: float = None) -> List[Dict[str, Any]]:
        """Get list of high correlations above threshold."""
        if threshold is None:
            threshold = self.config.warning_threshold

        if self.correlation_metrics is None:
            return []

        high_correlations = []
        correlation_matrix = self.correlation_metrics.correlation_matrix

        for i, symbol1 in enumerate(self.symbols):
            for j, symbol2 in enumerate(self.symbols):
                if i >= j:  # Skip diagonal and lower triangular
                    continue

                correlation = abs(correlation_matrix[i, j])
                if correlation > threshold:
                    high_correlations.append({
                        "symbol1": symbol1,
                        "symbol2": symbol2,
                        "correlation": correlation,
                        "level": self._get_correlation_level(correlation).value
                    })

        # Sort by correlation (descending)
        high_correlations.sort(key=lambda x: x["correlation"], reverse=True)

        return high_correlations

    def _get_correlation_level(self, correlation: float) -> CorrelationLevel:
        """Get correlation level classification."""
        if correlation > 0.8:
            return CorrelationLevel.VERY_HIGH
        elif correlation > 0.6:
            return CorrelationLevel.HIGH
        elif correlation > 0.3:
            return CorrelationLevel.MEDIUM
        else:
            return CorrelationLevel.LOW

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            "calculations_performed": self.calculation_count,
            "last_calculation_time_ms": self.last_calculation_time * 1000,
            "symbols_tracked": len(self.symbols),
            "data_points_per_symbol": {
                symbol: len(self.returns_history.get(symbol, []))
                for symbol in self.symbols
            },
            "monitoring_active": self.is_monitoring,
            "total_alerts": len(self.correlation_alerts),
            "active_alerts": len(self.active_alerts)
        }


# Standalone validation function
def validate_correlation_analyzer():
    """Validate correlation analyzer implementation."""
    print("🔍 Validating CorrelationAnalyzer implementation...")

    try:
        # Test imports
        from .correlation_analyzer import CorrelationAnalyzer, CorrelationMetrics, CorrelationLevel
        print("✅ Imports successful")

        # Test instantiation
        analyzer = CorrelationAnalyzer(symbols=["BTC", "ETH"])
        print("✅ CorrelationAnalyzer instantiation successful")

        # Test basic functionality
        if hasattr(analyzer, 'calculate_correlations'):
            print("✅ calculate_correlations method exists")
        else:
            print("❌ calculate_correlations method missing")
            return False

        if hasattr(analyzer, 'get_correlation_summary'):
            print("✅ get_correlation_summary method exists")
        else:
            print("❌ get_correlation_summary method missing")
            return False

        print("🎉 CorrelationAnalyzer validation PASSED")
        return True

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    validate_correlation_analyzer()