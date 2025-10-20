"""
Enhanced Institutional Scorer with HFT Integration

This enhanced version of the institutional scorer integrates HFT signals
to provide superior signal quality and confidence.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger

from .institutional_scorer import (
    InstitutionalScorer, InstitutionalSignal, SignalComponents, ConfigManager
)
from .hft_integration_adapter import HFTIntegrationAdapter, create_hft_adapter
from .hft_signal_bridge import HFTSignalBridge


class EnhancedInstitutionalScorer(InstitutionalScorer):
    """
    Enhanced institutional scorer with HFT signal integration.

    This class extends the base InstitutionalScorer to include:
    - HFT signal generation and integration
    - Enhanced confidence scoring
    - Real-time market microstructure analysis
    - Improved risk metrics
    """

    def __init__(self, config_manager: ConfigManager, enable_hft: bool = True):
        """
        Initialize enhanced institutional scorer.

        Args:
            config_manager: Configuration manager instance
            enable_hft: Whether to enable HFT signal integration
        """
        super().__init__(config_manager)

        self.enable_hft = enable_hft

        # Initialize HFT components
        if self.enable_hft:
            logger.info("Initializing Enhanced Institutional Scorer with HFT integration")
            self.hft_adapter = create_hft_adapter(config_manager, enable_hft)
            self.hft_bridge = HFTSignalBridge(self.hft_adapter)
            self.hft_integration_config = self._get_hft_config()
        else:
            logger.info("HFT integration disabled - using conventional analysis only")
            self.hft_adapter = None
            self.hft_bridge = None
            self.hft_integration_config = {}

        # Enhanced scoring weights
        self.enhanced_weights = self._get_enhanced_weights()

        # Performance tracking
        self.enhanced_metrics = {
            'total_analyses': 0,
            'hft_enhanced_analyses': 0,
            'hft_only_analyses': 0,
            'confidence_improvements': [],
            'signal_alignment_scores': []
        }

    def _get_hft_config(self) -> Dict[str, Any]:
        """Get HFT integration configuration."""
        return {
            'hft_weight_in_final_signal': 0.35,  # 35% weight to HFT signals
            'confidence_boost_threshold': 70,    # Boost confidence above this threshold
            'signal_alignment_required': True,   # Require alignment between HFT and institutional
            'hft_strength_multiplier': {
                'strong': 1.2,
                'moderate': 1.1,
                'weak': 1.0
            },
            'enable_pure_hft_signals': False,   # Can generate pure HFT signals if needed
            'fallback_to_conventional': True    # Fall back if HFT fails
        }

    def _get_enhanced_weights(self) -> Dict[str, float]:
        """Get enhanced scoring weights including HFT components."""
        base_weights = self.scoring_weights.copy()

        if self.enable_hft:
            # Adjust weights to accommodate HFT signals
            total_weight = sum(base_weights.values())
            hft_weight = 0.35  # 35% to HFT
            remaining_weight = 0.65  # 65% to traditional components

            # Scale down traditional weights
            for key in base_weights:
                base_weights[key] = (base_weights[key] / total_weight) * remaining_weight

            # Add HFT weights
            base_weights.update({
                'hft_ofi': hft_weight * 0.4,      # 14% of total (40% of HFT weight)
                'hft_book_skew': hft_weight * 0.3, # 10.5% of total (30% of HFT weight)
                'hft_fusion': hft_weight * 0.3    # 10.5% of total (30% of HFT weight)
            })

        return base_weights

    async def analyze_symbol(
        self,
        symbol: str,
        current_price: Optional[float] = None,
        time_horizon: str = "4h",
        use_hft: Optional[bool] = None
    ) -> InstitutionalSignal:
        """
        Perform enhanced institutional analysis with HFT integration.

        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            current_price: Current price level (optional)
            time_horizon: Analysis time horizon ("1h", "4h", "24h")
            use_hft: Override HFT usage (defaults to instance setting)

        Returns:
            Enhanced institutional signal with HFT components
        """
        # Determine whether to use HFT
        should_use_hft = use_hft if use_hft is not None else self.enable_hft

        try:
            self.enhanced_metrics['total_analyses'] += 1

            if should_use_hft and self.hft_bridge and self.hft_bridge.is_available():
                logger.info(f"Starting enhanced analysis for {symbol} with HFT integration")
                return await self._enhanced_analysis(symbol, current_price, time_horizon)
            else:
                logger.info(f"Starting conventional analysis for {symbol} (HFT disabled)")
                return await super().analyze_symbol(symbol, current_price, time_horizon)

        except Exception as e:
            logger.error(f"Enhanced analysis failed for {symbol}: {e}")
            if self.hft_integration_config.get('fallback_to_conventional', True):
                logger.info(f"Falling back to conventional analysis for {symbol}")
                return await super().analyze_symbol(symbol, current_price, time_horizon)
            else:
                return self._create_error_signal(symbol, current_price, str(e))

    async def _enhanced_analysis(
        self,
        symbol: str,
        current_price: Optional[float],
        time_horizon: str
    ) -> InstitutionalSignal:
        """Perform enhanced analysis with HFT integration."""
        # Get current price if not provided
        if current_price is None:
            ticker = await self.binance_adapter.get_ticker(symbol)
            current_price = float(ticker['last'])

        # Generate HFT signal concurrently with traditional analysis
        hft_task = self.hft_bridge.enhance_institutional_signal(
            symbol, {'symbol': symbol, 'timestamp': datetime.now().isoformat()}
        )

        traditional_task = self._traditional_analysis_wrapper(symbol, current_price, time_horizon)

        # Execute both analyses
        hft_enhanced_signal, traditional_signal = await asyncio.gather(
            hft_task, traditional_task, return_exceptions=True
        )

        # Handle exceptions
        if isinstance(traditional_signal, Exception):
            logger.error(f"Traditional analysis failed for {symbol}: {traditional_signal}")
            traditional_signal = self._create_error_signal(symbol, current_price, str(traditional_signal))

        if isinstance(hft_enhanced_signal, Exception):
            logger.error(f"HFT enhancement failed for {symbol}: {hft_enhanced_signal}")
            self.enhanced_metrics['hft_enhanced_analyses'] += 1
            return traditional_signal

        # Merge the signals
        final_signal = self._merge_enhanced_signals(traditional_signal, hft_enhanced_signal, symbol)

        # Update metrics
        self._update_enhanced_metrics(traditional_signal, hft_enhanced_signal, final_signal)

        logger.info(f"Enhanced analysis complete for {symbol}: {final_signal.direction} "
                   f"(Long: {final_signal.long_confidence:.1f}%, Short: {final_signal.short_confidence:.1f}%)")

        return final_signal

    async def _traditional_analysis_wrapper(
        self,
        symbol: str,
        current_price: float,
        time_horizon: str
    ) -> InstitutionalSignal:
        """Wrapper for traditional analysis to get signal in dict format."""
        signal = await super().analyze_symbol(symbol, current_price, time_horizon)

        # Convert to dict format for merging
        return {
            'symbol': signal.symbol,
            'timestamp': signal.timestamp.isoformat(),
            'direction': signal.direction,
            'confidence_level': signal.confidence_level,
            'long_confidence': signal.long_confidence,
            'short_confidence': signal.short_confidence,
            'entry_price': signal.entry_price,
            'stop_loss_price': signal.stop_loss_price,
            'take_profit_price': signal.take_profit_price,
            'position_size': signal.position_size,
            'time_horizon': signal.time_horizon,
            'rationale_points': signal.rationale_points,
            'risk_metrics': signal.risk_metrics,
            'institutional_factors': signal.institutional_factors
        }

    def _merge_enhanced_signals(
        self,
        traditional_signal: Dict,
        hft_enhanced_signal: Dict,
        symbol: str
    ) -> InstitutionalSignal:
        """Merge traditional and HFT-enhanced signals."""
        # Use the HFT enhanced signal as base (it already contains traditional components)
        merged = hft_enhanced_signal.copy()

        # Ensure we have all required fields from traditional signal
        for key, value in traditional_signal.items():
            if key not in merged:
                merged[key] = value

        # Calculate confidence improvement
        trad_long = traditional_signal.get('long_confidence', 50)
        trad_short = traditional_signal.get('short_confidence', 50)
        enh_long = merged.get('enhanced_long_confidence', trad_long)
        enh_short = merged.get('enhanced_short_confidence', trad_short)

        confidence_improvement = max(enh_long - trad_long, enh_short - trad_short)
        self.enhanced_metrics['confidence_improvements'].append(confidence_improvement)

        # Calculate signal alignment
        alignment = self._calculate_signal_alignment_score(traditional_signal, merged)
        self.enhanced_metrics['signal_alignment_scores'].append(alignment)

        # Add enhanced metadata
        merged['analysis_metadata'] = {
            'analysis_type': 'enhanced_with_hft',
            'confidence_improvement': round(confidence_improvement, 1),
            'signal_alignment_score': round(alignment, 3),
            'hft_components_active': True,
            'enhancement_timestamp': datetime.now(timezone.utc).isoformat()
        }

        # Convert back to InstitutionalSignal format
        return self._dict_to_institutional_signal(merged)

    def _dict_to_institutional_signal(self, signal_dict: Dict) -> InstitutionalSignal:
        """Convert signal dictionary back to InstitutionalSignal format."""
        return InstitutionalSignal(
            symbol=signal_dict['symbol'],
            timestamp=datetime.fromisoformat(signal_dict['timestamp'].replace('Z', '+00:00')),
            long_confidence=signal_dict.get('enhanced_long_confidence', signal_dict.get('long_confidence', 50)),
            short_confidence=signal_dict.get('enhanced_short_confidence', signal_dict.get('short_confidence', 50)),
            direction=signal_dict['direction'],
            confidence_level=signal_dict.get('confidence_level', 'medium'),
            rationale_points=signal_dict.get('rationale_points', []),
            risk_metrics=signal_dict.get('risk_metrics', {}),
            entry_price=signal_dict.get('entry_price', 0),
            stop_loss_price=signal_dict.get('stop_loss_price'),
            take_profit_price=signal_dict.get('take_profit_price'),
            position_size=signal_dict.get('position_size', 0.1),
            time_horizon=signal_dict.get('time_horizon', '4h'),
            institutional_factors=signal_dict.get('institutional_factors', {})
        )

    def _calculate_signal_alignment_score(self, traditional: Dict, enhanced: Dict) -> float:
        """Calculate alignment score between traditional and enhanced signals."""
        trad_dir = traditional.get('direction', 'neutral')
        enh_dir = enhanced.get('direction', 'neutral')

        if trad_dir == enh_dir:
            return 1.0
        elif trad_dir == 'neutral' or enh_dir == 'neutral':
            return 0.5
        else:
            return 0.0

    def _update_enhanced_metrics(
        self,
        traditional: Dict,
        hft_enhanced: Dict,
        final: InstitutionalSignal
    ):
        """Update enhanced performance metrics."""
        self.enhanced_metrics['hft_enhanced_analyses'] += 1

        # Keep only last 1000 metrics
        if len(self.enhanced_metrics['confidence_improvements']) > 1000:
            self.enhanced_metrics['confidence_improvements'] = self.enhanced_metrics['confidence_improvements'][-1000:]

        if len(self.enhanced_metrics['signal_alignment_scores']) > 1000:
            self.enhanced_metrics['signal_alignment_scores'] = self.enhanced_metrics['signal_alignment_scores'][-1000:]

    async def generate_pure_hft_signal(
        self,
        symbol: str,
        time_horizon: str = "1h"
    ) -> Optional[InstitutionalSignal]:
        """
        Generate a pure HFT signal (no traditional analysis).

        Args:
            symbol: Trading symbol
            time_horizon: Analysis time horizon

        Returns:
            Pure HFT signal or None
        """
        if not self.hft_bridge or not self.hft_bridge.is_available():
            logger.warning("Pure HFT signals not available")
            return None

        try:
            logger.info(f"Generating pure HFT signal for {symbol}")
            hft_signal_dict = await self.hft_bridge.generate_pure_hft_signal(symbol, time_horizon)

            if hft_signal_dict:
                self.enhanced_metrics['hft_only_analyses'] += 1
                return self._dict_to_institutional_signal(hft_signal_dict)
            else:
                return None

        except Exception as e:
            logger.error(f"Error generating pure HFT signal for {symbol}: {e}")
            return None

    async def batch_analyze_with_hft(
        self,
        symbols: List[str],
        time_horizon: str = "4h",
        max_concurrent: int = 5
    ) -> List[InstitutionalSignal]:
        """
        Analyze multiple symbols with HFT enhancement.

        Args:
            symbols: List of symbols to analyze
            time_horizon: Analysis time horizon
            max_concurrent: Maximum concurrent analyses

        Returns:
            List of enhanced signals
        """
        logger.info(f"Starting batch enhanced analysis for {len(symbols)} symbols")

        # Process in batches to avoid overwhelming resources
        results = []
        for i in range(0, len(symbols), max_concurrent):
            batch = symbols[i:i + max_concurrent]
            tasks = [
                self.analyze_symbol(symbol, time_horizon=time_horizon)
                for symbol in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch analysis failed for {batch[j]}: {result}")
                    # Create error signal
                    results.append(self._create_error_signal(batch[j], None, str(result)))
                else:
                    results.append(result)

        logger.info(f"Batch enhanced analysis complete: {len(results)} signals generated")
        return results

    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get enhanced performance metrics."""
        metrics = self.enhanced_metrics.copy()

        # Add calculated statistics
        if metrics['confidence_improvements']:
            metrics['avg_confidence_improvement'] = sum(metrics['confidence_improvements']) / len(metrics['confidence_improvements'])
            metrics['max_confidence_improvement'] = max(metrics['confidence_improvements'])
            metrics['min_confidence_improvement'] = min(metrics['confidence_improvements'])

        if metrics['signal_alignment_scores']:
            metrics['avg_signal_alignment'] = sum(metrics['signal_alignment_scores']) / len(metrics['signal_alignment_scores'])

        # Add HFT status
        if self.hft_bridge:
            metrics['hft_bridge_status'] = self.hft_bridge.get_bridge_status()
        else:
            metrics['hft_bridge_status'] = {'bridge_available': False}

        # Add configuration
        metrics['hft_integration_config'] = self.hft_integration_config
        metrics['enhanced_weights'] = self.enhanced_weights

        return metrics

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of enhanced scorer."""
        health_status = {
            'status': 'healthy',
            'components': {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        # Check traditional components
        try:
            # Test traditional analysis with a simple symbol
            test_signal = await super().analyze_symbol('BTCUSDT', time_horizon="1h")
            health_status['components']['traditional_analysis'] = 'healthy'
        except Exception as e:
            health_status['components']['traditional_analysis'] = f'error: {e}'
            health_status['status'] = 'degraded'

        # Check HFT components
        if self.hft_bridge:
            try:
                hft_health = await self.hft_adapter.health_check()
                health_status['components']['hft_integration'] = hft_health
                if hft_health.get('status') != 'healthy':
                    health_status['status'] = 'degraded'
            except Exception as e:
                health_status['components']['hft_integration'] = f'error: {e}'
                if self.enable_hft:
                    health_status['status'] = 'degraded'
        else:
            health_status['components']['hft_integration'] = 'disabled'

        # Add metrics
        health_status['metrics'] = self.get_enhanced_metrics()

        return health_status