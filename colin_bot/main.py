"""
Main entry point for Colin Trading Bot.

Institutional-Grade Signal Scoring Bot for Crypto Perpetuals
"""

import asyncio
import argparse
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

from .core.config import ConfigManager
from .engine.institutional_scorer import InstitutionalScorer
from .engine.enhanced_institutional_scorer import EnhancedInstitutionalScorer


class ColinTradingBot:
    """Main trading bot class."""

    def __init__(self, config_path: str = None, enable_hft: bool = True):
        """
        Initialize the trading bot.

        Args:
            config_path: Path to configuration file
            enable_hft: Whether to enable HFT signal integration
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.enable_hft = enable_hft

        # Initialize enhanced scorer with HFT integration
        if enable_hft:
            logger.info("Initializing Colin Trading Bot with HFT integration")
            try:
                self.scorer = EnhancedInstitutionalScorer(self.config_manager, enable_hft=True)
                self.bot_type = "enhanced_with_hft"
            except Exception as e:
                logger.warning(f"Failed to initialize HFT integration: {e}")
                logger.info("Falling back to conventional institutional scorer")
                self.scorer = InstitutionalScorer(self.config_manager)
                self.bot_type = "conventional"
        else:
            logger.info("Initializing Colin Trading Bot (conventional mode)")
            self.scorer = InstitutionalScorer(self.config_manager)
            self.bot_type = "conventional"

        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.logging

        # Remove default handler
        logger.remove()

        # Add console handler
        logger.add(
            sys.stdout,
            level=log_config.level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                  "<level>{level: <8}</level> | "
                  "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                  "<level>{message}</level>",
            colorize=True
        )

        # Add file handler if configured
        if log_config.file:
            log_path = Path(log_config.file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            logger.add(
                log_path,
                level=log_config.level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation=log_config.max_size,
                retention=log_config.backup_count,
                compression="zip"
            )

    async def analyze_single_symbol(self, symbol: str, time_horizon: str = "4h") -> Dict[str, Any]:
        """
        Analyze a single trading symbol.

        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            time_horizon: Analysis time horizon ("1h", "4h", "24h")

        Returns:
            Analysis results as dictionary
        """
        logger.info(f"Starting analysis for {symbol}")

        try:
            signal = await self.scorer.analyze_symbol(symbol, time_horizon=time_horizon)

            result = {
                'symbol': signal.symbol,
                'timestamp': signal.timestamp.isoformat(),
                'direction': signal.direction,
                'confidence_level': signal.confidence_level,
                'long_confidence': round(signal.long_confidence, 1),
                'short_confidence': round(signal.short_confidence, 1),
                'entry_price': signal.entry_price,
                'stop_loss_price': signal.stop_loss_price,
                'take_profit_price': signal.take_profit_price,
                'position_size': round(signal.position_size, 4),
                'time_horizon': signal.time_horizon,
                'rationale': signal.rationale_points,
                'risk_metrics': signal.risk_metrics,
                'institutional_factors': {
                    k: round(v, 3) for k, v in signal.institutional_factors.items()
                }
            }

            logger.info(f"Analysis complete for {symbol}: {signal.direction} "
                       f"with {signal.confidence_level} confidence")

            return result

        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'direction': 'neutral',
                'confidence_level': 'low'
            }

    async def analyze_multiple_symbols(
        self,
        symbols: List[str],
        time_horizon: str = "4h"
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple trading symbols.

        Args:
            symbols: List of trading symbols
            time_horizon: Analysis time horizon

        Returns:
            List of analysis results
        """
        logger.info(f"Starting batch analysis for {len(symbols)} symbols")

        signals = await self.scorer.batch_analyze(symbols, time_horizon=time_horizon)

        results = []
        for signal in signals:
            result = {
                'symbol': signal.symbol,
                'timestamp': signal.timestamp.isoformat(),
                'direction': signal.direction,
                'confidence_level': signal.confidence_level,
                'long_confidence': round(signal.long_confidence, 1),
                'short_confidence': round(signal.short_confidence, 1),
                'entry_price': signal.entry_price,
                'stop_loss_price': signal.stop_loss_price,
                'take_profit_price': signal.take_profit_price,
                'position_size': round(signal.position_size, 4),
                'time_horizon': signal.time_horizon,
                'rationale': signal.rationale_points,
                'risk_metrics': signal.risk_metrics,
                'institutional_factors': {
                    k: round(v, 3) for k, v in signal.institutional_factors.items()
                }
            }
            results.append(result)

        logger.info(f"Batch analysis complete: {len(results)} signals generated")
        return results

    def print_analysis_result(self, result: Dict[str, Any], format_type: str = "readable"):
        """
        Print analysis result in specified format.

        Args:
            result: Analysis result dictionary
            format_type: Output format ("readable", "json", "csv")
        """
        if format_type == "json":
            print(json.dumps(result, indent=2))
        elif format_type == "csv":
            if result.get('error'):
                print(f"{result['symbol']},ERROR,{result.get('error', '')}")
            else:
                print(f"{result['symbol']},{result['direction']},{result['confidence_level']},"
                      f"{result['long_confidence']},{result['short_confidence']},"
                      f"{result['entry_price']},{result['stop_loss_price']},"
                      f"\"{'; '.join(result['rationale'])}\"")
        else:  # readable format
            self._print_readable_format(result)

    def _print_readable_format(self, result: Dict[str, Any]):
        """Print result in human-readable format."""

        print("\n" + "="*60)
        print(f"🔍 INSTITUTIONAL SIGNAL ANALYSIS: {result['symbol']}")
        print("="*60)

        if result.get('error'):
            print(f"❌ Analysis failed: {result['error']}")
            return

        # Main signal
        direction_emoji = {"long": "🟢", "short": "🔴", "neutral": "🟡"}.get(result['direction'], "⚪")
        confidence_emoji = {"high": "🔥", "medium": "⚡", "low": "💧"}.get(result['confidence_level'], "❓")

        print(f"\n{direction_emoji} DIRECTION: {result['direction'].upper()}")
        print(f"{confidence_emoji} CONFIDENCE: {result['confidence_level'].upper()}")
        print(f"📊 Long Confidence: {result['long_confidence']}%")
        print(f"📊 Short Confidence: {result['short_confidence']}%")

        # Entry and risk levels
        print(f"\n💰 ENTRY PRICE: ${result['entry_price']:.2f}")
        if result['stop_loss_price']:
            print(f"🛑 STOP LOSS: ${result['stop_loss_price']:.2f}")
        if result['take_profit_price']:
            print(f"🎯 TAKE PROFIT: ${result['take_profit_price']:.2f}")
        print(f"📏 POSITION SIZE: {result['position_size']*100:.1f}%")

        # Rationale
        print(f"\n📋 RATIONALE:")
        for i, point in enumerate(result['rationale'], 1):
            print(f"   {i}. {point}")

        # Institutional factors
        print(f"\n🏦 INSTITUTIONAL FACTORS:")
        factors = result['institutional_factors']
        print(f"   Liquidity Score: {factors.get('liquidity', 0):.3f}")
        print(f"   ICT Score: {factors.get('ict', 0):.3f}")
        print(f"   Killzone Score: {factors.get('killzone', 0):.3f}")
        print(f"   Order Flow Score: {factors.get('order_flow', 0):.3f}")
        print(f"   Volume/OI Score: {factors.get('volume_oi', 0):.3f}")

        # Risk metrics
        if result['risk_metrics']:
            print(f"\n⚠️  RISK METRICS:")
            risk = result['risk_metrics']
            if 'volatility' in risk:
                print(f"   Volatility: {risk['volatility']*100:.1f}%")
            if 'signal_strength' in risk:
                print(f"   Signal Strength: {risk['signal_strength']:.3f}")
            if 'risk_reward_ratio' in risk:
                print(f"   Risk/Reward Ratio: 1:{risk['risk_reward_ratio']:.1f}")

        print(f"\n⏰ ANALYSIS TIME: {result['timestamp']}")
        print("="*60)

    async def run_continuous_analysis(
        self,
        symbols: List[str],
        interval_minutes: int = 15,
        time_horizon: str = "4h"
    ):
        """
        Run continuous analysis on specified symbols.

        Args:
            symbols: List of symbols to analyze
            interval_minutes: Analysis interval in minutes
            time_horizon: Analysis time horizon
        """
        logger.info(f"Starting continuous analysis for {symbols} every {interval_minutes} minutes")

        try:
            while True:
                logger.info("Running analysis cycle...")

                results = await self.analyze_multiple_symbols(symbols, time_horizon)

                # Print results
                print(f"\n🕐 Analysis Cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                for result in results:
                    self.print_analysis_result(result)

                # Wait for next interval
                logger.info(f"Waiting {interval_minutes} minutes for next analysis...")
                await asyncio.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("Continuous analysis stopped by user")
        except Exception as e:
            logger.error(f"Continuous analysis failed: {e}")
            raise

    async def get_hft_status(self) -> Dict[str, Any]:
        """
        Get HFT integration status and metrics.

        Returns:
            HFT status information
        """
        if not self.enable_hft:
            return {
                'hft_enabled': False,
                'bot_type': self.bot_type,
                'message': 'HFT integration is disabled'
            }

        try:
            # Check if scorer has HFT capabilities
            if hasattr(self.scorer, 'get_enhanced_metrics'):
                enhanced_metrics = self.scorer.get_enhanced_metrics()

                # Add bot-specific information
                status = {
                    'hft_enabled': True,
                    'bot_type': self.bot_type,
                    'enhanced_metrics': enhanced_metrics,
                    'timestamp': datetime.now().isoformat()
                }

                # Add health check if available
                if hasattr(self.scorer, 'health_check'):
                    health = await self.scorer.health_check()
                    status['health_check'] = health

                return status
            else:
                return {
                    'hft_enabled': False,
                    'bot_type': self.bot_type,
                    'message': 'HFT components not initialized - using conventional scorer'
                }

        except Exception as e:
            logger.error(f"Error getting HFT status: {e}")
            return {
                'hft_enabled': self.enable_hft,
                'bot_type': self.bot_type,
                'error': str(e),
                'message': 'Error retrieving HFT status'
            }

    async def test_hft_integration(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Test HFT integration with a sample symbol.

        Args:
            symbol: Symbol to test with

        Returns:
            Test results
        """
        if not self.enable_hft:
            return {
                'success': False,
                'message': 'HFT integration is disabled'
            }

        try:
            logger.info(f"Testing HFT integration with {symbol}")

            # Test enhanced analysis
            start_time = time.time()
            signal = await self.scorer.analyze_symbol(symbol, time_horizon="1h")
            analysis_time = time.time() - start_time

            # Check if signal has HFT components
            has_hft_components = hasattr(signal, 'institutional_factors') and any(
                key.startswith('hft_') for key in signal.institutional_factors.keys()
            )

            result = {
                'success': True,
                'symbol': symbol,
                'signal_generated': True,
                'has_hft_components': has_hft_components,
                'analysis_time_seconds': round(analysis_time, 3),
                'signal_direction': signal.direction,
                'signal_confidence': signal.confidence_level,
                'long_confidence': signal.long_confidence,
                'short_confidence': signal.short_confidence,
                'institutional_factors': signal.institutional_factors,
                'timestamp': datetime.now().isoformat()
            }

            if has_hft_components:
                result['hft_factors'] = {
                    k: v for k, v in signal.institutional_factors.items()
                    if k.startswith('hft_')
                }

            logger.info(f"HFT integration test successful for {symbol}")
            return result

        except Exception as e:
            logger.error(f"HFT integration test failed: {e}")
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Colin Trading Bot - Institutional Signal Scorer")
    parser.add_argument(
        "symbols",
        nargs="*",
        default=["ETHUSDT", "BTCUSDT"],
        help="Trading symbols to analyze (default: ETHUSDT BTCUSDT)"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--time-horizon",
        "-t",
        choices=["1h", "4h", "24h"],
        default="4h",
        help="Analysis time horizon (default: 4h)"
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["readable", "json", "csv"],
        default="readable",
        help="Output format (default: readable)"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuous analysis"
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=15,
        help="Analysis interval in minutes for continuous mode (default: 15)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path"
    )

    args = parser.parse_args()

    # Initialize bot
    bot = ColinTradingBot(args.config)

    if args.continuous:
        # Run continuous analysis
        await bot.run_continuous_analysis(
            args.symbols,
            args.interval,
            args.time_horizon
        )
    else:
        # Run single analysis
        if len(args.symbols) == 1:
            # Single symbol analysis
            result = await bot.analyze_single_symbol(args.symbols[0], args.time_horizon)

            if args.output:
                with open(args.output, 'w') as f:
                    if args.format == "json":
                        json.dump(result, f, indent=2)
                    else:
                        f.write(str(result))
                print(f"Results saved to {args.output}")
            else:
                bot.print_analysis_result(result, args.format)
        else:
            # Multiple symbols analysis
            results = await bot.analyze_multiple_symbols(args.symbols, args.time_horizon)

            if args.output:
                with open(args.output, 'w') as f:
                    if args.format == "json":
                        json.dump(results, f, indent=2)
                    else:
                        for result in results:
                            if args.format == "csv":
                                f.write(f"{result['symbol']},{result['direction']},"
                                       f"{result['confidence_level']},"
                                       f"{result['long_confidence']},"
                                       f"{result['short_confidence']}\n")
                            else:
                                f.write(str(result) + "\n")
                print(f"Results saved to {args.output}")
            else:
                for result in results:
                    bot.print_analysis_result(result, args.format)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        sys.exit(1)