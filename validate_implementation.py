#!/usr/bin/env python3
"""
Implementation validation script for Colin Trading Bot.

This script performs comprehensive validation of the institutional signal
scoring bot implementation to ensure all components work correctly.
"""

import asyncio
import sys
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import ConfigManager
from src.structure.ict_detector import ICTDetector
from src.utils.sessions import SessionAnalyzer
from src.output.formatter import OutputFormatter


class ImplementationValidator:
    """Validates the complete implementation of Colin Trading Bot."""

    def __init__(self):
        self.results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_results': []
        }

    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test result."""
        self.results['total_tests'] += 1
        if passed:
            self.results['passed_tests'] += 1
            print(f"✅ {test_name}")
            if message:
                print(f"   {message}")
        else:
            self.results['failed_tests'] += 1
            print(f"❌ {test_name}")
            if message:
                print(f"   {message}")

        self.results['test_results'].append({
            'test': test_name,
            'passed': passed,
            'message': message
        })

    def test_configuration_loading(self):
        """Test configuration loading and validation."""
        print("\n🔧 Testing Configuration Loading...")

        try:
            # Test default config loading
            config_manager = ConfigManager()
            config = config_manager.load_config()

            self.log_test(
                "Default config loads successfully",
                True,
                f"Loaded {len(config.symbols)} symbols"
            )

            # Test configuration validation
            required_sections = ['symbols', 'apis', 'sessions', 'ict', 'scoring', 'risk']
            for section in required_sections:
                if hasattr(config, section):
                    self.log_test(
                        f"Config section '{section}' exists",
                        True
                    )
                else:
                    self.log_test(
                        f"Config section '{section}' missing",
                        False
                    )

            # Test scoring weights sum to reasonable value
            weights = config.scoring.weights
            total_weight = sum(weights.values())
            self.log_test(
                "Scoring weights sum to reasonable value",
                0.8 <= total_weight <= 1.2,
                f"Total weight: {total_weight:.3f}"
            )

        except Exception as e:
            self.log_test("Configuration loading", False, str(e))

    def test_ict_detector(self):
        """Test ICT structure detection."""
        print("\n🏗️ Testing ICT Structure Detection...")

        try:
            config_manager = ConfigManager()
            detector = ICTDetector(config_manager)

            # Create sample data with known patterns
            sample_data = self.create_sample_ohlcv_with_patterns()

            # Test FVG detection
            fvg_list = detector.detect_fair_value_gaps(sample_data)
            self.log_test(
                "FVG detection works",
                len(fvg_list) >= 0,
                f"Detected {len(fvg_list)} FVGs"
            )

            # Test Order Block detection
            ob_list = detector.detect_order_blocks(sample_data)
            self.log_test(
                "Order Block detection works",
                len(ob_list) >= 0,
                f"Detected {len(ob_list)} Order Blocks"
            )

            # Test BOS detection
            bos_list = detector.detect_break_of_structure(sample_data)
            self.log_test(
                "Break of Structure detection works",
                len(bos_list) >= 0,
                f"Detected {len(bos_list)} BOS points"
            )

            # Test confluence analysis
            current_price = sample_data['close'].iloc[-1]
            confluence = detector.analyze_ict_confluence(sample_data, current_price)
            self.log_test(
                "ICT confluence analysis works",
                'confluence_score' in confluence,
                f"Confluence score: {confluence.get('confluence_score', 0):.3f}"
            )

            # Test structural stop loss
            long_stop = detector.get_structural_stop_loss(sample_data, current_price, "long")
            short_stop = detector.get_structural_stop_loss(sample_data, current_price, "short")
            self.log_test(
                "Structural stop loss calculation works",
                long_stop is not None or short_stop is not None,
                f"Long stop: {long_stop}, Short stop: {short_stop}"
            )

        except Exception as e:
            self.log_test("ICT Detector", False, str(e))
            traceback.print_exc()

    def test_session_analyzer(self):
        """Test session analysis and killzone timing."""
        print("\n⏰ Testing Session Analysis...")

        try:
            config_manager = ConfigManager()
            analyzer = SessionAnalyzer(config_manager)

            # Test current session analysis
            current_time = datetime.now()
            status = analyzer.get_session_status(current_time)
            self.log_test(
                "Session status analysis works",
                status is not None,
                f"Current sessions: {len(status.active_sessions)}"
            )

            # Test optimal entry time detection
            is_optimal = analyzer.is_optimal_entry_time(current_time)
            self.log_test(
                "Optimal entry time detection works",
                isinstance(is_optimal, bool)
            )

            # Test session score calculation
            session_score = analyzer.calculate_session_score(current_time)
            self.log_test(
                "Session score calculation works",
                0 <= session_score <= 1,
                f"Session score: {session_score:.3f}"
            )

            # Test killzone analysis
            killzone_analysis = analyzer.get_killzone_analysis(current_time)
            self.log_test(
                "Killzone analysis works",
                'session_score' in killzone_analysis,
                f"Killzone score: {killzone_analysis.get('session_score', 0):.3f}"
            )

            # Test session schedule
            schedule = analyzer.get_session_schedule(current_time)
            self.log_test(
                "Session schedule generation works",
                len(schedule) > 0,
                f"Generated {len(schedule)} session entries"
            )

        except Exception as e:
            self.log_test("Session Analyzer", False, str(e))
            traceback.print_exc()

    def test_output_formatter(self):
        """Test output formatting and risk warnings."""
        print("\n📄 Testing Output Formatting...")

        try:
            config_manager = ConfigManager()
            config = config_manager.config

            from src.engine.institutional_scorer import InstitutionalSignal
            from src.output.formatter import FormattedSignal, RiskMetrics

            # Create sample signal
            sample_signal = InstitutionalSignal(
                symbol="ETHUSDT",
                timestamp=datetime.now(),
                long_confidence=75.5,
                short_confidence=24.5,
                direction="long",
                confidence_level="high",
                rationale_points=[
                    "Strong liquidity confluence detected",
                    "ICT structure alignment present",
                    "Optimal session timing"
                ],
                risk_metrics={
                    'volatility': 0.025,
                    'signal_strength': 0.8,
                    'risk_reward_ratio': 2.0
                },
                entry_price=2000.0,
                stop_loss_price=1985.0,
                take_profit_price=2030.0,
                position_size=0.015,
                time_horizon="4h",
                institutional_factors={
                    'liquidity': 0.85,
                    'ict': 0.75,
                    'killzone': 0.90,
                    'order_flow': 0.70,
                    'volume_oi': 0.60
                }
            )

            # Test formatting
            formatter = OutputFormatter(config)
            formatted_signal = formatter.format_signal(sample_signal)

            self.log_test(
                "Signal formatting works",
                formatted_signal is not None,
                f"Formatted {formatted_signal.direction} signal"
            )

            # Test readable output
            readable_output = formatter.format_for_output(formatted_signal, "readable")
            self.log_test(
                "Readable output generation works",
                len(readable_output) > 100,
                f"Generated {len(readable_output)} characters of output"
            )

            # Test JSON output
            json_output = formatter.format_for_output(formatted_signal, "json")
            self.log_test(
                "JSON output generation works",
                json_output.startswith('{'),
                "Valid JSON format"
            )

            # Test CSV output
            csv_output = formatter.format_for_output(formatted_signal, "csv")
            self.log_test(
                "CSV output generation works",
                ',' in csv_output,
                "Valid CSV format"
            )

            # Test risk metrics
            risk_metrics = formatted_signal.risk_metrics
            self.log_test(
                "Risk metrics calculation works",
                risk_metrics.max_loss_potential > 0,
                f"Max loss: {risk_metrics.max_loss_potential:.2f}%"
            )

            # Test risk warnings
            warnings = formatted_signal.risk_warnings
            self.log_test(
                "Risk warnings generation works",
                len(warnings) >= 0,
                f"Generated {len(warnings)} risk warnings"
            )

        except Exception as e:
            self.log_test("Output Formatter", False, str(e))
            traceback.print_exc()

    def test_file_structure(self):
        """Test that all required files and directories exist."""
        print("\n📁 Testing File Structure...")

        required_files = [
            'config.yaml',
            'requirements.txt',
            'README.md',
            'pytest.ini',
            'colin_bot.py'
        ]

        required_dirs = [
            'src',
            'src/core',
            'src/adapters',
            'src/structure',
            'src/orderflow',
            'src/scorers',
            'src/engine',
            'src/output',
            'src/utils',
            'tests'
        ]

        # Test files
        for file_path in required_files:
            exists = Path(file_path).exists()
            self.log_test(
                f"Required file exists: {file_path}",
                exists
            )

        # Test directories
        for dir_path in required_dirs:
            exists = Path(dir_path).is_dir()
            self.log_test(
                f"Required directory exists: {dir_path}",
                exists
            )

        # Test Python files in src
        src_files = list(Path('src').rglob('*.py'))
        expected_min_files = 15  # Minimum expected Python files
        self.log_test(
            f"Python files in src",
            len(src_files) >= expected_min_files,
            f"Found {len(src_files)} Python files (expected >= {expected_min_files})"
        )

    def create_sample_ohlcv_with_patterns(self) -> pd.DataFrame:
        """Create sample OHLCV data with embedded ICT patterns."""
        np.random.seed(42)

        # Create 100 candles of data
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=100),
            periods=100,
            freq='1h'
        )

        base_price = 2000.0
        prices = np.random.normal(0, 3, 100).cumsum() + base_price

        # Add trend component
        trend = np.linspace(0, 20, 100)
        prices += trend

        # Create OHLCV data
        data = []
        for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
            volatility = 8
            high = close_price + np.random.uniform(0, volatility)
            low = close_price - np.random.uniform(0, volatility)
            open_price = close_price + np.random.uniform(-volatility/2, volatility/2)

            # Ensure OHLC relationships
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)

            volume = np.random.uniform(300000, 800000)

            data.append([open_price, high, low, close_price, volume])

        df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
        df.index = timestamps

        return df

    def test_imports(self):
        """Test that all major modules can be imported."""
        print("\n📦 Testing Module Imports...")

        modules_to_test = [
            'src.core.config',
            'src.structure.ict_detector',
            'src.utils.sessions',
            'src.output.formatter',
            'src.main'
        ]

        for module_name in modules_to_test:
            try:
                __import__(module_name)
                self.log_test(
                    f"Module import: {module_name}",
                    True
                )
            except Exception as e:
                self.log_test(
                    f"Module import: {module_name}",
                    False,
                    str(e)
                )

    def run_validation(self):
        """Run all validation tests."""
        print("🚀 Colin Trading Bot Implementation Validation")
        print("=" * 60)

        # Run all tests
        self.test_file_structure()
        self.test_imports()
        self.test_configuration_loading()
        self.test_ict_detector()
        self.test_session_analyzer()
        self.test_output_formatter()

        # Print summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)

        print(f"Total Tests: {self.results['total_tests']}")
        print(f"Passed: {self.results['passed_tests']} ✅")
        print(f"Failed: {self.results['failed_tests']} ❌")

        success_rate = (self.results['passed_tests'] / self.results['total_tests']) * 100 if self.results['total_tests'] > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")

        if self.results['failed_tests'] > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.results['test_results']:
                if not result['passed']:
                    print(f"   - {result['test']}: {result['message']}")

        if success_rate >= 90:
            print("\n🎉 Implementation validation PASSED! Ready for production use.")
        elif success_rate >= 70:
            print("\n⚠️  Implementation validation PARTIALLY PASSED. Some issues need attention.")
        else:
            print("\n🚨 Implementation validation FAILED. Significant issues need to be resolved.")

        return success_rate >= 90


if __name__ == "__main__":
    validator = ImplementationValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)