"""
Performance monitoring and utilities for HFT engine.

Latency tracking, performance metrics, and optimization tools.
"""

import time
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import threading
import asyncio


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    timestamp: datetime
    latency_ms: float
    component: str


class LatencyTracker:
    """
    Track and analyze system latency for HFT operations.

    Provides real-time latency monitoring and statistical analysis.
    """

    def __init__(self, max_measurements: int = 10000):
        self.max_measurements = max_measurements
        self.measurements: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_measurements))
        self.lock = threading.Lock()

    def record_latency(self, component: str, start_time: datetime, end_time: datetime):
        """
        Record a latency measurement.

        Args:
            component: Component name
            start_time: Start timestamp
            end_time: End timestamp
        """
        latency_ms = (end_time - start_time).total_seconds() * 1000

        measurement = LatencyMeasurement(
            timestamp=datetime.now(timezone.utc),
            latency_ms=latency_ms,
            component=component
        )

        with self.lock:
            self.measurements[component].append(measurement)

    def record_latency_ms(self, component: str, latency_ms: float):
        """
        Record latency directly in milliseconds.

        Args:
            component: Component name
            latency_ms: Latency in milliseconds
        """
        measurement = LatencyMeasurement(
            timestamp=datetime.now(timezone.utc),
            latency_ms=latency_ms,
            component=component
        )

        with self.lock:
            self.measurements[component].append(measurement)

    def get_average_latency(self, component: str, window_seconds: int = 60) -> float:
        """
        Get average latency for a component.

        Args:
            component: Component name
            window_seconds: Time window in seconds

        Returns:
            Average latency in seconds
        """
        with self.lock:
            measurements = self.measurements.get(component, [])
            if not measurements:
                return 0.0

            # Filter by time window
            cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
            recent_measurements = [m for m in measurements if m.timestamp >= cutoff_time]

            if not recent_measurements:
                return 0.0

            return statistics.mean(m.latency_ms for m in recent_measurements) / 1000.0

    def get_percentile_latency(self, component: str, percentile: float = 95.0) -> float:
        """
        Get percentile latency for a component.

        Args:
            component: Component name
            percentile: Percentile (0-100)

        Returns:
            Percentile latency in seconds
        """
        with self.lock:
            measurements = self.measurements.get(component, [])
            if not measurements:
                return 0.0

            latencies = [m.latency_ms for m in measurements]
            return statistics.quantiles(latencies, n=100)[int(percentile)] / 1000.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive latency statistics."""
        with self.lock:
            stats = {}
            for component, measurements in self.measurements.items():
                if not measurements:
                    continue

                latencies = [m.latency_ms for m in measurements]

                stats[component] = {
                    'count': len(measurements),
                    'avg_ms': statistics.mean(latencies),
                    'median_ms': statistics.median(latencies),
                    'min_ms': min(latencies),
                    'max_ms': max(latencies),
                    'p95_ms': statistics.quantiles(latencies, n=100)[94],
                    'p99_ms': statistics.quantiles(latencies, n=100)[98],
                    'std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0.0
                }

            return stats


@dataclass
class MetricValue:
    """Single metric value."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Monitor and analyze performance metrics for HFT system.

    Tracks system performance, resource usage, and trading metrics.
    """

    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics))
        self.lock = threading.Lock()

    def record_metric(self, name: str, value: float, metadata: Dict[str, Any] = None):
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            metadata: Additional metadata
        """
        metric = MetricValue(
            timestamp=datetime.now(timezone.utc),
            value=value,
            metadata=metadata or {}
        )

        with self.lock:
            self.metrics[name].append(metric)

    def record_counter(self, name: str, increment: float = 1.0):
        """Record a counter metric."""
        current_value = self.get_current_value(name) or 0.0
        self.record_metric(name, current_value + increment)

    def get_current_value(self, name: str) -> Optional[float]:
        """Get current value of a metric."""
        with self.lock:
            metrics = self.metrics.get(name)
            return metrics[-1].value if metrics else None

    def get_average_value(self, name: str, window_seconds: int = 60) -> float:
        """
        Get average value of a metric over time window.

        Args:
            name: Metric name
            window_seconds: Time window in seconds

        Returns:
            Average value
        """
        with self.lock:
            metrics = self.metrics.get(name, [])
            if not metrics:
                return 0.0

            # Filter by time window
            cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
            recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]

            if not recent_metrics:
                return 0.0

            return statistics.mean(m.value for m in recent_metrics)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        with self.lock:
            summary = {}
            for name, metrics in self.metrics.items():
                if not metrics:
                    continue

                values = [m.value for m in metrics]
                summary[name] = {
                    'current': values[-1] if values else 0.0,
                    'avg': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }

            return summary


class HFTPerformanceProfiler:
    """
    Performance profiler for HFT operations.

    Context manager for measuring execution time and performance.
    """

    def __init__(self, name: str, latency_tracker: LatencyTracker):
        self.name = name
        self.latency_tracker = latency_tracker
        self.start_time = None

    def __enter__(self):
        """Start profiling."""
        self.start_time = datetime.now(timezone.utc)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End profiling and record latency."""
        if self.start_time:
            end_time = datetime.now(timezone.utc)
            self.latency_tracker.record_latency(self.name, self.start_time, end_time)


class CircuitBreakerMetrics:
    """
    Track circuit breaker performance and triggering patterns.
    """

    def __init__(self):
        self.trigger_events: List[Dict[str, Any]] = []
        self.performance_before_triggers: List[float] = []
        self.performance_after_triggers: List[float] = []

    def record_trigger(self, trigger_type: str, reason: str, severity: str, metadata: Dict[str, Any] = None):
        """Record a circuit breaker trigger event."""
        event = {
            'timestamp': datetime.now(timezone.utc),
            'type': trigger_type,
            'reason': reason,
            'severity': severity,
            'metadata': metadata or {}
        }
        self.trigger_events.append(event)

    def record_performance_before_trigger(self, metric_value: float):
        """Record performance metric before circuit breaker trigger."""
        self.performance_before_triggers.append(metric_value)

    def record_performance_after_trigger(self, metric_value: float):
        """Record performance metric after circuit breaker trigger."""
        self.performance_after_triggers.append(metric_value)

    def get_trigger_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker trigger statistics."""
        if not self.trigger_events:
            return {}

        # Count triggers by type
        trigger_counts = {}
        for event in self.trigger_events:
            trigger_counts[event['type']] = trigger_counts.get(event['type'], 0) + 1

        # Calculate average improvement
        avg_improvement = 0.0
        if self.performance_before_triggers and self.performance_after_triggers:
            before_avg = statistics.mean(self.performance_before_triggers)
            after_avg = statistics.mean(self.performance_after_triggers)
            avg_improvement = (before_avg - after_avg) / before_avg if before_avg > 0 else 0.0

        return {
            'total_triggers': len(self.trigger_events),
            'triggers_by_type': trigger_counts,
            'average_improvement_percent': avg_improvement * 100,
            'last_trigger': self.trigger_events[-1] if self.trigger_events else None
        }


def profile_hft_operation(name: str, latency_tracker: LatencyTracker):
    """
    Decorator for profiling HFT operations.

    Args:
        name: Operation name
        latency_tracker: Latency tracker instance
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with HFTPerformanceProfiler(name, latency_tracker):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def profile_async_hft_operation(name: str, latency_tracker: LatencyTracker):
    """
    Decorator for profiling async HFT operations.

    Args:
        name: Operation name
        latency_tracker: Latency tracker instance
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = datetime.now(timezone.utc)
            try:
                result = await func(*args, **kwargs)
                end_time = datetime.now(timezone.utc)
                latency_tracker.record_latency(name, start_time, end_time)
                return result
            except Exception as e:
                end_time = datetime.now(timezone.utc)
                latency_tracker.record_latency(name, start_time, end_time)
                raise
        return wrapper
    return decorator


class ResourceMonitor:
    """
    Monitor system resources for HFT performance.

    Tracks CPU, memory, and network usage.
    """

    def __init__(self):
        self.metrics = defaultdict(list)
        self.lock = threading.Lock()

    def record_cpu_usage(self, cpu_percent: float):
        """Record CPU usage percentage."""
        with self.lock:
            self.metrics['cpu'].append({
                'timestamp': datetime.now(timezone.utc),
                'value': cpu_percent
            })

    def record_memory_usage(self, memory_mb: float):
        """Record memory usage in MB."""
        with self.lock:
            self.metrics['memory'].append({
                'timestamp': datetime.now(timezone.utc),
                'value': memory_mb
            })

    def record_network_io(self, bytes_sent: float, bytes_received: float):
        """Record network I/O statistics."""
        with self.lock:
            self.metrics['network'].append({
                'timestamp': datetime.now(timezone.utc),
                'bytes_sent': bytes_sent,
                'bytes_received': bytes_received
            })

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        with self.lock:
            summary = {}

            # CPU summary
            if self.metrics['cpu']:
                cpu_values = [m['value'] for m in self.metrics['cpu']]
                summary['cpu'] = {
                    'current': cpu_values[-1] if cpu_values else 0.0,
                    'avg': statistics.mean(cpu_values) if cpu_values else 0.0,
                    'max': max(cpu_values) if cpu_values else 0.0
                }

            # Memory summary
            if self.metrics['memory']:
                memory_values = [m['value'] for m in self.metrics['memory']]
                summary['memory'] = {
                    'current': memory_values[-1] if memory_values else 0.0,
                    'avg': statistics.mean(memory_values) if memory_values else 0.0,
                    'max': max(memory_values) if memory_values else 0.0
                }

            # Network summary
            if self.metrics['network']:
                recent_network = self.metrics['network'][-1]
                summary['network'] = {
                    'bytes_sent_total': recent_network['bytes_sent'],
                    'bytes_received_total': recent_network['bytes_received']
                }

            return summary


async def measure_async_latency(func, *args, **kwargs):
    """
    Measure execution time of an async function.

    Args:
        func: Async function to measure
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Tuple of (result, execution_time_seconds)
    """
    start_time = time.time()
    try:
        result = await func(*args, **kwargs)
        return result, time.time() - start_time
    except Exception as e:
        return e, time.time() - start_time