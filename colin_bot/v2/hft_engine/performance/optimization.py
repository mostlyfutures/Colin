"""
Performance Optimization for Sub-50ms Latency

Advanced performance optimization techniques including caching,
memory management, CPU optimization, and I/O optimization
for high-frequency trading systems.
"""

import asyncio
import time
import psutil
import gc
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import pickle
from functools import lru_cache, wraps
import weakref


@dataclass
class LatencyMetric:
    """Latency measurement data."""
    operation: str
    start_time: float
    end_time: float
    latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    thread_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceProfile:
    """Performance profile for an operation."""
    operation: str
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    total_operations: int
    operations_per_second: float
    memory_efficiency: float
    cpu_efficiency: float


class MemoryPool:
    """High-performance memory pool for object allocation."""

    def __init__(self, object_class: type, initial_size: int = 1000, max_size: int = 10000):
        self.object_class = object_class
        self.initial_size = initial_size
        self.max_size = max_size
        self.pool = deque()
        self.created_count = 0
        self.lock = threading.Lock()

        # Pre-allocate objects
        self._pre_allocate()

    def _pre_allocate(self):
        """Pre-allocate objects to the pool."""
        for _ in range(self.initial_size):
            obj = self.object_class()
            self.pool.append(obj)
            self.created_count += 1

    def acquire(self) -> Any:
        """Acquire an object from the pool."""
        with self.lock:
            if self.pool:
                return self.pool.popleft()
            elif self.created_count < self.max_size:
                obj = self.object_class()
                self.created_count += 1
                return obj
            else:
                # Pool is full, create temporary object
                return self.object_class()

    def release(self, obj: Any):
        """Release an object back to the pool."""
        with self.lock:
            if len(self.pool) < self.max_size:
                # Reset object state if possible
                if hasattr(obj, 'reset'):
                    obj.reset()
                self.pool.append(obj)

    def get_stats(self) -> Dict[str, int]:
        """Get memory pool statistics."""
        with self.lock:
            return {
                'pool_size': len(self.pool),
                'created_count': self.created_count,
                'max_size': self.max_size
            }


class CircularBuffer:
    """Lock-free circular buffer for high-performance data storage."""

    def __init__(self, size: int):
        self.size = size
        self.buffer = [None] * size
        self.head = 0
        self.tail = 0
        self.count = 0
        self.lock = threading.Lock()

    def append(self, item: Any):
        """Append item to buffer."""
        with self.lock:
            self.buffer[self.tail] = item
            self.tail = (self.tail + 1) % self.size
            if self.count < self.size:
                self.count += 1
            else:
                self.head = (self.head + 1) % self.size

    def get_latest(self, n: int = 1) -> List[Any]:
        """Get latest n items."""
        with self.lock:
            if self.count == 0:
                return []

            items = []
            for i in range(min(n, self.count)):
                idx = (self.tail - 1 - i) % self.size
                items.append(self.buffer[idx])

            return items

    def get_all(self) -> List[Any]:
        """Get all items in buffer."""
        with self.lock:
            if self.count == 0:
                return []

            items = []
            for i in range(self.count):
                idx = (self.head + i) % self.size
                items.append(self.buffer[idx])

            return items


class LatencyTracker:
    """High-performance latency tracking system."""

    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.metrics: Dict[str, CircularBuffer] = defaultdict(lambda: CircularBuffer(max_samples))
        self.lock = threading.Lock()

    def start_operation(self, operation: str) -> float:
        """Start timing an operation."""
        return time.perf_counter()

    def end_operation(self, operation: str, start_time: float, metadata: Dict = None) -> LatencyMetric:
        """End timing an operation and record metric."""
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Get system metrics
        process = psutil.Process()
        memory_usage_mb = process.memory_info().rss / 1024 / 1024
        cpu_usage_percent = process.cpu_percent()
        thread_id = threading.get_ident()

        metric = LatencyMetric(
            operation=operation,
            start_time=start_time,
            end_time=end_time,
            latency_ms=latency_ms,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent,
            thread_id=thread_id,
            metadata=metadata or {}
        )

        self.metrics[operation].append(metric)
        return metric

    def get_profile(self, operation: str) -> Optional[PerformanceProfile]:
        """Get performance profile for an operation."""
        if operation not in self.metrics:
            return None

        metrics = self.metrics[operation].get_all()
        if not metrics:
            return None

        latencies = [m.latency_ms for m in metrics]

        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)

        # Calculate operations per second
        if len(metrics) > 1:
            time_span = metrics[-1].end_time - metrics[0].start_time
            ops_per_second = len(metrics) / time_span if time_span > 0 else 0
        else:
            ops_per_second = 0

        return PerformanceProfile(
            operation=operation,
            avg_latency_ms=np.mean(latencies),
            p95_latency_ms=sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else 0,
            p99_latency_ms=sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else 0,
            max_latency_ms=max(latencies),
            min_latency_ms=min(latencies),
            total_operations=len(metrics),
            operations_per_second=ops_per_second,
            memory_efficiency=self._calculate_memory_efficiency(metrics),
            cpu_efficiency=self._calculate_cpu_efficiency(metrics)
        )

    def _calculate_memory_efficiency(self, metrics: List[LatencyMetric]) -> float:
        """Calculate memory efficiency score."""
        if not metrics:
            return 0.0

        avg_memory = np.mean([m.memory_usage_mb for m in metrics])
        avg_latency = np.mean([m.latency_ms for m in metrics])

        # Lower memory usage and latency = higher efficiency
        memory_score = max(0, (100 - avg_memory) / 100)
        latency_score = max(0, (50 - avg_latency) / 50)

        return (memory_score + latency_score) / 2

    def _calculate_cpu_efficiency(self, metrics: List[LatencyMetric]) -> float:
        """Calculate CPU efficiency score."""
        if not metrics:
            return 0.0

        avg_cpu = np.mean([m.cpu_usage_percent for m in metrics])
        avg_latency = np.mean([m.latency_ms for m in metrics])

        # Lower CPU usage and latency = higher efficiency
        cpu_score = max(0, (100 - avg_cpu) / 100)
        latency_score = max(0, (50 - avg_latency) / 50)

        return (cpu_score + latency_score) / 2


class CacheManager:
    """High-performance cache manager with multiple eviction strategies."""

    def __init__(self, max_size_mb: int = 100):
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: Dict[str, Tuple[Any, float, int]] = {}  # key -> (value, timestamp, size)
        self.access_times: Dict[str, float] = {}
        self.current_size_bytes = 0
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                value, timestamp, size = self.cache[key]
                current_time = time.time()

                # Check if item is expired (default TTL: 5 minutes)
                if current_time - timestamp < 300:
                    self.access_times[key] = current_time
                    self.hits += 1
                    return value
                else:
                    # Remove expired item
                    self._remove_item(key)
                    self.misses += 1
                    return None
            else:
                self.misses += 1
                return None

    def put(self, key: str, value: Any, ttl: int = 300):
        """Put value in cache with TTL."""
        # Calculate size
        try:
            size = len(pickle.dumps(value))
        except:
            size = 1024  # Default size estimate

        with self.lock:
            current_time = time.time()

            # Remove existing item if present
            if key in self.cache:
                self._remove_item(key)

            # Evict items if necessary
            while self.current_size_bytes + size > self.max_size_bytes:
                if not self._evict_lru():
                    break  # Can't evict more items

            # Add new item
            self.cache[key] = (value, current_time, size)
            self.access_times[key] = current_time
            self.current_size_bytes += size

    def _remove_item(self, key: str):
        """Remove item from cache."""
        if key in self.cache:
            _, _, size = self.cache.pop(key)
            self.access_times.pop(key, None)
            self.current_size_bytes -= size

    def _evict_lru(self) -> bool:
        """Evict least recently used item."""
        if not self.access_times:
            return False

        # Find LRU key
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_item(lru_key)
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'cache_size_mb': self.current_size_bytes / 1024 / 1024,
                'max_size_mb': self.max_size_mb,
                'items_count': len(self.cache)
            }


class ParallelProcessor:
    """High-performance parallel processing system."""

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1))

    async def parallel_process(self, func: Callable, items: List[Any], use_processes: bool = False) -> List[Any]:
        """Process items in parallel."""
        if use_processes:
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(self.process_executor, func, item)
                for item in items
            ]
        else:
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(self.thread_executor, func, item)
                for item in items
            ]

        results = await asyncio.gather(*futures, return_exceptions=True)
        return results

    def shutdown(self):
        """Shutdown executors."""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)


class PerformanceOptimizer:
    """
    Comprehensive performance optimization system for HFT applications.

    Provides caching, memory management, parallel processing, and
    latency optimization to achieve sub-50ms performance targets.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.latency_tracker = LatencyTracker(
            max_samples=self.config.get('max_latency_samples', 10000)
        )
        self.cache_manager = CacheManager(
            max_size_mb=self.config.get('cache_size_mb', 100)
        )
        self.parallel_processor = ParallelProcessor(
            max_workers=self.config.get('max_workers')
        )

        # Memory pools for common objects
        self.memory_pools: Dict[type, MemoryPool] = {}

        # Performance monitoring
        self.optimization_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_operations': 0,
            'optimization_suggestions': []
        }

        # Optimization thresholds
        self.latency_threshold_ms = self.config.get('latency_threshold_ms', 50.0)
        self.memory_threshold_mb = self.config.get('memory_threshold_mb', 1000)

    def create_memory_pool(self, object_class: type, initial_size: int = 1000, max_size: int = 10000) -> MemoryPool:
        """Create a memory pool for object allocation."""
        pool = MemoryPool(object_class, initial_size, max_size)
        self.memory_pools[object_class] = pool
        return pool

    def get_memory_pool(self, object_class: type) -> Optional[MemoryPool]:
        """Get memory pool for object class."""
        return self.memory_pools.get(object_class)

    def timed_operation(self, operation_name: str):
        """Decorator for timing operations."""
        def decorator(func):
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = self.latency_tracker.start_operation(operation_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.latency_tracker.end_operation(operation_name, start_time)

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = self.latency_tracker.start_operation(operation_name)
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    self.latency_tracker.end_operation(operation_name, start_time)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    def cached_operation(self, ttl: int = 300, key_func: Callable = None):
        """Decorator for caching operation results."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"

                # Try to get from cache
                cached_result = self.cache_manager.get(cache_key)
                if cached_result is not None:
                    self.optimization_stats['cache_hits'] += 1
                    return cached_result

                # Execute function
                self.optimization_stats['cache_misses'] += 1
                result = func(*args, **kwargs)

                # Cache result
                self.cache_manager.put(cache_key, result, ttl)
                return result

            return wrapper
        return decorator

    async def parallel_map(self, func: Callable, items: List[Any], use_processes: bool = False) -> List[Any]:
        """Apply function to items in parallel."""
        self.optimization_stats['parallel_operations'] += 1
        return await self.parallel_processor.parallel_process(func, items, use_processes)

    def optimize_memory_usage(self):
        """Optimize memory usage."""
        # Force garbage collection
        gc.collect()

        # Clear cache if memory usage is high
        process = psutil.Process()
        memory_usage_mb = process.memory_info().rss / 1024 / 1024

        if memory_usage_mb > self.memory_threshold_mb:
            # Clear 25% of cache
            self.cache_manager.max_size_bytes = int(self.cache_manager.max_size_bytes * 0.75)
            self.logger.warning(f"High memory usage ({memory_usage_mb:.1f}MB), reduced cache size")

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance and suggest optimizations."""
        analysis = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'profiles': {},
            'cache_stats': self.cache_manager.get_stats(),
            'optimization_stats': self.optimization_stats.copy(),
            'suggestions': []
        }

        # Analyze operation profiles
        for operation_name in list(self.latency_tracker.metrics.keys())[:10]:  # Top 10 operations
            profile = self.latency_tracker.get_profile(operation_name)
            if profile:
                analysis['profiles'][operation_name] = {
                    'avg_latency_ms': profile.avg_latency_ms,
                    'p95_latency_ms': profile.p95_latency_ms,
                    'ops_per_second': profile.operations_per_second,
                    'meets_target': profile.p95_latency_ms < self.latency_threshold_ms
                }

                # Generate suggestions
                if profile.p95_latency_ms > self.latency_threshold_ms:
                    analysis['suggestions'].append(
                        f"Operation '{operation_name}' exceeds latency target "
                        f"({profile.p95_latency_ms:.1f}ms > {self.latency_threshold_ms}ms)"
                    )

                if profile.operations_per_second < 100:
                    analysis['suggestions'].append(
                        f"Operation '{operation_name}' has low throughput "
                        f"({profile.operations_per_second:.1f} ops/sec)"
                    )

        # Cache optimization suggestions
        cache_stats = analysis['cache_stats']
        if cache_stats['hit_rate'] < 0.7:
            analysis['suggestions'].append(
                f"Low cache hit rate ({cache_stats['hit_rate']:.1%}), consider cache tuning"
            )

        return analysis

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics."""
        process = psutil.Process()

        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'thread_count': process.num_threads(),
            'open_files': process.num_fds(),
            'cache_hit_rate': self.cache_manager.hits / max(1, self.cache_manager.hits + self.cache_manager.misses),
            'active_operations': len(self.latency_tracker.metrics),
            'optimization_score': self._calculate_optimization_score()
        }

    def _calculate_optimization_score(self) -> float:
        """Calculate overall optimization score (0-100)."""
        score = 50.0  # Base score

        # Memory efficiency
        process = psutil.Process()
        memory_usage_mb = process.memory_info().rss / 1024 / 1024
        if memory_usage_mb < 500:
            score += 20
        elif memory_usage_mb < 1000:
            score += 10

        # Cache efficiency
        cache_hit_rate = self.cache_manager.hits / max(1, self.cache_manager.hits + self.cache_manager.misses)
        score += cache_hit_rate * 20

        # Latency efficiency
        total_profiles = 0
        fast_operations = 0
        for operation_name in self.latency_tracker.metrics.keys():
            profile = self.latency_tracker.get_profile(operation_name)
            if profile and profile.p95_latency_ms < self.latency_threshold_ms:
                fast_operations += 1
                total_profiles += 1

        if total_profiles > 0:
            score += (fast_operations / total_profiles) * 10

        return min(100.0, max(0.0, score))

    def auto_optimize(self):
        """Automatically optimize system performance."""
        # Memory optimization
        self.optimize_memory_usage()

        # Adjust cache size based on hit rate
        cache_stats = self.cache_manager.get_stats()
        if cache_stats['hit_rate'] < 0.5:
            # Increase cache size
            self.cache_manager.max_size_bytes = min(
                self.cache_manager.max_size_bytes * 1.2,
                500 * 1024 * 1024  # 500MB max
            )
        elif cache_stats['hit_rate'] > 0.9:
            # Decrease cache size to save memory
            self.cache_manager.max_size_bytes = max(
                self.cache_manager.max_size_bytes * 0.8,
                50 * 1024 * 1024  # 50MB min
            )

    def shutdown(self):
        """Shutdown performance optimizer."""
        self.parallel_processor.shutdown()
        self.logger.info("Performance optimizer shutdown complete")


# Global performance optimizer instance
_performance_optimizer = None

def get_performance_optimizer(config: Dict = None) -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer(config)
    return _performance_optimizer

# Convenience decorators
def timed(operation_name: str):
    """Convenience decorator for timing operations."""
    optimizer = get_performance_optimizer()
    return optimizer.timed_operation(operation_name)

def cached(ttl: int = 300, key_func: Callable = None):
    """Convenience decorator for caching operations."""
    optimizer = get_performance_optimizer()
    return optimizer.cached_operation(ttl, key_func)

async def parallel_map(func: Callable, items: List[Any], use_processes: bool = False) -> List[Any]:
    """Convenience function for parallel processing."""
    optimizer = get_performance_optimizer()
    return await optimizer.parallel_map(func, items, use_processes)