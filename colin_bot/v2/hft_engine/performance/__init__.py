"""
HFT Performance Optimization Layer

Advanced performance optimization tools for achieving sub-50ms latency
in high-frequency trading applications.
"""

from .optimization import (
    PerformanceOptimizer, LatencyTracker, MemoryPool, CircularBuffer,
    CacheManager, ParallelProcessor, LatencyMetric, PerformanceProfile,
    timed, cached, parallel_map, get_performance_optimizer
)

__all__ = [
    "PerformanceOptimizer",
    "LatencyTracker",
    "MemoryPool",
    "CircularBuffer",
    "CacheManager",
    "ParallelProcessor",
    "LatencyMetric",
    "PerformanceProfile",
    "timed",
    "cached",
    "parallel_map",
    "get_performance_optimizer"
]