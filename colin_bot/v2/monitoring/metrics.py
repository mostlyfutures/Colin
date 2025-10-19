"""
Metrics Collection for Colin Trading Bot v2.0

This module implements metrics collection system.
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


class MetricsCollector:
    """Metrics collection system."""

    def __init__(self):
        self.metrics: Dict[str, List[MetricPoint]] = {}
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}

    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment counter metric."""
        key = self._make_key(name, labels)
        self.counters[key] = self.counters.get(key, 0) + value

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set gauge metric."""
        key = self._make_key(name, labels)
        self.gauges[key] = value

    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record histogram metric."""
        key = self._make_key(name, labels)
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)

        # Keep only last 1000 values
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]

    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create metric key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"

    def get_metric_summary(self, name: str, labels: Dict[str, str] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        key = self._make_key(name, labels)

        if key in self.counters:
            return {"type": "counter", "value": self.counters[key]}
        elif key in self.gauges:
            return {"type": "gauge", "value": self.gauges[key]}
        elif key in self.histograms:
            values = self.histograms[key]
            if values:
                return {
                    "type": "histogram",
                    "count": len(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        return {"type": "unknown", "value": None}

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        return {
            "counters": self.counters.copy(),
            "gauges": self.gauges.copy(),
            "histograms": {
                key: self.get_metric_summary(key.split("[")[0],
                    self._parse_labels(key) if "[" in key else None)
                for key in self.histograms.keys()
            }
        }

    def _parse_labels(self, key: str) -> Dict[str, str]:
        """Parse labels from metric key."""
        if "[" not in key or not key.endswith("]"):
            return {}

        label_part = key[key.index("[") + 1:key.index("]")]
        labels = {}
        for pair in label_part.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                labels[k] = v
        return labels