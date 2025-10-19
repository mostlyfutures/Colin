"""
Real-Time Monitoring Dashboard

Web-based monitoring dashboard for real-time HFT system analytics,
performance metrics, and system health monitoring.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import logging
import threading
import weakref

# Web framework imports
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    WEB_FRAMEWORK_AVAILABLE = True
except ImportError:
    WEB_FRAMEWORK_AVAILABLE = False

# Data visualization imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.utils import PlotlyJSONEncoder
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..utils.data_structures import TradingSignal, OrderBook
from ..performance.optimization import get_performance_optimizer
from ..signal_processing.signal_integration_manager import SignalIntegrationManager
from ..risk_management.circuit_breaker import CircuitBreakerSystem
from ..risk_management.position_sizing import DynamicPositionSizer


@dataclass
class DashboardMetric:
    """Dashboard metric data structure."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str
    status: str  # good, warning, critical
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    history: deque = None

    def __post_init__(self):
        if self.history is None:
            self.history = deque(maxlen=1000)

    def update(self, value: float, timestamp: datetime = None):
        """Update metric value."""
        self.value = value
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.history.append((self.timestamp, value))

        # Update status based on thresholds
        if self.threshold_min is not None and value < self.threshold_min:
            self.status = "critical"
        elif self.threshold_max is not None and value > self.threshold_max:
            self.status = "critical"
        elif (self.threshold_min and value < self.threshold_min * 1.1) or \
             (self.threshold_max and value > self.threshold_max * 0.9):
            self.status = "warning"
        else:
            self.status = "good"


@dataclass
class AlertData:
    """Alert data structure."""
    id: str
    level: str  # info, warning, error, critical
    title: str
    message: str
    timestamp: datetime
    source: str
    acknowledged: bool = False
    resolved: bool = False


class MetricsCollector:
    """Collects and manages metrics for the dashboard."""

    def __init__(self):
        self.metrics: Dict[str, DashboardMetric] = {}
        self.alerts: deque = deque(maxlen=1000)
        self.subscribers: List[Callable] = []
        self.collection_interval = 1.0  # seconds
        self.is_collecting = False
        self.lock = threading.Lock()

    def register_metric(self, name: str, category: str, unit: str = "",
                       threshold_min: float = None, threshold_max: float = None):
        """Register a new metric."""
        with self.lock:
            metric = DashboardMetric(
                name=name,
                value=0.0,
                unit=unit,
                timestamp=datetime.now(timezone.utc),
                category=category,
                status="good",
                threshold_min=threshold_min,
                threshold_max=threshold_max
            )
            self.metrics[name] = metric

    def update_metric(self, name: str, value: float, timestamp: datetime = None):
        """Update a metric value."""
        with self.lock:
            if name in self.metrics:
                old_status = self.metrics[name].status
                self.metrics[name].update(value, timestamp)

                # Create alert if status changed to critical
                if self.metrics[name].status == "critical" and old_status != "critical":
                    self._create_alert(
                        level="critical",
                        title=f"Critical Threshold: {name}",
                        message=f"Metric {name} exceeded critical threshold: {value} {self.metrics[name].unit}",
                        source="metrics_collector"
                    )

    def get_metric(self, name: str) -> Optional[DashboardMetric]:
        """Get a metric by name."""
        with self.lock:
            return self.metrics.get(name)

    def get_all_metrics(self) -> Dict[str, Dict]:
        """Get all metrics as dictionary."""
        with self.lock:
            return {name: asdict(metric) for name, metric in self.metrics.items()}

    def get_metrics_by_category(self, category: str) -> Dict[str, DashboardMetric]:
        """Get metrics by category."""
        with self.lock:
            return {name: metric for name, metric in self.metrics.items()
                   if metric.category == category}

    def _create_alert(self, level: str, title: str, message: str, source: str):
        """Create a new alert."""
        alert = AlertData(
            id=f"{int(time.time() * 1000)}_{hash(title)}",
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(timezone.utc),
            source=source
        )
        self.alerts.append(alert)
        self._notify_subscribers(alert)

    def _notify_subscribers(self, alert: AlertData):
        """Notify subscribers of new alerts."""
        for callback in self.subscribers:
            try:
                callback(alert)
            except Exception as e:
                logging.error(f"Error notifying alert subscriber: {e}")

    def subscribe_to_alerts(self, callback: Callable):
        """Subscribe to alert notifications."""
        self.subscribers.append(callback)

    def get_recent_alerts(self, hours: int = 24) -> List[AlertData]:
        """Get recent alerts."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp > cutoff_time]

    def start_collection(self, hft_system_components: Dict[str, Any]):
        """Start collecting metrics from HFT system components."""
        self.is_collecting = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(hft_system_components,),
            daemon=True
        )
        self.collection_thread.start()

    def stop_collection(self):
        """Stop collecting metrics."""
        self.is_collecting = False
        if hasattr(self, 'collection_thread'):
            self.collection_thread.join(timeout=5)

    def _collection_loop(self, components: Dict[str, Any]):
        """Main collection loop."""
        while self.is_collecting:
            try:
                self._collect_system_metrics()
                self._collect_component_metrics(components)
                time.sleep(self.collection_interval)
            except Exception as e:
                logging.error(f"Error in metrics collection: {e}")
                time.sleep(5)  # Wait before retrying

    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        import psutil
        import os

        # CPU and memory metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        process = psutil.Process(os.getpid())

        self.update_metric("system_cpu_percent", cpu_percent)
        self.update_metric("system_memory_percent", memory.percent)
        self.update_metric("process_memory_mb", process.memory_info().rss / 1024 / 1024)
        self.update_metric("process_cpu_percent", process.cpu_percent())
        self.update_metric("open_files", process.num_fds())
        self.update_metric("thread_count", process.num_threads())

    def _collect_component_metrics(self, components: Dict[str, Any]):
        """Collect metrics from HFT system components."""
        # Signal integration metrics
        if 'signal_integration' in components:
            sig_int = components['signal_integration']
            if hasattr(sig_int, 'get_component_statistics'):
                stats = asyncio.run(sig_int.get_component_statistics())

                for component, component_stats in stats.items():
                    if isinstance(component_stats, dict):
                        for metric_name, value in component_stats.items():
                            if isinstance(value, (int, float)):
                                full_name = f"{component}_{metric_name}"
                                self.update_metric(full_name, value)

        # Position sizing metrics
        if 'position_sizer' in components:
            pos_sizer = components['position_sizer']
            if hasattr(pos_sizer, 'get_sizing_statistics'):
                stats = pos_sizer.get_sizing_statistics()

                for metric_name, value in stats.items():
                    if isinstance(value, (int, float)):
                        full_name = f"position_sizer_{metric_name}"
                        self.update_metric(full_name, value)

        # Circuit breaker metrics
        if 'circuit_breaker' in components:
            cb = components['circuit_breaker']
            if hasattr(cb, 'get_status'):
                status = cb.get_status()

                self.update_metric("circuit_breaker_state",
                                1 if status['state'] == 'closed' else 0)
                self.update_metric("circuit_breaker_stress_level",
                                self._stress_level_to_number(status.get('stress_level', 'normal')))

        # Performance optimizer metrics
        perf_optimizer = get_performance_optimizer()
        real_time_metrics = perf_optimizer.get_real_time_metrics()

        for metric_name, value in real_time_metrics.items():
            if isinstance(value, (int, float)):
                self.update_metric(f"perf_{metric_name}", value)

    def _stress_level_to_number(self, stress_level: str) -> float:
        """Convert stress level to numeric value."""
        stress_map = {
            'normal': 0.0,
            'elevated': 0.25,
            'high': 0.5,
            'severe': 0.75,
            'critical': 1.0
        }
        return stress_map.get(stress_level.lower(), 0.0)


class RealTimeDashboard:
    """
    Real-time monitoring dashboard for HFT systems.

    Provides web-based interface for monitoring system performance,
    trading signals, risk metrics, and system health.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        if not WEB_FRAMEWORK_AVAILABLE:
            raise ImportError("FastAPI and uvicorn are required for the dashboard. Install with: pip install fastapi uvicorn")

        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.app = FastAPI(title="HFT Dashboard", version="1.0.0")
        self.websocket_connections: List[WebSocket] = []
        self.hft_components: Dict[str, Any] = {}

        # Dashboard configuration
        self.host = self.config.get('host', '0.0.0.0')
        self.port = self.config.get('port', 8080)
        self.refresh_interval = self.config.get('refresh_interval', 1.0)

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Serve dashboard home page."""
            return self._generate_dashboard_html()

        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get current metrics."""
            return self.metrics_collector.get_all_metrics()

        @self.app.get("/api/alerts")
        async def get_alerts():
            """Get recent alerts."""
            alerts = self.metrics_collector.get_recent_alerts()
            return [asdict(alert) for alert in alerts]

        @self.app.get("/api/performance")
        async def get_performance():
            """Get performance data."""
            perf_optimizer = get_performance_optimizer()
            return perf_optimizer.get_real_time_metrics()

        @self.app.get("/api/system_health")
        async def get_system_health():
            """Get system health status."""
            return self._calculate_system_health()

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.websocket_connections.append(websocket)

            try:
                while True:
                    # Send real-time updates
                    await self._send_websocket_update(websocket)
                    await asyncio.sleep(self.refresh_interval)
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)

    def _generate_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HFT Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .dashboard {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
        }
        .metric-title {
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .metric-status {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-left: 10px;
        }
        .status-good { background-color: #4CAF50; }
        .status-warning { background-color: #FF9800; }
        .status-critical { background-color: #f44336; }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .alerts-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .alert {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }
        .alert-info { border-left-color: #2196F3; background-color: #E3F2FD; }
        .alert-warning { border-left-color: #FF9800; background-color: #FFF3E0; }
        .alert-error { border-left-color: #f44336; background-color: #FFEBEE; }
        .alert-critical { border-left-color: #9C27B0; background-color: #F3E5F5; }
        .system-status {
            display: flex;
            justify-content: space-between;
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .status-item {
            text-align: center;
        }
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px;
            border-radius: 5px;
            background: #4CAF50;
            color: white;
        }
        .connection-status.disconnected {
            background: #f44336;
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🚀 HFT Trading Dashboard</h1>
            <p>Real-time monitoring for high-frequency trading system</p>
        </div>

        <div class="connection-status" id="connectionStatus">
            🟢 Connected
        </div>

        <div class="system-status" id="systemStatus">
            <div class="status-item">
                <div class="metric-title">System Status</div>
                <div class="metric-value" id="systemHealth">Healthy</div>
            </div>
            <div class="status-item">
                <div class="metric-title">Active Signals</div>
                <div class="metric-value" id="activeSignals">0</div>
            </div>
            <div class="status-item">
                <div class="metric-title">Latency</div>
                <div class="metric-value" id="avgLatency">0ms</div>
            </div>
            <div class="status-item">
                <div class="metric-title">Success Rate</div>
                <div class="metric-value" id="successRate">0%</div>
            </div>
        </div>

        <div class="metrics-grid" id="metricsGrid">
            <!-- Metrics will be populated here -->
        </div>

        <div class="chart-container">
            <h3>System Performance</h3>
            <canvas id="performanceChart" width="400" height="200"></canvas>
        </div>

        <div class="chart-container">
            <h3>Signal Generation Rate</h3>
            <canvas id="signalsChart" width="400" height="200"></canvas>
        </div>

        <div class="alerts-container">
            <h3>Recent Alerts</h3>
            <div id="alertsContainer">
                <!-- Alerts will be populated here -->
            </div>
        </div>
    </div>

    <script>
        let ws;
        let performanceChart;
        let signalsChart;
        let performanceData = [];
        let signalsData = [];

        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;

            ws = new WebSocket(wsUrl);

            ws.onopen = function() {
                document.getElementById('connectionStatus').className = 'connection-status';
                document.getElementById('connectionStatus').innerHTML = '🟢 Connected';
                console.log('WebSocket connected');
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };

            ws.onclose = function() {
                document.getElementById('connectionStatus').className = 'connection-status disconnected';
                document.getElementById('connectionStatus').innerHTML = '🔴 Disconnected';
                console.log('WebSocket disconnected');
                // Try to reconnect
                setTimeout(initWebSocket, 5000);
            };

            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }

        function initCharts() {
            // Performance chart
            const perfCtx = document.getElementById('performanceChart').getContext('2d');
            performanceChart = new Chart(perfCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU %',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }, {
                        label: 'Memory %',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });

            // Signals chart
            const signalsCtx = document.getElementById('signalsChart').getContext('2d');
            signalsChart = new Chart(signalsCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Signals per Minute',
                        data: [],
                        backgroundColor: 'rgba(54, 162, 235, 0.5)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }

        function updateDashboard(data) {
            // Update metrics
            updateMetrics(data.metrics);

            // Update alerts
            updateAlerts(data.alerts);

            // Update system status
            updateSystemStatus(data.systemHealth);

            // Update charts
            updateCharts(data);
        }

        function updateMetrics(metrics) {
            const metricsGrid = document.getElementById('metricsGrid');
            metricsGrid.innerHTML = '';

            const categories = ['System', 'Performance', 'Trading', 'Risk'];

            categories.forEach(category => {
                const categoryMetrics = Object.entries(metrics).filter(([name, metric]) =>
                    metric.category === category
                );

                if (categoryMetrics.length > 0) {
                    const card = document.createElement('div');
                    card.className = 'metric-card';

                    let html = `<h4>${category}</h4>`;
                    categoryMetrics.forEach(([name, metric]) => {
                        const statusClass = `status-${metric.status}`;
                        html += `
                            <div style="margin-bottom: 10px;">
                                <div class="metric-title">${metric.name}</div>
                                <div class="metric-value">
                                    ${metric.value.toFixed(2)} ${metric.unit}
                                    <span class="metric-status ${statusClass}"></span>
                                </div>
                            </div>
                        `;
                    });

                    card.innerHTML = html;
                    metricsGrid.appendChild(card);
                }
            });
        }

        function updateAlerts(alerts) {
            const container = document.getElementById('alertsContainer');
            container.innerHTML = '';

            alerts.slice(0, 10).forEach(alert => {
                const alertDiv = document.createElement('div');
                alertDiv.className = `alert alert-${alert.level}`;
                alertDiv.innerHTML = `
                    <strong>${alert.title}</strong>
                    <div>${alert.message}</div>
                    <small>${new Date(alert.timestamp).toLocaleString()}</small>
                `;
                container.appendChild(alertDiv);
            });
        }

        function updateSystemStatus(health) {
            document.getElementById('systemHealth').textContent = health.status;
            document.getElementById('activeSignals').textContent = health.active_signals;
            document.getElementById('avgLatency').textContent = health.avg_latency + 'ms';
            document.getElementById('successRate').textContent = health.success_rate + '%';
        }

        function updateCharts(data) {
            // Update performance chart
            if (performanceChart && data.performance) {
                const now = new Date().toLocaleTimeString();

                // Keep only last 20 data points
                if (performanceChart.data.labels.length > 20) {
                    performanceChart.data.labels.shift();
                    performanceChart.data.datasets[0].data.shift();
                    performanceChart.data.datasets[1].data.shift();
                }

                performanceChart.data.labels.push(now);
                performanceChart.data.datasets[0].data.push(data.performance.system_cpu_percent || 0);
                performanceChart.data.datasets[1].data.push(data.performance.system_memory_percent || 0);
                performanceChart.update();
            }
        }

        function fetchInitialData() {
            // Fetch initial metrics
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    updateDashboard({ metrics: data, alerts: [], systemHealth: {} });
                });

            // Fetch initial alerts
            fetch('/api/alerts')
                .then(response => response.json())
                .then(alerts => {
                    updateDashboard({ metrics: {}, alerts: alerts, systemHealth: {} });
                });

            // Fetch system health
            fetch('/api/system_health')
                .then(response => response.json())
                .then(health => {
                    updateDashboard({ metrics: {}, alerts: [], systemHealth: health });
                });
        }

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            fetchInitialData();
            initWebSocket();
        });
    </script>
</body>
</html>
        """

    async def _send_websocket_update(self, websocket: WebSocket):
        """Send real-time update via WebSocket."""
        try:
            update_data = {
                metrics: self.metrics_collector.get_all_metrics(),
                alerts: [asdict(alert) for alert in self.metrics_collector.get_recent_alerts(hours=1)],
                systemHealth: self._calculate_system_health(),
                performance: get_performance_optimizer().get_real_time_metrics(),
                timestamp: datetime.now(timezone.utc).isoformat()
            }

            await websocket.send_text(json.dumps(update_data, default=str))
        except Exception as e:
            self.logger.error(f"Error sending WebSocket update: {e}")

    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health."""
        health_metrics = {
            'status': 'healthy',
            'active_signals': 0,
            'avg_latency': 0.0,
            'success_rate': 0.0,
            'issues': []
        }

        # Check circuit breaker status
        if 'circuit_breaker' in self.hft_components:
            cb_status = self.hft_components['circuit_breaker'].get_status()
            if cb_status['state'] != 'closed':
                health_metrics['status'] = 'degraded'
                health_metrics['issues'].append(f"Circuit breaker is {cb_status['state']}")

        # Check performance metrics
        perf_optimizer = get_performance_optimizer()
        perf_metrics = perf_optimizer.get_real_time_metrics()

        if perf_metrics.get('memory_usage_mb', 0) > 1000:
            health_metrics['status'] = 'warning'
            health_metrics['issues'].append("High memory usage")

        if perf_metrics.get('optimization_score', 100) < 50:
            health_metrics['status'] = 'degraded'
            health_metrics['issues'].append("Low optimization score")

        # Calculate active signals
        if 'signal_integration' in self.hft_components:
            # This would need to be implemented in the signal integration manager
            health_metrics['active_signals'] = 0  # Placeholder

        return health_metrics

    def register_hft_components(self, **components):
        """Register HFT system components for monitoring."""
        self.hft_components.update(components)

        # Register metrics for each component
        self._register_component_metrics()

    def _register_component_metrics(self):
        """Register metrics for all components."""
        # System metrics
        self.metrics_collector.register_metric("system_cpu_percent", "System", "%", threshold_max=80)
        self.metrics_collector.register_metric("system_memory_percent", "System", "%", threshold_max=85)
        self.metrics_collector.register_metric("process_memory_mb", "System", "MB", threshold_max=1000)
        self.metrics_collector.register_metric("process_cpu_percent", "System", "%", threshold_max=90)

        # Performance metrics
        self.metrics_collector.register_metric("perf_optimization_score", "Performance", "%", threshold_min=70)
        self.metrics_collector.register_metric("perf_cache_hit_rate", "Performance", "%", threshold_min=70)

        # Trading metrics
        self.metrics_collector.register_metric("signals_generated_last_hour", "Trading", "count")
        self.metrics_collector.register_metric("avg_signal_latency_ms", "Trading", "ms", threshold_max=50)

        # Risk metrics
        self.metrics_collector.register_metric("circuit_breaker_state", "Risk", "binary")
        self.metrics_collector.register_metric("circuit_breaker_stress_level", "Risk", "normalized", threshold_max=0.7)

    async def start(self, host: str = None, port: int = None):
        """Start the dashboard server."""
        host = host or self.host
        port = port or self.port

        # Start metrics collection
        self.metrics_collector.start_collection(self.hft_components)

        # Start FastAPI server
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)

        self.logger.info(f"Starting dashboard on http://{host}:{port}")
        await server.serve()

    def stop(self):
        """Stop the dashboard."""
        self.metrics_collector.stop_collection()


# Dashboard factory function
def create_dashboard(config: Dict = None) -> RealTimeDashboard:
    """Create and configure HFT dashboard."""
    return RealTimeDashboard(config)


# Convenience function for quick dashboard setup
async def run_dashboard(signal_integration: SignalIntegrationManager = None,
                       position_sizer: DynamicPositionSizer = None,
                       circuit_breaker: CircuitBreakerSystem = None,
                       config: Dict = None):
    """Run dashboard with HFT components."""
    dashboard = create_dashboard(config)

    # Register components
    components = {}
    if signal_integration:
        components['signal_integration'] = signal_integration
    if position_sizer:
        components['position_sizer'] = position_sizer
    if circuit_breaker:
        components['circuit_breaker'] = circuit_breaker

    dashboard.register_hft_components(**components)

    # Start dashboard
    await dashboard.start()