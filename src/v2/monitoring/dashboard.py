"""
Monitoring Dashboard for Colin Trading Bot v2.0

This module implements monitoring dashboard for real-time visualization.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from loguru import logger


class MonitoringDashboard:
    """Monitoring dashboard system."""

    def __init__(self):
        self.dashboard_data = {}
        self.last_update = None

    def update_dashboard_data(self, data: Dict[str, Any]):
        """Update dashboard data."""
        self.dashboard_data = data
        self.last_update = datetime.now()

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return {
            "data": self.dashboard_data,
            "last_updated": self.last_update.isoformat() if self.last_update else None,
            "timestamp": datetime.now().isoformat()
        }

    def generate_html_dashboard(self) -> str:
        """Generate simple HTML dashboard."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Colin Trading Bot v2.0 Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ background-color: #f9f9f9; margin: 10px 0; padding: 15px; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
                .status-good {{ color: #27ae60; }}
                .status-warning {{ color: #f39c12; }}
                .status-critical {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Colin Trading Bot v2.0 Dashboard</h1>
                <p>Last updated: {self.last_update or 'Never'}</p>
            </div>

            <div class="metric">
                <div class="metric-value status-good">
                    {self.dashboard_data.get('signals_generated', 0)}
                </div>
                <div class="metric-label">Signals Generated</div>
            </div>

            <div class="metric">
                <div class="metric-value status-good">
                    {self.dashboard_data.get('executions_completed', 0)}
                </div>
                <div class="metric-label">Executions Completed</div>
            </div>

            <div class="metric">
                <div class="metric-value">
                    ${self.dashboard_data.get('portfolio_value', 0):,.2f}
                </div>
                <div class="metric-label">Portfolio Value</div>
            </div>

            <div class="metric">
                <div class="metric-value">
                    {self.dashboard_data.get('active_positions', 0)}
                </div>
                <div class="metric-label">Active Positions</div>
            </div>

            <script>
                // Auto-refresh every 5 seconds
                setTimeout(function() {{
                    location.reload();
                }}, 5000);
            </script>
        </body>
        </html>
        """