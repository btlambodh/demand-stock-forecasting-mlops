#!/usr/bin/env python3
"""
Performance Monitoring System for Chinese Produce Market Forecasting
Tracks model performance, system health, and provides real-time monitoring

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import argparse
import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import threading
import warnings

import yaml
import pandas as pd
import numpy as np
import boto3
import psutil
from prometheus_client import Gauge, Counter, Histogram, start_http_server, generate_latest
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import requests

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy percentage', ['model_name', 'metric_type'])
MODEL_LATENCY = Histogram('model_prediction_latency_seconds', 'Model prediction latency', ['model_name'])
SYSTEM_CPU = Gauge('system_cpu_percent', 'System CPU usage percentage')
SYSTEM_MEMORY = Gauge('system_memory_percent', 'System memory usage percentage')
SYSTEM_DISK = Gauge('system_disk_percent', 'System disk usage percentage')
API_REQUESTS_TOTAL = Counter('api_requests_total', 'Total API requests', ['endpoint', 'status'])
API_ERRORS_TOTAL = Counter('api_errors_total', 'Total API errors', ['error_type'])
DATA_QUALITY_SCORE = Gauge('data_quality_score', 'Data quality score percentage')
DRIFT_SCORE = Gauge('drift_score', 'Data drift score', ['feature_name'])
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made', ['model_name'])


class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self, config_path: str):
        """Initialize performance monitor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.aws_config = self.config['aws']
        self.monitoring_config = self.config['monitoring']
        
        # Initialize AWS clients
        self.cloudwatch = boto3.client('cloudwatch', region_name=self.aws_config['region'])
        self.s3_client = boto3.client('s3', region_name=self.aws_config['region'])
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metrics_history = {}
        self.alert_history = []
        
        # Performance thresholds
        self.thresholds = self.monitoring_config.get('performance', {})
        
        # Dashboard app
        self.dash_app = None
        
        logger.info("PerformanceMonitor initialized successfully")

    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return
        
        logger.info(f"Starting performance monitoring (interval: {interval_seconds}s)")
        self.is_monitoring = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        # Start Prometheus metrics server
        try:
            start_http_server(8002)  # Different port from API
            logger.info("Prometheus metrics server started on port 8002")
        except Exception as e:
            logger.warning(f"Could not start Prometheus server: {e}")

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        logger.info("Stopping performance monitoring")
        self.is_monitoring = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        logger.info("Performance monitoring loop started")
        
        while self.is_monitoring:
            try:
                # Collect all metrics
                self.collect_system_metrics()
                self.collect_model_metrics()
                self.collect_api_metrics()
                self.collect_data_quality_metrics()
                
                # Check thresholds and send alerts
                self.check_alert_conditions()
                
                # Sleep until next collection
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
        
        logger.info("Performance monitoring loop stopped")

    def collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_CPU.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            SYSTEM_MEMORY.set(memory_percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            SYSTEM_DISK.set(disk_percent)
            
            # Store in history
            timestamp = datetime.now()
            if 'system' not in self.metrics_history:
                self.metrics_history['system'] = []
            
            self.metrics_history['system'].append({
                'timestamp': timestamp.isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_free_gb': disk.free / (1024**3)
            })
            
            # Keep only last 1000 entries
            if len(self.metrics_history['system']) > 1000:
                self.metrics_history['system'] = self.metrics_history['system'][-1000:]
            
            # Send to CloudWatch
            self._send_cloudwatch_metrics('System', [
                {'MetricName': 'CPUUtilization', 'Value': cpu_percent, 'Unit': 'Percent'},
                {'MetricName': 'MemoryUtilization', 'Value': memory_percent, 'Unit': 'Percent'},
                {'MetricName': 'DiskUtilization', 'Value': disk_percent, 'Unit': 'Percent'}
            ])
            
            logger.debug(f"System metrics - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, Disk: {disk_percent:.1f}%")
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def collect_model_metrics(self):
        """Collect model performance metrics"""
        try:
            # Try to get metrics from the API or model files
            model_metrics = self._fetch_model_performance()
            
            for model_name, metrics in model_metrics.items():
                # Update Prometheus metrics
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        MODEL_ACCURACY.labels(model_name=model_name, metric_type=metric_name).set(value)
                
                # Store in history
                if 'models' not in self.metrics_history:
                    self.metrics_history['models'] = {}
                
                if model_name not in self.metrics_history['models']:
                    self.metrics_history['models'][model_name] = []
                
                entry = {
                    'timestamp': datetime.now().isoformat(),
                    **metrics
                }
                self.metrics_history['models'][model_name].append(entry)
                
                # Keep only last 500 entries per model
                if len(self.metrics_history['models'][model_name]) > 500:
                    self.metrics_history['models'][model_name] = self.metrics_history['models'][model_name][-500:]
            
            # Send to CloudWatch
            cloudwatch_metrics = []
            for model_name, metrics in model_metrics.items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        cloudwatch_metrics.append({
                            'MetricName': f'Model_{metric_name}',
                            'Value': value,
                            'Unit': 'Percent' if 'mape' in metric_name.lower() else 'None',
                            'Dimensions': [{'Name': 'ModelName', 'Value': model_name}]
                        })
            
            if cloudwatch_metrics:
                self._send_cloudwatch_metrics('Models', cloudwatch_metrics)
            
            logger.debug(f"Collected metrics for {len(model_metrics)} models")
            
        except Exception as e:
            logger.error(f"Error collecting model metrics: {e}")

    def _fetch_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Fetch model performance from various sources"""
        model_metrics = {}
        
        try:
            # Try to load from evaluation file
            eval_file = "models/evaluation.json"
            if os.path.exists(eval_file):
                with open(eval_file, 'r') as f:
                    evaluation_data = json.load(f)
                
                for model_name, metrics in evaluation_data.items():
                    model_metrics[model_name] = {
                        'val_mape': metrics.get('val_mape', 0),
                        'val_rmse': metrics.get('val_rmse', 0),
                        'val_r2': metrics.get('val_r2', 0),
                        'test_mape': metrics.get('test_mape', 0),
                        'test_rmse': metrics.get('test_rmse', 0),
                        'test_r2': metrics.get('test_r2', 0)
                    }
            
            # Try to fetch from API health endpoint
            try:
                response = requests.get('http://localhost:8000/health', timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    models_loaded = health_data.get('models_loaded', 0)
                    
                    # Add API health metrics
                    model_metrics['api_health'] = {
                        'models_loaded': models_loaded,
                        'uptime_seconds': health_data.get('uptime_seconds', 0),
                        'status_healthy': 1 if health_data.get('status') == 'healthy' else 0
                    }
            except:
                pass  # API might not be running
            
        except Exception as e:
            logger.debug(f"Could not fetch all model metrics: {e}")
        
        return model_metrics

    def collect_api_metrics(self):
        """Collect API performance metrics"""
        try:
            # Try to get metrics from API metrics endpoint
            api_metrics = self._fetch_api_metrics()
            
            # Store in history
            if 'api' not in self.metrics_history:
                self.metrics_history['api'] = []
            
            timestamp = datetime.now()
            entry = {
                'timestamp': timestamp.isoformat(),
                **api_metrics
            }
            self.metrics_history['api'].append(entry)
            
            # Keep only last 1000 entries
            if len(self.metrics_history['api']) > 1000:
                self.metrics_history['api'] = self.metrics_history['api'][-1000:]
            
            # Send to CloudWatch
            cloudwatch_metrics = []
            for metric_name, value in api_metrics.items():
                if isinstance(value, (int, float)):
                    cloudwatch_metrics.append({
                        'MetricName': f'API_{metric_name}',
                        'Value': value,
                        'Unit': 'Count' if 'requests' in metric_name else 'Seconds'
                    })
            
            if cloudwatch_metrics:
                self._send_cloudwatch_metrics('API', cloudwatch_metrics)
            
            logger.debug(f"Collected API metrics: {len(api_metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Error collecting API metrics: {e}")

    def _fetch_api_metrics(self) -> Dict[str, float]:
        """Fetch API metrics from Prometheus endpoint"""
        api_metrics = {}
        
        try:
            # Try to get metrics from Prometheus endpoint
            response = requests.get('http://localhost:8000/metrics', timeout=5)
            if response.status_code == 200:
                metrics_text = response.text
                
                # Parse basic metrics (simplified parsing)
                lines = metrics_text.split('\n')
                for line in lines:
                    if 'api_requests_total' in line and not line.startswith('#'):
                        # Extract value
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                value = float(parts[-1])
                                api_metrics['total_requests'] = api_metrics.get('total_requests', 0) + value
                            except:
                                pass
                    
                    elif 'prediction_latency_seconds' in line and not line.startswith('#'):
                        # Extract latency metrics
                        if '_sum' in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    api_metrics['total_latency'] = float(parts[-1])
                                except:
                                    pass
                        elif '_count' in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    api_metrics['prediction_count'] = float(parts[-1])
                                except:
                                    pass
            
            # Calculate derived metrics
            if 'total_latency' in api_metrics and 'prediction_count' in api_metrics and api_metrics['prediction_count'] > 0:
                api_metrics['average_latency'] = api_metrics['total_latency'] / api_metrics['prediction_count']
            
            # Add timestamp-based metrics
            api_metrics['monitoring_timestamp'] = time.time()
            
        except Exception as e:
            logger.debug(f"Could not fetch API metrics: {e}")
            # Provide default values
            api_metrics = {
                'total_requests': 0,
                'prediction_count': 0,
                'average_latency': 0,
                'monitoring_timestamp': time.time()
            }
        
        return api_metrics

    def collect_data_quality_metrics(self):
        """Collect data quality metrics"""
        try:
            # Calculate data quality from recent predictions or validation data
            quality_score = self._calculate_data_quality()
            
            DATA_QUALITY_SCORE.set(quality_score)
            
            # Store in history
            if 'data_quality' not in self.metrics_history:
                self.metrics_history['data_quality'] = []
            
            entry = {
                'timestamp': datetime.now().isoformat(),
                'quality_score': quality_score,
                'data_freshness': self._check_data_freshness(),
                'validation_status': self._check_validation_status()
            }
            self.metrics_history['data_quality'].append(entry)
            
            # Keep only last 500 entries
            if len(self.metrics_history['data_quality']) > 500:
                self.metrics_history['data_quality'] = self.metrics_history['data_quality'][-500:]
            
            # Send to CloudWatch
            self._send_cloudwatch_metrics('DataQuality', [
                {'MetricName': 'QualityScore', 'Value': quality_score, 'Unit': 'Percent'},
                {'MetricName': 'DataFreshness', 'Value': entry['data_freshness'], 'Unit': 'Hours'}
            ])
            
            logger.debug(f"Data quality score: {quality_score:.1f}%")
            
        except Exception as e:
            logger.error(f"Error collecting data quality metrics: {e}")

    def _calculate_data_quality(self) -> float:
        """Calculate overall data quality score"""
        try:
            # Load recent validation data if available
            validation_file = "data/validation/validation_report.json"
            if os.path.exists(validation_file):
                with open(validation_file, 'r') as f:
                    validation_data = json.load(f)
                
                return validation_data.get('overall_quality_score', 85.0)
            
            # Default quality score
            return 85.0
            
        except Exception as e:
            logger.debug(f"Could not calculate data quality: {e}")
            return 80.0

    def _check_data_freshness(self) -> float:
        """Check how fresh the data is (hours since last update)"""
        try:
            # Check timestamp of latest data file
            data_files = [
                "data/processed/train.parquet",
                "data/processed/validation.parquet",
                "data/raw/annex2.csv"
            ]
            
            latest_time = 0
            for file_path in data_files:
                if os.path.exists(file_path):
                    file_time = os.path.getmtime(file_path)
                    latest_time = max(latest_time, file_time)
            
            if latest_time > 0:
                hours_old = (time.time() - latest_time) / 3600
                return hours_old
            
            return 24.0  # Default: assume 24 hours old
            
        except Exception as e:
            logger.debug(f"Could not check data freshness: {e}")
            return 24.0

    def _check_validation_status(self) -> str:
        """Check latest validation status"""
        try:
            validation_file = "data/validation/validation_report.json"
            if os.path.exists(validation_file):
                with open(validation_file, 'r') as f:
                    validation_data = json.load(f)
                
                return validation_data.get('validation_status', 'unknown')
            
            return 'unknown'
            
        except Exception as e:
            logger.debug(f"Could not check validation status: {e}")
            return 'unknown'

    def check_alert_conditions(self):
        """Check if any alert conditions are met"""
        try:
            alerts_triggered = []
            
            # Check system thresholds
            if self.metrics_history.get('system'):
                latest_system = self.metrics_history['system'][-1]
                
                if latest_system['cpu_percent'] > 80:
                    alerts_triggered.append({
                        'type': 'system',
                        'severity': 'warning',
                        'message': f"High CPU usage: {latest_system['cpu_percent']:.1f}%",
                        'metric': 'cpu_percent',
                        'value': latest_system['cpu_percent'],
                        'threshold': 80
                    })
                
                if latest_system['memory_percent'] > 85:
                    alerts_triggered.append({
                        'type': 'system',
                        'severity': 'warning',
                        'message': f"High memory usage: {latest_system['memory_percent']:.1f}%",
                        'metric': 'memory_percent',
                        'value': latest_system['memory_percent'],
                        'threshold': 85
                    })
                
                if latest_system['disk_percent'] > 90:
                    alerts_triggered.append({
                        'type': 'system',
                        'severity': 'critical',
                        'message': f"High disk usage: {latest_system['disk_percent']:.1f}%",
                        'metric': 'disk_percent',
                        'value': latest_system['disk_percent'],
                        'threshold': 90
                    })
            
            # Check model performance thresholds
            if self.metrics_history.get('models'):
                for model_name, model_history in self.metrics_history['models'].items():
                    if model_history:
                        latest_model = model_history[-1]
                        
                        # Check MAPE threshold
                        val_mape = latest_model.get('val_mape', 0)
                        if val_mape > 20:  # 20% MAPE threshold
                            alerts_triggered.append({
                                'type': 'model_performance',
                                'severity': 'warning',
                                'message': f"High MAPE for {model_name}: {val_mape:.2f}%",
                                'model': model_name,
                                'metric': 'val_mape',
                                'value': val_mape,
                                'threshold': 20
                            })
            
            # Check data quality thresholds
            if self.metrics_history.get('data_quality'):
                latest_quality = self.metrics_history['data_quality'][-1]
                
                if latest_quality['quality_score'] < 70:
                    alerts_triggered.append({
                        'type': 'data_quality',
                        'severity': 'warning',
                        'message': f"Low data quality score: {latest_quality['quality_score']:.1f}%",
                        'metric': 'quality_score',
                        'value': latest_quality['quality_score'],
                        'threshold': 70
                    })
                
                if latest_quality['data_freshness'] > 48:  # 48 hours
                    alerts_triggered.append({
                        'type': 'data_freshness',
                        'severity': 'warning',
                        'message': f"Stale data: {latest_quality['data_freshness']:.1f} hours old",
                        'metric': 'data_freshness',
                        'value': latest_quality['data_freshness'],
                        'threshold': 48
                    })
            
            # Process alerts
            for alert in alerts_triggered:
                self._process_alert(alert)
            
            if alerts_triggered:
                logger.warning(f"Triggered {len(alerts_triggered)} alerts")
            
        except Exception as e:
            logger.error(f"Error checking alert conditions: {e}")

    def _process_alert(self, alert: Dict[str, Any]):
        """Process and send an alert"""
        try:
            # Add timestamp
            alert['timestamp'] = datetime.now().isoformat()
            
            # Add to alert history
            self.alert_history.append(alert)
            
            # Keep only last 100 alerts
            if len(self.alert_history) > 100:
                self.alert_history = self.alert_history[-100:]
            
            # Send to CloudWatch
            self.cloudwatch.put_metric_data(
                Namespace='ChineseProduceForecast/Alerts',
                MetricData=[{
                    'MetricName': 'AlertTriggered',
                    'Value': 1.0,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'AlertType', 'Value': alert['type']},
                        {'Name': 'Severity', 'Value': alert['severity']}
                    ]
                }]
            )
            
            # Log alert
            logger.warning(f"ALERT [{alert['severity'].upper()}] {alert['type']}: {alert['message']}")
            
            # TODO: Send to SNS, email, Slack, etc.
            
        except Exception as e:
            logger.error(f"Error processing alert: {e}")

    def _send_cloudwatch_metrics(self, namespace_suffix: str, metrics: List[Dict]):
        """Send metrics to CloudWatch"""
        try:
            if not metrics:
                return
            
            namespace = f"{self.aws_config['cloudwatch']['metrics_namespace']}/{namespace_suffix}"
            
            # CloudWatch accepts max 20 metrics per request
            for i in range(0, len(metrics), 20):
                batch = metrics[i:i+20]
                
                self.cloudwatch.put_metric_data(
                    Namespace=namespace,
                    MetricData=batch
                )
            
            logger.debug(f"Sent {len(metrics)} metrics to CloudWatch namespace: {namespace}")
            
        except Exception as e:
            logger.error(f"Error sending CloudWatch metrics: {e}")

    def create_dashboard(self, port: int = 8050) -> str:
        """Create interactive monitoring dashboard"""
        logger.info(f"Creating monitoring dashboard on port {port}")
        
        # Initialize Dash app
        self.dash_app = dash.Dash(__name__)
        
        # Define dashboard layout
        self.dash_app.layout = html.Div([
            html.H1("Chinese Produce Forecasting - Monitoring Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Status indicators
            html.Div([
                html.Div([
                    html.H3("System Status", style={'textAlign': 'center'}),
                    html.Div(id='system-status', style={'textAlign': 'center', 'fontSize': 24})
                ], className='four columns'),
                
                html.Div([
                    html.H3("Model Status", style={'textAlign': 'center'}),
                    html.Div(id='model-status', style={'textAlign': 'center', 'fontSize': 24})
                ], className='four columns'),
                
                html.Div([
                    html.H3("Data Quality", style={'textAlign': 'center'}),
                    html.Div(id='data-quality-status', style={'textAlign': 'center', 'fontSize': 24})
                ], className='four columns'),
            ], className='row', style={'marginBottom': 30}),
            
            # Charts
            html.Div([
                dcc.Graph(id='system-metrics-chart'),
            ], style={'marginBottom': 20}),
            
            html.Div([
                dcc.Graph(id='model-performance-chart'),
            ], style={'marginBottom': 20}),
            
            html.Div([
                dcc.Graph(id='api-metrics-chart'),
            ], style={'marginBottom': 20}),
            
            # Recent alerts
            html.Div([
                html.H3("Recent Alerts"),
                html.Div(id='alerts-table')
            ], style={'marginBottom': 20}),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            )
        ])
        
        # Register callbacks
        self._register_dashboard_callbacks()
        
        # Run dashboard
        try:
            self.dash_app.run_server(host='0.0.0.0', port=port, debug=False)
        except Exception as e:
            logger.error(f"Error running dashboard: {e}")
            return f"Error: {e}"
        
        return f"Dashboard running on http://localhost:{port}"

    def _register_dashboard_callbacks(self):
        """Register dashboard callbacks"""
        
        @self.dash_app.callback(
            [Output('system-status', 'children'),
             Output('model-status', 'children'),
             Output('data-quality-status', 'children'),
             Output('system-metrics-chart', 'figure'),
             Output('model-performance-chart', 'figure'),
             Output('api-metrics-chart', 'figure'),
             Output('alerts-table', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            try:
                # Status indicators
                system_status = self._get_system_status_indicator()
                model_status = self._get_model_status_indicator()
                data_quality_status = self._get_data_quality_indicator()
                
                # Charts
                system_chart = self._create_system_metrics_chart()
                model_chart = self._create_model_performance_chart()
                api_chart = self._create_api_metrics_chart()
                
                # Alerts table
                alerts_table = self._create_alerts_table()
                
                return (system_status, model_status, data_quality_status,
                       system_chart, model_chart, api_chart, alerts_table)
                
            except Exception as e:
                logger.error(f"Error updating dashboard: {e}")
                return ("Error", "Error", "Error", {}, {}, {}, "Error loading alerts")

    def _get_system_status_indicator(self) -> html.Div:
        """Get system status indicator"""
        if not self.metrics_history.get('system'):
            return html.Div("No Data", style={'color': 'gray'})
        
        latest = self.metrics_history['system'][-1]
        
        if latest['cpu_percent'] > 80 or latest['memory_percent'] > 85:
            status = "Warning"
            color = 'orange'
        elif latest['cpu_percent'] > 90 or latest['memory_percent'] > 95:
            status = "Critical"
            color = 'red'
        else:
            status = "Healthy"
            color = 'green'
        
        return html.Div(status, style={'color': color, 'fontWeight': 'bold'})

    def _get_model_status_indicator(self) -> html.Div:
        """Get model status indicator"""
        if not self.metrics_history.get('models'):
            return html.Div("No Data", style={'color': 'gray'})
        
        # Check if any model has high MAPE
        high_mape = False
        for model_history in self.metrics_history['models'].values():
            if model_history:
                latest = model_history[-1]
                if latest.get('val_mape', 0) > 20:
                    high_mape = True
                    break
        
        if high_mape:
            status = "Performance Issue"
            color = 'orange'
        else:
            status = "Performing Well"
            color = 'green'
        
        return html.Div(status, style={'color': color, 'fontWeight': 'bold'})

    def _get_data_quality_indicator(self) -> html.Div:
        """Get data quality indicator"""
        if not self.metrics_history.get('data_quality'):
            return html.Div("No Data", style={'color': 'gray'})
        
        latest = self.metrics_history['data_quality'][-1]
        quality_score = latest['quality_score']
        
        if quality_score >= 85:
            status = f"Excellent ({quality_score:.1f}%)"
            color = 'green'
        elif quality_score >= 70:
            status = f"Good ({quality_score:.1f}%)"
            color = 'orange'
        else:
            status = f"Poor ({quality_score:.1f}%)"
            color = 'red'
        
        return html.Div(status, style={'color': color, 'fontWeight': 'bold'})

    def _create_system_metrics_chart(self) -> Dict:
        """Create system metrics chart"""
        if not self.metrics_history.get('system'):
            return {}
        
        df = pd.DataFrame(self.metrics_history['system'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('CPU Usage (%)', 'Memory Usage (%)', 'Disk Usage (%)'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cpu_percent'], name='CPU %'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['memory_percent'], name='Memory %'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['disk_percent'], name='Disk %'),
            row=3, col=1
        )
        
        fig.update_layout(height=600, title_text="System Metrics")
        return fig

    def _create_model_performance_chart(self) -> Dict:
        """Create model performance chart"""
        if not self.metrics_history.get('models'):
            return {}
        
        fig = go.Figure()
        
        for model_name, model_history in self.metrics_history['models'].items():
            if model_history:
                df = pd.DataFrame(model_history)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                if 'val_mape' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'], 
                            y=df['val_mape'], 
                            name=f'{model_name} MAPE'
                        )
                    )
        
        fig.update_layout(title="Model Performance (MAPE %)", yaxis_title="MAPE %")
        return fig

    def _create_api_metrics_chart(self) -> Dict:
        """Create API metrics chart"""
        if not self.metrics_history.get('api'):
            return {}
        
        df = pd.DataFrame(self.metrics_history['api'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Request Count', 'Average Latency (s)'),
            vertical_spacing=0.1
        )
        
        if 'total_requests' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['total_requests'], name='Requests'),
                row=1, col=1
            )
        
        if 'average_latency' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['average_latency'], name='Latency'),
                row=2, col=1
            )
        
        fig.update_layout(height=400, title_text="API Metrics")
        return fig

    def _create_alerts_table(self) -> html.Table:
        """Create alerts table"""
        if not self.alert_history:
            return html.P("No recent alerts")
        
        # Get last 10 alerts
        recent_alerts = self.alert_history[-10:]
        
        table_header = [
            html.Thead([
                html.Tr([
                    html.Th("Timestamp"),
                    html.Th("Type"),
                    html.Th("Severity"),
                    html.Th("Message")
                ])
            ])
        ]
        
        table_body = [
            html.Tbody([
                html.Tr([
                    html.Td(alert['timestamp'][:19]),  # Remove milliseconds
                    html.Td(alert['type']),
                    html.Td(alert['severity'], style={
                        'color': 'red' if alert['severity'] == 'critical' else 'orange'
                    }),
                    html.Td(alert['message'])
                ]) for alert in reversed(recent_alerts)  # Most recent first
            ])
        ]
        
        return html.Table(table_header + table_body, style={'width': '100%'})

    def export_metrics(self, output_path: str = None) -> str:
        """Export metrics history to file"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"performance_metrics_{timestamp}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'metrics_history': self.metrics_history,
            'alert_history': self.alert_history,
            'configuration': {
                'thresholds': self.thresholds,
                'monitoring_config': self.monitoring_config
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to: {output_path}")
        return output_path

    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'system_health': {},
            'model_health': {},
            'data_health': {},
            'alerts_count': len(self.alert_history),
            'monitoring_uptime': time.time() - (self.metrics_history.get('system', [{}])[0].get('timestamp', time.time()) if self.metrics_history.get('system') else time.time())
        }
        
        try:
            # System health
            if self.metrics_history.get('system'):
                latest_system = self.metrics_history['system'][-1]
                summary['system_health'] = {
                    'cpu_percent': latest_system['cpu_percent'],
                    'memory_percent': latest_system['memory_percent'],
                    'disk_percent': latest_system['disk_percent'],
                    'status': 'healthy' if latest_system['cpu_percent'] < 80 and latest_system['memory_percent'] < 85 else 'warning'
                }
            
            # Model health
            if self.metrics_history.get('models'):
                model_statuses = []
                for model_name, model_history in self.metrics_history['models'].items():
                    if model_history:
                        latest = model_history[-1]
                        mape = latest.get('val_mape', 0)
                        status = 'healthy' if mape < 15 else 'warning' if mape < 25 else 'critical'
                        model_statuses.append(status)
                
                summary['model_health'] = {
                    'models_count': len(self.metrics_history['models']),
                    'healthy_models': sum(1 for s in model_statuses if s == 'healthy'),
                    'overall_status': 'healthy' if all(s in ['healthy', 'warning'] for s in model_statuses) else 'warning'
                }
            
            # Data health
            if self.metrics_history.get('data_quality'):
                latest_quality = self.metrics_history['data_quality'][-1]
                summary['data_health'] = {
                    'quality_score': latest_quality['quality_score'],
                    'freshness_hours': latest_quality['data_freshness'],
                    'status': 'healthy' if latest_quality['quality_score'] > 80 and latest_quality['data_freshness'] < 24 else 'warning'
                }
            
            # Overall status
            statuses = [
                summary['system_health'].get('status', 'unknown'),
                summary['model_health'].get('overall_status', 'unknown'),
                summary['data_health'].get('status', 'unknown')
            ]
            
            if 'critical' in statuses:
                summary['overall_status'] = 'critical'
            elif 'warning' in statuses:
                summary['overall_status'] = 'warning'
            else:
                summary['overall_status'] = 'healthy'
            
        except Exception as e:
            logger.error(f"Error creating health summary: {e}")
            summary['overall_status'] = 'unknown'
            summary['error'] = str(e)
        
        return summary


def main():
    """Main function for performance monitoring"""
    parser = argparse.ArgumentParser(description='Performance Monitor for Chinese Produce Forecasting')
    parser.add_argument('--config', required=True, help='Path to config.yaml')
    parser.add_argument('--action', required=True,
                       choices=['start', 'dashboard', 'export', 'health'],
                       help='Action to perform')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in seconds')
    parser.add_argument('--port', type=int, default=8050, help='Dashboard port')
    parser.add_argument('--output', help='Output file path for export')
    
    args = parser.parse_args()
    
    try:
        # Initialize performance monitor
        monitor = PerformanceMonitor(args.config)
        
        if args.action == 'start':
            print(f"Starting performance monitoring (interval: {args.interval}s)")
            print("Press Ctrl+C to stop...")
            
            monitor.start_monitoring(args.interval)
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping monitoring...")
                monitor.stop_monitoring()
        
        elif args.action == 'dashboard':
            print(f"Starting monitoring dashboard on port {args.port}")
            print("Starting background monitoring...")
            
            # Start monitoring in background
            monitor.start_monitoring(args.interval)
            
            # Start dashboard (this will block)
            dashboard_url = monitor.create_dashboard(args.port)
            print(f"Dashboard available at: {dashboard_url}")
        
        elif args.action == 'export':
            output_file = monitor.export_metrics(args.output)
            print(f"Metrics exported to: {output_file}")
        
        elif args.action == 'health':
            health = monitor.get_health_summary()
            print("\nSystem Health Summary:")
            print("="*50)
            print(f"Overall Status: {health['overall_status'].upper()}")
            print(f"Timestamp: {health['timestamp']}")
            
            if 'system_health' in health:
                print(f"\nSystem Health: {health['system_health'].get('status', 'unknown').upper()}")
                print(f"  CPU: {health['system_health'].get('cpu_percent', 0):.1f}%")
                print(f"  Memory: {health['system_health'].get('memory_percent', 0):.1f}%")
                print(f"  Disk: {health['system_health'].get('disk_percent', 0):.1f}%")
            
            if 'model_health' in health:
                print(f"\nModel Health: {health['model_health'].get('overall_status', 'unknown').upper()}")
                print(f"  Models: {health['model_health'].get('models_count', 0)}")
                print(f"  Healthy: {health['model_health'].get('healthy_models', 0)}")
            
            if 'data_health' in health:
                print(f"\nData Health: {health['data_health'].get('status', 'unknown').upper()}")
                print(f"  Quality Score: {health['data_health'].get('quality_score', 0):.1f}%")
                print(f"  Freshness: {health['data_health'].get('freshness_hours', 0):.1f} hours")
            
            print(f"\nAlerts: {health.get('alerts_count', 0)}")
        
        logger.info("Performance monitoring operation completed successfully")
        
    except Exception as e:
        logger.error(f"Performance monitoring operation failed: {e}")
        print(f"\n❌ Operation failed: {e}")


if __name__ == "__main__":
    main()