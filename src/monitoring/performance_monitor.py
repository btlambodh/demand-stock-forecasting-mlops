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
import psutil
from prometheus_client import Gauge, Counter, Histogram, start_http_server, generate_latest, CollectorRegistry, REGISTRY
import requests

# Optional dependencies
try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    logging.warning("boto3 not available, AWS features disabled")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import dash
    from dash import dcc, html, Input, Output, callback
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    logging.warning("Dashboard dependencies not available")

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('performance_monitoring')


class PrometheusMetrics:
    """Manages Prometheus metrics with safe registration for testing"""
    
    def __init__(self, registry=None):
        self.registry = registry or REGISTRY
        self._metrics = {}
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize metrics with safe registration"""
        try:
            self._metrics = {
                'model_accuracy': Gauge(
                    'model_accuracy', 
                    'Model accuracy percentage', 
                    ['model_name', 'metric_type'],
                    registry=self.registry
                ),
                'model_latency': Histogram(
                    'model_prediction_latency_seconds', 
                    'Model prediction latency', 
                    ['model_name'],
                    registry=self.registry
                ),
                'system_cpu': Gauge(
                    'system_cpu_percent', 
                    'System CPU usage percentage',
                    registry=self.registry
                ),
                'system_memory': Gauge(
                    'system_memory_percent', 
                    'System memory usage percentage',
                    registry=self.registry
                ),
                'system_disk': Gauge(
                    'system_disk_percent', 
                    'System disk usage percentage',
                    registry=self.registry
                ),
                'api_requests_total': Counter(
                    'api_requests_total', 
                    'Total API requests', 
                    ['endpoint', 'status'],
                    registry=self.registry
                ),
                'api_errors_total': Counter(
                    'api_errors_total', 
                    'Total API errors', 
                    ['error_type'],
                    registry=self.registry
                ),
                'data_quality_score': Gauge(
                    'data_quality_score', 
                    'Data quality score percentage',
                    registry=self.registry
                ),
                'drift_score': Gauge(
                    'drift_score', 
                    'Data drift score', 
                    ['feature_name'],
                    registry=self.registry
                ),
                'prediction_count': Counter(
                    'predictions_total', 
                    'Total predictions made', 
                    ['model_name'],
                    registry=self.registry
                )
            }
        except ValueError as e:
            if "Duplicated timeseries" in str(e):
                # Metrics already exist in registry, get references to them
                logger.warning("Metrics already registered, using existing instances")
                self._get_existing_metrics()
            else:
                raise e
    
    def _get_existing_metrics(self):
        """Get references to existing metrics from registry"""
        # This is a fallback - in testing we should use a clean registry
        # For now, create dummy metrics that won't conflict
        self._metrics = {
            'model_accuracy': None,
            'model_latency': None,
            'system_cpu': None,
            'system_memory': None,
            'system_disk': None,
            'api_requests_total': None,
            'api_errors_total': None,
            'data_quality_score': None,
            'drift_score': None,
            'prediction_count': None
        }
    
    def __getattr__(self, name):
        """Allow access to metrics as attributes"""
        if name in self._metrics:
            return self._metrics[name]
        raise AttributeError(f"No metric named {name}")


class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self, config_path: str, local_mode: bool = False, registry=None):
        """Initialize performance monitor with configuration"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            self.config = self._get_default_config()
        
        self.local_mode = local_mode
        self.aws_enabled = AWS_AVAILABLE and not local_mode
        
        # Initialize Prometheus metrics with safe registry
        if registry is None and local_mode:
            # Create a separate registry for testing
            self.metrics_registry = CollectorRegistry()
        else:
            self.metrics_registry = registry or REGISTRY
        
        try:
            self.metrics = PrometheusMetrics(self.metrics_registry)
        except Exception as e:
            logger.warning(f"Could not initialize Prometheus metrics: {e}")
            self.metrics = None
        
        # Initialize AWS clients only if available and not in local mode
        if self.aws_enabled:
            try:
                self.aws_config = self.config.get('aws', {})
                region = self.aws_config.get('region', 'us-east-1')
                self.cloudwatch = boto3.client('cloudwatch', region_name=region)
                self.s3_client = boto3.client('s3', region_name=region)
                logger.info("AWS CloudWatch integration enabled")
            except Exception as e:
                logger.warning(f"AWS initialization failed: {e}, falling back to local mode")
                self.aws_enabled = False
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metrics_history = {}
        self.alert_history = []
        
        # Performance thresholds
        self.thresholds = self.config.get('monitoring', {}).get('performance', {})
        
        # Create local storage directories
        self.metrics_dir = "data/monitoring/metrics"
        self.alerts_dir = "data/monitoring/alerts"
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.alerts_dir, exist_ok=True)
        
        # Dashboard app
        self.dash_app = None
        
        logger.info(f"PerformanceMonitor initialized (local_mode: {local_mode}, aws_enabled: {self.aws_enabled})")

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'monitoring': {
                'performance': {
                    'drift_threshold': 0.25,
                    'performance_degradation_threshold': 0.15,
                    'cpu_threshold': 80,
                    'memory_threshold': 85,
                    'disk_threshold': 90
                },
                'alerts': {
                    'enabled': True,
                    'cooldown_minutes': 30
                }
            },
            'aws': {
                'region': 'us-east-1',
                'cloudwatch': {
                    'metrics_namespace': 'ChineseProduceForecast'
                }
            }
        }

    def _safe_set_metric(self, metric_name: str, value: float, labels: Dict = None):
        """Safely set a metric value"""
        if not self.metrics:
            return
        
        try:
            metric = getattr(self.metrics, metric_name, None)
            if metric is None:
                return
            
            if labels:
                if hasattr(metric, 'labels'):
                    metric.labels(**labels).set(value)
                else:
                    # For metrics without labels
                    metric.set(value)
            else:
                metric.set(value)
        except Exception as e:
            logger.debug(f"Could not set metric {metric_name}: {e}")

    def collect_current_metrics(self):
        """Collect current metrics once for immediate health reporting"""
        logger.info("Collecting current system metrics for health report")
        
        try:
            # Collect all current metrics
            self.collect_system_metrics()
            self.collect_model_metrics()
            self.collect_api_metrics()
            self.collect_data_quality_metrics()
            
            logger.info("Current metrics collection completed")
            
        except Exception as e:
            logger.error(f"Error collecting current metrics: {e}")

    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return
        
        logger.info(f"Starting performance monitoring")
        logger.info(f"   Interval: {interval_seconds} seconds")
        logger.info(f"   Local mode: {self.local_mode}")
        logger.info(f"   AWS enabled: {self.aws_enabled}")
        
        self.is_monitoring = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        # Start Prometheus metrics server (only if not in local mode)
        if not self.local_mode:
            try:
                start_http_server(8002, registry=self.metrics_registry)
                logger.info("Prometheus metrics server started on port 8002")
                logger.info("Metrics available at: http://localhost:8002/metrics")
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
                start_time = time.time()
                
                # Collect all metrics
                self.collect_system_metrics()
                self.collect_model_metrics()
                self.collect_api_metrics()
                self.collect_data_quality_metrics()
                
                # Check thresholds and send alerts
                self.check_alert_conditions()
                
                # Save metrics locally
                self._save_metrics_locally()
                
                # Calculate sleep time
                collection_time = time.time() - start_time
                sleep_time = max(0, interval_seconds - collection_time)
                
                logger.debug(f"Metrics collection took {collection_time:.2f}s, sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
        
        logger.info("Performance monitoring loop stopped")

    def collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._safe_set_metric('system_cpu', cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self._safe_set_metric('system_memory', memory_percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self._safe_set_metric('system_disk', disk_percent)
            
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
            
            # Send to CloudWatch (only if enabled)
            if self.aws_enabled:
                self._send_cloudwatch_metrics('System', [
                    {'MetricName': 'CPUUtilization', 'Value': cpu_percent, 'Unit': 'Percent'},
                    {'MetricName': 'MemoryUtilization', 'Value': memory_percent, 'Unit': 'Percent'},
                    {'MetricName': 'DiskUtilization', 'Value': disk_percent, 'Unit': 'Percent'}
                ])
            
            logger.debug(f"System: CPU={cpu_percent:.1f}%, Memory={memory_percent:.1f}%, Disk={disk_percent:.1f}%")
            
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
                        self._safe_set_metric('model_accuracy', value, {
                            'model_name': model_name, 
                            'metric_type': metric_name
                        })
                
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
            
            # Send to CloudWatch (only if enabled)
            if self.aws_enabled:
                cloudwatch_metrics = []
                for model_name, metrics in model_metrics.items():
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            unit = self._get_valid_cloudwatch_unit(metric_name)
                            cloudwatch_metrics.append({
                                'MetricName': f'Model_{metric_name}',
                                'Value': value,
                                'Unit': unit,
                                'Dimensions': [{'Name': 'ModelName', 'Value': model_name}]
                            })
                
                if cloudwatch_metrics:
                    self._send_cloudwatch_metrics('Models', cloudwatch_metrics)
            
            logger.debug(f"Collected metrics for {len(model_metrics)} models")
            
        except Exception as e:
            logger.error(f"Error collecting model metrics: {e}")

    def _get_valid_cloudwatch_unit(self, metric_name: str) -> str:
        """Get valid CloudWatch unit for metric"""
        metric_name_lower = metric_name.lower()
        
        if any(x in metric_name_lower for x in ['mape', 'percent', 'rate', 'ratio', 'accuracy']):
            return 'Percent'
        elif any(x in metric_name_lower for x in ['latency', 'time', 'duration']):
            return 'Seconds'
        elif any(x in metric_name_lower for x in ['count', 'total', 'number']):
            return 'Count'
        elif any(x in metric_name_lower for x in ['memory', 'disk', 'size']):
            return 'Bytes'
        else:
            return 'None'  # Valid CloudWatch unit for dimensionless metrics

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
                    if isinstance(metrics, dict):
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
                    uptime = health_data.get('uptime_seconds', 0)
                    
                    # Add API health metrics
                    model_metrics['api_health'] = {
                        'models_loaded': models_loaded,
                        'uptime_seconds': uptime,
                        'status_healthy': 1 if health_data.get('status') == 'healthy' else 0,
                        'response_time_ms': 200  # Approximate based on successful response
                    }
            except:
                # API might not be running
                model_metrics['api_health'] = {
                    'models_loaded': 0,
                    'uptime_seconds': 0,
                    'status_healthy': 0,
                    'response_time_ms': 0
                }
            
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
            
            # Send to CloudWatch (only if enabled)
            if self.aws_enabled:
                cloudwatch_metrics = []
                for metric_name, value in api_metrics.items():
                    if isinstance(value, (int, float)):
                        unit = self._get_valid_cloudwatch_unit(metric_name)
                        cloudwatch_metrics.append({
                            'MetricName': f'API_{metric_name}',
                            'Value': value,
                            'Unit': unit
                        })
                
                if cloudwatch_metrics:
                    self._send_cloudwatch_metrics('API', cloudwatch_metrics)
            
            logger.debug(f"Collected API metrics: {len(api_metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Error collecting API metrics: {e}")

    def _fetch_api_metrics(self) -> Dict[str, float]:
        """Fetch API metrics from Prometheus endpoint"""
        api_metrics = {
            'total_requests': 0,
            'prediction_count': 0,
            'average_latency_ms': 0,
            'error_count': 0,
            'monitoring_timestamp': time.time()
        }
        
        try:
            # Try to get metrics from Prometheus endpoint
            response = requests.get('http://localhost:8000/metrics', timeout=5)
            if response.status_code == 200:
                metrics_text = response.text
                
                # Parse basic metrics (simplified parsing)
                lines = metrics_text.split('\n')
                total_latency = 0
                prediction_count = 0
                
                for line in lines:
                    if 'api_requests_total' in line and not line.startswith('#'):
                        # Extract value
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                value = float(parts[-1])
                                api_metrics['total_requests'] += value
                            except:
                                pass
                    
                    elif 'prediction_latency_seconds' in line and not line.startswith('#'):
                        # Extract latency metrics
                        if '_sum' in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    total_latency = float(parts[-1])
                                except:
                                    pass
                        elif '_count' in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    prediction_count = float(parts[-1])
                                except:
                                    pass
                
                # Calculate derived metrics
                if prediction_count > 0:
                    api_metrics['prediction_count'] = prediction_count
                    api_metrics['average_latency_ms'] = (total_latency / prediction_count) * 1000
            
            # Test API responsiveness
            try:
                start_time = time.time()
                health_response = requests.get('http://localhost:8000/health', timeout=5)
                response_time = (time.time() - start_time) * 1000
                
                if health_response.status_code == 200:
                    api_metrics['api_response_time_ms'] = response_time
                    api_metrics['api_available'] = 1
                else:
                    api_metrics['api_available'] = 0
            except:
                api_metrics['api_available'] = 0
                api_metrics['api_response_time_ms'] = 0
            
        except Exception as e:
            logger.debug(f"Could not fetch API metrics: {e}")
        
        return api_metrics

    def collect_data_quality_metrics(self):
        """Collect data quality metrics"""
        try:
            # Calculate data quality from recent predictions or validation data
            quality_score = self._calculate_data_quality()
            
            self._safe_set_metric('data_quality_score', quality_score)
            
            # Store in history
            if 'data_quality' not in self.metrics_history:
                self.metrics_history['data_quality'] = []
            
            entry = {
                'timestamp': datetime.now().isoformat(),
                'quality_score': quality_score,
                'data_freshness_hours': self._check_data_freshness(),
                'validation_status': self._check_validation_status()
            }
            self.metrics_history['data_quality'].append(entry)
            
            # Keep only last 500 entries
            if len(self.metrics_history['data_quality']) > 500:
                self.metrics_history['data_quality'] = self.metrics_history['data_quality'][-500:]
            
            # Send to CloudWatch (only if enabled)
            if self.aws_enabled:
                self._send_cloudwatch_metrics('DataQuality', [
                    {'MetricName': 'QualityScore', 'Value': quality_score, 'Unit': 'Percent'},
                    {'MetricName': 'DataFreshness', 'Value': entry['data_freshness_hours'], 'Unit': 'Count'}
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
            
            # Check if basic data files exist and are recent
            data_files = [
                "data/processed/train.parquet",
                "data/processed/validation.parquet",
                "models/evaluation.json"
            ]
            
            existing_files = sum(1 for f in data_files if os.path.exists(f))
            file_ratio = existing_files / len(data_files)
            
            # Base score on file availability
            base_score = file_ratio * 80
            
            # Add bonus for recent files
            if existing_files > 0:
                freshness_bonus = max(0, 20 - self._check_data_freshness())
                return min(100, base_score + freshness_bonus)
            
            return base_score
            
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
                "data/raw/annex2.csv",
                "models/evaluation.json"
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
            
            # Check if API is responding as validation
            try:
                response = requests.get('http://localhost:8000/health', timeout=3)
                if response.status_code == 200:
                    return 'api_healthy'
            except:
                pass
            
            return 'unknown'
            
        except Exception as e:
            logger.debug(f"Could not check validation status: {e}")
            return 'unknown'

    def check_alert_conditions(self):
        """Check if any alert conditions are met"""
        try:
            alerts_triggered = []
            current_time = datetime.now()
            
            # Check system thresholds
            if self.metrics_history.get('system'):
                latest_system = self.metrics_history['system'][-1]
                
                cpu_threshold = self.thresholds.get('cpu_threshold', 80)
                memory_threshold = self.thresholds.get('memory_threshold', 85)
                disk_threshold = self.thresholds.get('disk_threshold', 90)
                
                if latest_system['cpu_percent'] > cpu_threshold:
                    alerts_triggered.append({
                        'type': 'system_cpu',
                        'severity': 'critical' if latest_system['cpu_percent'] > 95 else 'warning',
                        'message': f"High CPU usage: {latest_system['cpu_percent']:.1f}%",
                        'metric': 'cpu_percent',
                        'value': latest_system['cpu_percent'],
                        'threshold': cpu_threshold,
                        'timestamp': current_time.isoformat()
                    })
                
                if latest_system['memory_percent'] > memory_threshold:
                    alerts_triggered.append({
                        'type': 'system_memory',
                        'severity': 'critical' if latest_system['memory_percent'] > 95 else 'warning',
                        'message': f"High memory usage: {latest_system['memory_percent']:.1f}%",
                        'metric': 'memory_percent',
                        'value': latest_system['memory_percent'],
                        'threshold': memory_threshold,
                        'timestamp': current_time.isoformat()
                    })
                
                if latest_system['disk_percent'] > disk_threshold:
                    alerts_triggered.append({
                        'type': 'system_disk',
                        'severity': 'critical',
                        'message': f"High disk usage: {latest_system['disk_percent']:.1f}%",
                        'metric': 'disk_percent',
                        'value': latest_system['disk_percent'],
                        'threshold': disk_threshold,
                        'timestamp': current_time.isoformat()
                    })
            
            # Check model performance thresholds
            if self.metrics_history.get('models'):
                for model_name, model_history in self.metrics_history['models'].items():
                    if model_history and model_name != 'api_health':
                        latest_model = model_history[-1]
                        
                        # Check MAPE threshold
                        val_mape = latest_model.get('val_mape', 0)
                        mape_threshold = 20  # 20% MAPE threshold
                        
                        if val_mape > mape_threshold:
                            alerts_triggered.append({
                                'type': 'model_performance',
                                'severity': 'critical' if val_mape > 30 else 'warning',
                                'message': f"High MAPE for {model_name}: {val_mape:.2f}%",
                                'model': model_name,
                                'metric': 'val_mape',
                                'value': val_mape,
                                'threshold': mape_threshold,
                                'timestamp': current_time.isoformat()
                            })
            
            # Check API health
            if self.metrics_history.get('api'):
                latest_api = self.metrics_history['api'][-1]
                
                if latest_api.get('api_available', 1) == 0:
                    alerts_triggered.append({
                        'type': 'api_unavailable',
                        'severity': 'critical',
                        'message': "API is not responding",
                        'metric': 'api_available',
                        'value': 0,
                        'threshold': 1,
                        'timestamp': current_time.isoformat()
                    })
                
                avg_latency = latest_api.get('average_latency_ms', 0)
                if avg_latency > 2000:  # 2 second threshold
                    alerts_triggered.append({
                        'type': 'api_latency',
                        'severity': 'warning',
                        'message': f"High API latency: {avg_latency:.1f}ms",
                        'metric': 'average_latency_ms',
                        'value': avg_latency,
                        'threshold': 2000,
                        'timestamp': current_time.isoformat()
                    })
            
            # Check data quality thresholds
            if self.metrics_history.get('data_quality'):
                latest_quality = self.metrics_history['data_quality'][-1]
                
                quality_score = latest_quality['quality_score']
                if quality_score < 70:
                    alerts_triggered.append({
                        'type': 'data_quality',
                        'severity': 'critical' if quality_score < 50 else 'warning',
                        'message': f"Low data quality score: {quality_score:.1f}%",
                        'metric': 'quality_score',
                        'value': quality_score,
                        'threshold': 70,
                        'timestamp': current_time.isoformat()
                    })
                
                freshness_hours = latest_quality['data_freshness_hours']
                if freshness_hours > 48:  # 48 hours
                    alerts_triggered.append({
                        'type': 'data_freshness',
                        'severity': 'warning',
                        'message': f"Stale data: {freshness_hours:.1f} hours old",
                        'metric': 'data_freshness_hours',
                        'value': freshness_hours,
                        'threshold': 48,
                        'timestamp': current_time.isoformat()
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
            # Add to alert history
            self.alert_history.append(alert)
            
            # Keep only last 100 alerts
            if len(self.alert_history) > 100:
                self.alert_history = self.alert_history[-100:]
            
            # Save alert locally
            alert_file = os.path.join(self.alerts_dir, f"alert_{int(time.time())}.json")
            with open(alert_file, 'w') as f:
                json.dump(alert, f, indent=2)
            
            # Send to CloudWatch (only if enabled)
            if self.aws_enabled:
                try:
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
                except Exception as e:
                    logger.warning(f"Could not send alert to CloudWatch: {e}")
            
            # Log alert
            severity_emoji = {"critical": "CRITICAL", "warning": "WARNING", "info": "INFO"}.get(alert['severity'], "UNKNOWN")
            logger.warning(f"ALERT [{severity_emoji}] {alert['type']}: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Error processing alert: {e}")

    def _send_cloudwatch_metrics(self, namespace_suffix: str, metrics: List[Dict]):
        """Send metrics to CloudWatch (only if AWS enabled)"""
        if not self.aws_enabled:
            return
        
        try:
            if not metrics:
                return
            
            base_namespace = self.config.get('aws', {}).get('cloudwatch', {}).get('metrics_namespace', 'ChineseProduceForecast')
            namespace = f"{base_namespace}/{namespace_suffix}"
            
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

    def _save_metrics_locally(self):
        """Save metrics to local files"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_file = os.path.join(self.metrics_dir, f"metrics_{timestamp}.json")
            
            # Only save recent metrics to avoid large files
            recent_metrics = {}
            for category, data in self.metrics_history.items():
                if isinstance(data, dict):
                    # For model metrics, keep last 10 entries per model
                    recent_metrics[category] = {}
                    for model, history in data.items():
                        recent_metrics[category][model] = history[-10:] if history else []
                elif isinstance(data, list):
                    # For other metrics, keep last 50 entries
                    recent_metrics[category] = data[-50:] if data else []
            
            with open(metrics_file, 'w') as f:
                json.dump(recent_metrics, f, indent=2)
            
            # Clean up old metric files (keep last 24 hours)
            cutoff_time = time.time() - 24 * 3600
            for filename in os.listdir(self.metrics_dir):
                if filename.startswith('metrics_') and filename.endswith('.json'):
                    filepath = os.path.join(self.metrics_dir, filename)
                    if os.path.getmtime(filepath) < cutoff_time:
                        os.remove(filepath)
            
        except Exception as e:
            logger.debug(f"Could not save metrics locally: {e}")

    def create_dashboard(self, port: int = 8050) -> str:
        """Create interactive monitoring dashboard"""
        if not DASHBOARD_AVAILABLE:
            logger.error("Dashboard dependencies not available. Install: pip install plotly dash")
            return "Dashboard dependencies missing"
        
        logger.info(f"Creating monitoring dashboard on port {port}")
        
        # Initialize Dash app
        self.dash_app = dash.Dash(__name__)
        
        # Define dashboard layout
        self.dash_app.layout = html.Div([
            html.H1("Chinese Produce Forecasting - Monitoring Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': 30, 'color': '#2c3e50'}),
            
            # Status indicators row
            html.Div([
                html.Div([
                    html.H3("System Status", style={'textAlign': 'center', 'color': '#34495e'}),
                    html.Div(id='system-status', style={'textAlign': 'center', 'fontSize': 24, 'fontWeight': 'bold'})
                ], className='four columns', style={'backgroundColor': '#f8f9fa', 'padding': 20, 'margin': 10, 'borderRadius': 10}),
                
                html.Div([
                    html.H3("Model Status", style={'textAlign': 'center', 'color': '#34495e'}),
                    html.Div(id='model-status', style={'textAlign': 'center', 'fontSize': 24, 'fontWeight': 'bold'})
                ], className='four columns', style={'backgroundColor': '#f8f9fa', 'padding': 20, 'margin': 10, 'borderRadius': 10}),
                
                html.Div([
                    html.H3("Data Quality", style={'textAlign': 'center', 'color': '#34495e'}),
                    html.Div(id='data-quality-status', style={'textAlign': 'center', 'fontSize': 24, 'fontWeight': 'bold'})
                ], className='four columns', style={'backgroundColor': '#f8f9fa', 'padding': 20, 'margin': 10, 'borderRadius': 10}),
            ], className='row', style={'marginBottom': 30}),
            
            # Charts
            html.Div([
                dcc.Graph(id='system-metrics-chart'),
            ], style={'marginBottom': 20, 'backgroundColor': 'white', 'padding': 20, 'borderRadius': 10}),
            
            html.Div([
                dcc.Graph(id='model-performance-chart'),
            ], style={'marginBottom': 20, 'backgroundColor': 'white', 'padding': 20, 'borderRadius': 10}),
            
            html.Div([
                dcc.Graph(id='api-metrics-chart'),
            ], style={'marginBottom': 20, 'backgroundColor': 'white', 'padding': 20, 'borderRadius': 10}),
            
            # Recent alerts
            html.Div([
                html.H3("Recent Alerts", style={'color': '#34495e'}),
                html.Div(id='alerts-table')
            ], style={'marginBottom': 20, 'backgroundColor': 'white', 'padding': 20, 'borderRadius': 10}),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            )
        ], style={'backgroundColor': '#ecf0f1', 'padding': 20})
        
        # Register callbacks
        self._register_dashboard_callbacks()
        
        # Run dashboard
        try:
            logger.info("Dashboard server starting...")
            logger.info(f"Dashboard will be available at: http://localhost:{port}")
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
                error_div = html.Div("Error", style={'color': 'red'})
                empty_fig = {}
                return (error_div, error_div, error_div, empty_fig, empty_fig, empty_fig, "Error loading alerts")

    def _get_system_status_indicator(self) -> html.Div:
        """Get system status indicator"""
        if not self.metrics_history.get('system'):
            return html.Div("No Data", style={'color': 'gray'})
        
        latest = self.metrics_history['system'][-1]
        
        cpu_pct = latest['cpu_percent']
        mem_pct = latest['memory_percent']
        
        if cpu_pct > 90 or mem_pct > 95:
            status = "Critical"
            color = '#e74c3c'
        elif cpu_pct > 80 or mem_pct > 85:
            status = "Warning"
            color = '#f39c12'
        else:
            status = "Healthy"
            color = '#27ae60'
        
        details = f"CPU: {cpu_pct:.1f}% | Memory: {mem_pct:.1f}%"
        
        return html.Div([
            html.Div(status, style={'color': color, 'fontWeight': 'bold', 'fontSize': 18}),
            html.Div(details, style={'color': '#7f8c8d', 'fontSize': 14, 'marginTop': 5})
        ])

    def _get_model_status_indicator(self) -> html.Div:
        """Get model status indicator"""
        if not self.metrics_history.get('models'):
            return html.Div("No Data", style={'color': 'gray'})
        
        # Check API health first
        api_health = self.metrics_history['models'].get('api_health', [])
        if api_health:
            latest_api = api_health[-1]
            models_loaded = latest_api.get('models_loaded', 0)
            api_healthy = latest_api.get('status_healthy', 0)
            
            if not api_healthy:
                return html.Div([
                    html.Div("API Down", style={'color': '#e74c3c', 'fontWeight': 'bold', 'fontSize': 18}),
                    html.Div("API not responding", style={'color': '#7f8c8d', 'fontSize': 14, 'marginTop': 5})
                ])
        
        # Check model performance
        high_mape_models = 0
        total_models = 0
        
        for model_name, model_history in self.metrics_history['models'].items():
            if model_name != 'api_health' and model_history:
                total_models += 1
                latest = model_history[-1]
                if latest.get('val_mape', 0) > 20:
                    high_mape_models += 1
        
        if total_models == 0:
            status = "No Models"
            color = '#95a5a6'
            details = "No model metrics available"
        elif high_mape_models > total_models * 0.5:
            status = "Performance Issues"
            color = '#f39c12'
            details = f"{high_mape_models}/{total_models} models with high MAPE"
        else:
            status = "Performing Well"
            color = '#27ae60'
            details = f"{total_models} models loaded, {total_models - high_mape_models} healthy"
        
        return html.Div([
            html.Div(status, style={'color': color, 'fontWeight': 'bold', 'fontSize': 18}),
            html.Div(details, style={'color': '#7f8c8d', 'fontSize': 14, 'marginTop': 5})
        ])

    def _get_data_quality_indicator(self) -> html.Div:
        """Get data quality indicator"""
        if not self.metrics_history.get('data_quality'):
            return html.Div("No Data", style={'color': 'gray'})
        
        latest = self.metrics_history['data_quality'][-1]
        quality_score = latest['quality_score']
        freshness_hours = latest['data_freshness_hours']
        
        if quality_score >= 85 and freshness_hours < 24:
            status = "Excellent"
            color = '#27ae60'
        elif quality_score >= 70 and freshness_hours < 48:
            status = "Good"
            color = '#f39c12'
        else:
            status = "Issues"
            color = '#e74c3c'
        
        details = f"Score: {quality_score:.1f}% | Age: {freshness_hours:.1f}h"
        
        return html.Div([
            html.Div(status, style={'color': color, 'fontWeight': 'bold', 'fontSize': 18}),
            html.Div(details, style={'color': '#7f8c8d', 'fontSize': 14, 'marginTop': 5})
        ])

    def _create_system_metrics_chart(self) -> Dict:
        """Create system metrics chart"""
        if not self.metrics_history.get('system'):
            return {'data': [], 'layout': {'title': 'No System Data Available'}}
        
        df = pd.DataFrame(self.metrics_history['system'][-50:])  # Last 50 entries
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('CPU Usage (%)', 'Memory Usage (%)', 'Disk Usage (%)'),
            vertical_spacing=0.08
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cpu_percent'], 
                      name='CPU %', line=dict(color='#3498db')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['memory_percent'], 
                      name='Memory %', line=dict(color='#e74c3c')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['disk_percent'], 
                      name='Disk %', line=dict(color='#f39c12')),
            row=3, col=1
        )
        
        # Add threshold lines
        fig.add_hline(y=80, line_dash="dash", line_color="orange", row=1, col=1)
        fig.add_hline(y=85, line_dash="dash", line_color="orange", row=2, col=1)
        fig.add_hline(y=90, line_dash="dash", line_color="red", row=3, col=1)
        
        fig.update_layout(height=600, title_text="System Performance Metrics", showlegend=False)
        return fig

    def _create_model_performance_chart(self) -> Dict:
        """Create model performance chart"""
        if not self.metrics_history.get('models'):
            return {'data': [], 'layout': {'title': 'No Model Data Available'}}
        
        fig = go.Figure()
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        color_idx = 0
        
        for model_name, model_history in self.metrics_history['models'].items():
            if model_name != 'api_health' and model_history:
                df = pd.DataFrame(model_history[-20:])  # Last 20 entries
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                if 'val_mape' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'], 
                            y=df['val_mape'], 
                            name=f'{model_name} MAPE',
                            line=dict(color=colors[color_idx % len(colors)])
                        )
                    )
                    color_idx += 1
        
        # Add threshold line
        fig.add_hline(y=20, line_dash="dash", line_color="red", 
                     annotation_text="MAPE Threshold (20%)")
        
        fig.update_layout(
            title="Model Performance (MAPE %)", 
            yaxis_title="MAPE %",
            xaxis_title="Time",
            height=400
        )
        return fig

    def _create_api_metrics_chart(self) -> Dict:
        """Create API metrics chart"""
        if not self.metrics_history.get('api'):
            return {'data': [], 'layout': {'title': 'No API Data Available'}}
        
        df = pd.DataFrame(self.metrics_history['api'][-30:])  # Last 30 entries
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Request Count', 'Response Time (ms)'),
            vertical_spacing=0.15
        )
        
        if 'total_requests' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['total_requests'], 
                          name='Total Requests', line=dict(color='#3498db')),
                row=1, col=1
            )
        
        if 'api_response_time_ms' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['api_response_time_ms'], 
                          name='Response Time', line=dict(color='#e74c3c')),
                row=2, col=1
            )
        
        # Add threshold line for response time
        fig.add_hline(y=2000, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.update_layout(height=400, title_text="API Performance Metrics", showlegend=False)
        return fig

    def _create_alerts_table(self) -> html.Table:
        """Create alerts table"""
        if not self.alert_history:
            return html.Div("No recent alerts", style={'color': '#27ae60', 'fontSize': 16, 'textAlign': 'center', 'padding': 20})
        
        # Get last 10 alerts
        recent_alerts = self.alert_history[-10:]
        
        table_header = [
            html.Thead([
                html.Tr([
                    html.Th("Time", style={'backgroundColor': '#34495e', 'color': 'white', 'padding': 10}),
                    html.Th("Type", style={'backgroundColor': '#34495e', 'color': 'white', 'padding': 10}),
                    html.Th("Severity", style={'backgroundColor': '#34495e', 'color': 'white', 'padding': 10}),
                    html.Th("Message", style={'backgroundColor': '#34495e', 'color': 'white', 'padding': 10})
                ])
            ])
        ]
        
        table_body = [
            html.Tbody([
                html.Tr([
                    html.Td(alert['timestamp'][:19], style={'padding': 8}),
                    html.Td(alert['type'], style={'padding': 8}),
                    html.Td(alert['severity'], style={
                        'color': '#e74c3c' if alert['severity'] == 'critical' else '#f39c12',
                        'fontWeight': 'bold',
                        'padding': 8
                    }),
                    html.Td(alert['message'], style={'padding': 8})
                ], style={'backgroundColor': '#f8f9fa' if i % 2 == 0 else 'white'}) 
                for i, alert in enumerate(reversed(recent_alerts))  # Most recent first
            ])
        ]
        
        return html.Table(table_header + table_body, style={'width': '100%', 'border': '1px solid #ddd'})

    def export_metrics(self, output_path: str = None) -> str:
        """Export metrics history to file"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"performance_metrics_{timestamp}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'local_mode': self.local_mode,
            'aws_enabled': self.aws_enabled,
            'metrics_history': self.metrics_history,
            'alert_history': self.alert_history,
            'configuration': {
                'thresholds': self.thresholds,
                'monitoring_config': self.config.get('monitoring', {})
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
            'local_mode': self.local_mode,
            'aws_enabled': self.aws_enabled,
            'system_health': {},
            'model_health': {},
            'data_health': {},
            'api_health': {},
            'alerts_count': len(self.alert_history),
            'monitoring_uptime_minutes': 0
        }
        
        try:
            # Calculate monitoring uptime
            if self.metrics_history.get('system'):
                first_metric_time = self.metrics_history['system'][0]['timestamp']
                first_time = datetime.fromisoformat(first_metric_time)
                uptime_seconds = (datetime.now() - first_time).total_seconds()
                summary['monitoring_uptime_minutes'] = uptime_seconds / 60
            
            # System health
            if self.metrics_history.get('system'):
                latest_system = self.metrics_history['system'][-1]
                cpu_pct = latest_system['cpu_percent']
                mem_pct = latest_system['memory_percent']
                disk_pct = latest_system['disk_percent']
                
                if cpu_pct > 90 or mem_pct > 95:
                    sys_status = 'critical'
                elif cpu_pct > 80 or mem_pct > 85:
                    sys_status = 'warning'
                else:
                    sys_status = 'healthy'
                
                summary['system_health'] = {
                    'cpu_percent': cpu_pct,
                    'memory_percent': mem_pct,
                    'disk_percent': disk_pct,
                    'memory_available_gb': latest_system['memory_available_gb'],
                    'disk_free_gb': latest_system['disk_free_gb'],
                    'status': sys_status
                }
            else:
                summary['system_health'] = {
                    'cpu_percent': 0.0,
                    'memory_percent': 0.0,
                    'disk_percent': 0.0,
                    'memory_available_gb': 0.0,
                    'disk_free_gb': 0.0,
                    'status': 'no_data'
                }
            
            # API health
            api_health_data = self.metrics_history.get('models', {}).get('api_health', [])
            if api_health_data:
                latest_api = api_health_data[-1]
                summary['api_health'] = {
                    'models_loaded': latest_api.get('models_loaded', 0),
                    'uptime_seconds': latest_api.get('uptime_seconds', 0),
                    'status_healthy': latest_api.get('status_healthy', 0),
                    'response_time_ms': latest_api.get('response_time_ms', 0),
                    'status': 'healthy' if latest_api.get('status_healthy', 0) else 'unhealthy'
                }
            else:
                summary['api_health'] = {
                    'models_loaded': 0,
                    'uptime_seconds': 0,
                    'status_healthy': 0,
                    'response_time_ms': 0,
                    'status': 'no_data'
                }
            
            # Model health
            if self.metrics_history.get('models'):
                model_statuses = []
                healthy_models = 0
                total_models = 0
                
                for model_name, model_history in self.metrics_history['models'].items():
                    if model_name != 'api_health' and model_history:
                        total_models += 1
                        latest = model_history[-1]
                        mape = latest.get('val_mape', 0)
                        
                        if mape < 15:
                            status = 'healthy'
                            healthy_models += 1
                        elif mape < 25:
                            status = 'warning'
                        else:
                            status = 'critical'
                        
                        model_statuses.append(status)
                
                overall_model_status = 'healthy'
                if total_models > 0:
                    critical_count = sum(1 for s in model_statuses if s == 'critical')
                    warning_count = sum(1 for s in model_statuses if s == 'warning')
                    
                    if critical_count > 0:
                        overall_model_status = 'critical'
                    elif warning_count > total_models * 0.5:
                        overall_model_status = 'warning'
                
                summary['model_health'] = {
                    'models_count': total_models,
                    'healthy_models': healthy_models,
                    'warning_models': sum(1 for s in model_statuses if s == 'warning'),
                    'critical_models': sum(1 for s in model_statuses if s == 'critical'),
                    'overall_status': overall_model_status
                }
            else:
                summary['model_health'] = {
                    'models_count': 0,
                    'healthy_models': 0,
                    'warning_models': 0,
                    'critical_models': 0,
                    'overall_status': 'no_data'
                }
            
            # Data health
            if self.metrics_history.get('data_quality'):
                latest_quality = self.metrics_history['data_quality'][-1]
                quality_score = latest_quality['quality_score']
                freshness_hours = latest_quality['data_freshness_hours']
                
                if quality_score > 85 and freshness_hours < 24:
                    data_status = 'healthy'
                elif quality_score > 70 and freshness_hours < 48:
                    data_status = 'warning'
                else:
                    data_status = 'critical'
                
                summary['data_health'] = {
                    'quality_score': quality_score,
                    'freshness_hours': freshness_hours,
                    'validation_status': latest_quality['validation_status'],
                    'status': data_status
                }
            else:
                summary['data_health'] = {
                    'quality_score': 0.0,
                    'freshness_hours': 0.0,
                    'validation_status': 'unknown',
                    'status': 'no_data'
                }
            
            # Overall status
            statuses = [
                summary['system_health'].get('status', 'no_data'),
                summary.get('api_health', {}).get('status', 'no_data'),
                summary['model_health'].get('overall_status', 'no_data'),
                summary['data_health'].get('status', 'no_data')
            ]
            
            if 'critical' in statuses:
                summary['overall_status'] = 'critical'
            elif 'warning' in statuses:
                summary['overall_status'] = 'warning'
            elif 'unhealthy' in statuses:
                summary['overall_status'] = 'warning'
            elif all(s == 'no_data' for s in statuses):
                summary['overall_status'] = 'no_data'
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
    parser.add_argument('--config', default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--action', required=True,
                       choices=['start', 'dashboard', 'export', 'health', 'alert-test'],
                       help='Action to perform')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in seconds')
    parser.add_argument('--port', type=int, default=8050, help='Dashboard port')
    parser.add_argument('--output', help='Output file path for export')
    parser.add_argument('--local-mode', action='store_true', help='Run in local mode (no AWS)')
    parser.add_argument('--cloudwatch', action='store_true', help='Enable CloudWatch integration')
    
    args = parser.parse_args()
    
    # Override local mode if cloudwatch is explicitly requested
    local_mode = args.local_mode and not args.cloudwatch
    
    try:
        # Initialize performance monitor
        monitor = PerformanceMonitor(args.config, local_mode=local_mode)
        
        if args.action == 'start':
            print(f"Starting Performance Monitoring")
            print(f"   Interval: {args.interval} seconds")
            print(f"   Local mode: {local_mode}")
            print(f"   AWS enabled: {not local_mode and AWS_AVAILABLE}")
            print(f"   Press Ctrl+C to stop...")
            
            monitor.start_monitoring(args.interval)
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping monitoring...")
                monitor.stop_monitoring()
        
        elif args.action == 'dashboard':
            print(f"Starting Monitoring Dashboard")
            print(f"   Port: {args.port}")
            print(f"   Local mode: {local_mode}")
            print("   Starting background monitoring...")
            
            if not DASHBOARD_AVAILABLE:
                print("Dashboard dependencies missing. Install:")
                print("   pip install plotly==5.17.0 dash==2.16.0")
                return
            
            # Start monitoring in background
            monitor.start_monitoring(args.interval)
            
            # Start dashboard (this will block)
            dashboard_url = monitor.create_dashboard(args.port)
            print(f"Dashboard: {dashboard_url}")
        
        elif args.action == 'export':
            output_file = monitor.export_metrics(args.output)
            print(f"Metrics exported to: {output_file}")
        
        elif args.action == 'health':
            # COLLECT METRICS FIRST before generating health summary
            print("Collecting current metrics for health report...")
            monitor.collect_current_metrics()
            
            health = monitor.get_health_summary()
            
            print("\nSystem Health Summary")
            print("="*50)
            print(f"Overall Status: {health['overall_status'].upper()}")
            print(f"Timestamp: {health['timestamp']}")
            print(f"Local Mode: {health['local_mode']}")
            print(f"AWS Enabled: {health['aws_enabled']}")
            
            if 'system_health' in health:
                sys_health = health['system_health']
                print(f"\nSystem Health: {sys_health.get('status', 'unknown').upper()}")
                print(f"  CPU: {sys_health.get('cpu_percent', 0):.1f}%")
                print(f"  Memory: {sys_health.get('memory_percent', 0):.1f}%")
                print(f"  Disk: {sys_health.get('disk_percent', 0):.1f}%")
                print(f"  Memory Available: {sys_health.get('memory_available_gb', 0):.1f} GB")
                print(f"  Disk Free: {sys_health.get('disk_free_gb', 0):.1f} GB")
            
            if 'api_health' in health:
                api_health = health['api_health']
                print(f"\nAPI Health: {api_health.get('status', 'unknown').upper()}")
                print(f"  Models Loaded: {api_health.get('models_loaded', 0)}")
                print(f"  Uptime: {api_health.get('uptime_seconds', 0):.1f} seconds")
                print(f"  Response Time: {api_health.get('response_time_ms', 0):.1f} ms")
            
            if 'model_health' in health:
                model_health = health['model_health']
                print(f"\nModel Health: {model_health.get('overall_status', 'unknown').upper()}")
                print(f"  Total Models: {model_health.get('models_count', 0)}")
                print(f"  Healthy: {model_health.get('healthy_models', 0)}")
                print(f"  Warning: {model_health.get('warning_models', 0)}")
                print(f"  Critical: {model_health.get('critical_models', 0)}")
            
            if 'data_health' in health:
                data_health = health['data_health']
                print(f"\nData Health: {data_health.get('status', 'unknown').upper()}")
                print(f"  Quality Score: {data_health.get('quality_score', 0):.1f}%")
                print(f"  Freshness: {data_health.get('freshness_hours', 0):.1f} hours")
                print(f"  Validation: {data_health.get('validation_status', 'unknown')}")
            
            print(f"\nActive Alerts: {health.get('alerts_count', 0)}")
            print(f"  Monitoring Uptime: {health.get('monitoring_uptime_minutes', 0):.1f} minutes")
            
            # Status summary
            status = health['overall_status']
            if status == 'healthy':
                print(f"\nAll systems operating normally!")
            elif status == 'warning':
                print(f"\nSome issues detected, monitoring recommended")
            elif status == 'critical':
                print(f"\nCritical issues detected, immediate attention required")
            elif status == 'no_data':
                print(f"\nNo monitoring data available yet")
            else:
                print(f"\nSystem status unknown")
        
        elif args.action == 'alert-test':
            print("Testing alert system...")
            
            # Create test alert
            test_alert = {
                'type': 'test_alert',
                'severity': 'warning',
                'message': 'This is a test alert from the monitoring system',
                'metric': 'test_metric',
                'value': 99.9,
                'threshold': 95.0,
                'timestamp': datetime.now().isoformat()
            }
            
            monitor._process_alert(test_alert)
            print("Test alert processed successfully")
            print(f"Alert saved to: {monitor.alerts_dir}")
            
            if monitor.aws_enabled:
                print("  Test alert sent to CloudWatch")
            else:
                print("Running in local mode - no CloudWatch integration")
        
        logger.info("Performance monitoring operation completed successfully")
        
    except Exception as e:
        logger.error(f"Performance monitoring operation failed: {e}")
        print(f"\nOperation failed: {e}")


if __name__ == "__main__":
    main()