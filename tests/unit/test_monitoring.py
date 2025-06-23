#!/usr/bin/env python3
"""
Unit tests for monitoring components
Tests performance monitoring and drift detection functionality

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import os
import pytest
import pandas as pd
import numpy as np
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import modules to test
from src.monitoring.performance_monitor import PerformanceMonitor
from src.monitoring.drift_detector import DriftDetector, ensure_python_type, safe_float, safe_bool


class TestPerformanceMonitor:
    """Unit tests for PerformanceMonitor class"""

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_monitor_initialization(self, config_file):
        """Test PerformanceMonitor initialization"""
        monitor = PerformanceMonitor(config_file, local_mode=True)
        assert monitor.config is not None
        assert monitor.local_mode is True
        assert monitor.aws_enabled is False
        assert hasattr(monitor, 'metrics_history')
        assert hasattr(monitor, 'alert_history')

    @pytest.mark.unit
    @pytest.mark.monitoring
    @patch('boto3.client')
    def test_monitor_initialization_aws(self, mock_boto_client, config_file):
        """Test PerformanceMonitor initialization with AWS"""
        monitor = PerformanceMonitor(config_file, local_mode=False)
        assert monitor.config is not None
        assert monitor.local_mode is False

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_collect_system_metrics(self, config_file):
        """Test system metrics collection"""
        monitor = PerformanceMonitor(config_file, local_mode=True)
        
        # Mock psutil calls
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_cpu.return_value = 45.5
            
            mock_memory_obj = Mock()
            mock_memory_obj.percent = 67.2
            mock_memory_obj.available = 8 * 1024**3  # 8GB
            mock_memory.return_value = mock_memory_obj
            
            mock_disk_obj = Mock()
            mock_disk_obj.used = 100 * 1024**3  # 100GB
            mock_disk_obj.total = 500 * 1024**3  # 500GB
            mock_disk_obj.free = 400 * 1024**3   # 400GB
            mock_disk.return_value = mock_disk_obj
            
            monitor.collect_system_metrics()
            
            # Check metrics were collected
            assert 'system' in monitor.metrics_history
            assert len(monitor.metrics_history['system']) > 0
            
            latest_metrics = monitor.metrics_history['system'][-1]
            assert 'cpu_percent' in latest_metrics
            assert 'memory_percent' in latest_metrics
            assert 'disk_percent' in latest_metrics
            assert latest_metrics['cpu_percent'] == 45.5
            assert latest_metrics['memory_percent'] == 67.2

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_collect_model_metrics(self, config_file, temp_dir, sample_evaluation_results):
        """Test model metrics collection"""
        monitor = PerformanceMonitor(config_file, local_mode=True)
        
        # Create mock evaluation file
        eval_file = os.path.join(temp_dir, 'evaluation.json')
        with open(eval_file, 'w') as f:
            json.dump(sample_evaluation_results, f)
        
        # Mock the evaluation file path
        with patch('os.path.exists') as mock_exists, \
             patch('builtins.open', create=True) as mock_open:
            
            mock_exists.return_value = True
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(sample_evaluation_results)
            
            monitor.collect_model_metrics()
            
            # Check metrics were collected
            assert 'models' in monitor.metrics_history

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_collect_api_metrics(self, config_file):
        """Test API metrics collection"""
        monitor = PerformanceMonitor(config_file, local_mode=True)
        
        # Mock requests to API
        with patch('requests.get') as mock_get:
            # Mock health endpoint response
            mock_health_response = Mock()
            mock_health_response.status_code = 200
            mock_health_response.json.return_value = {
                'status': 'healthy',
                'models_loaded': 3,
                'uptime_seconds': 3600
            }
            
            # Mock metrics endpoint response
            mock_metrics_response = Mock()
            mock_metrics_response.status_code = 200
            mock_metrics_response.text = """
            # HELP api_requests_total Total API requests
            # TYPE api_requests_total counter
            api_requests_total{endpoint="/predict",status="200"} 150.0
            # HELP prediction_latency_seconds_sum Prediction latency sum
            # TYPE prediction_latency_seconds_sum counter
            prediction_latency_seconds_sum 30.0
            # HELP prediction_latency_seconds_count Prediction latency count
            # TYPE prediction_latency_seconds_count counter
            prediction_latency_seconds_count 100.0
            """
            
            mock_get.side_effect = [mock_health_response, mock_metrics_response]
            
            monitor.collect_api_metrics()
            
            # Check metrics were collected
            assert 'api' in monitor.metrics_history
            assert len(monitor.metrics_history['api']) > 0

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_collect_data_quality_metrics(self, config_file, temp_dir):
        """Test data quality metrics collection"""
        monitor = PerformanceMonitor(config_file, local_mode=True)
        
        # Create mock validation report
        validation_data = {
            'overall_quality_score': 87.5,
            'validation_status': 'passed'
        }
        
        validation_file = os.path.join(temp_dir, 'validation_report.json')
        with open(validation_file, 'w') as f:
            json.dump(validation_data, f)
        
        # Mock file paths
        with patch('os.path.exists') as mock_exists, \
             patch('builtins.open', create=True) as mock_open:
            
            mock_exists.return_value = True
            mock_open.return_value.__enter__.return_value = Mock()
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(validation_data)
            
            monitor.collect_data_quality_metrics()
            
            # Check metrics were collected
            assert 'data_quality' in monitor.metrics_history
            assert len(monitor.metrics_history['data_quality']) > 0

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_check_alert_conditions(self, config_file):
        """Test alert condition checking"""
        monitor = PerformanceMonitor(config_file, local_mode=True)
        
        # Set up test metrics that should trigger alerts
        monitor.metrics_history = {
            'system': [{
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': 95.0,  # High CPU
                'memory_percent': 90.0,  # High memory
                'disk_percent': 95.0   # High disk
            }],
            'api': [{
                'timestamp': datetime.now().isoformat(),
                'api_available': 0,  # API down
                'average_latency_ms': 3000  # High latency
            }],
            'data_quality': [{
                'timestamp': datetime.now().isoformat(),
                'quality_score': 60.0,  # Low quality
                'data_freshness_hours': 72.0  # Stale data
            }]
        }
        
        # Mock alert processing
        with patch.object(monitor, '_process_alert') as mock_process_alert:
            monitor.check_alert_conditions()
            
            # Should have processed multiple alerts
            assert mock_process_alert.call_count > 0

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_process_alert(self, config_file, temp_dir):
        """Test alert processing"""
        monitor = PerformanceMonitor(config_file, local_mode=True)
        monitor.alerts_dir = temp_dir
        
        test_alert = {
            'type': 'test_alert',
            'severity': 'warning',
            'message': 'Test alert message',
            'metric': 'test_metric',
            'value': 85.0,
            'threshold': 80.0,
            'timestamp': datetime.now().isoformat()
        }
        
        monitor._process_alert(test_alert)
        
        # Check alert was added to history
        assert len(monitor.alert_history) > 0
        assert monitor.alert_history[-1]['type'] == 'test_alert'
        
        # Check alert file was created
        alert_files = [f for f in os.listdir(temp_dir) if f.startswith('alert_')]
        assert len(alert_files) > 0

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_get_health_summary(self, config_file):
        """Test health summary generation"""
        monitor = PerformanceMonitor(config_file, local_mode=True)
        
        # Set up some test metrics
        monitor.metrics_history = {
            'system': [{
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': 45.0,
                'memory_percent': 60.0,
                'disk_percent': 30.0,
                'memory_available_gb': 8.0,
                'disk_free_gb': 350.0
            }],
            'models': {
                'test_model': [{
                    'timestamp': datetime.now().isoformat(),
                    'val_mape': 12.5,
                    'val_rmse': 2.1,
                    'val_r2': 0.82
                }],
                'api_health': [{
                    'timestamp': datetime.now().isoformat(),
                    'models_loaded': 3,
                    'uptime_seconds': 3600,
                    'status_healthy': 1,
                    'response_time_ms': 150
                }]
            },
            'data_quality': [{
                'timestamp': datetime.now().isoformat(),
                'quality_score': 85.0,
                'data_freshness_hours': 12.0,
                'validation_status': 'passed'
            }]
        }
        
        summary = monitor.get_health_summary()
        
        assert isinstance(summary, dict)
        assert 'timestamp' in summary
        assert 'overall_status' in summary
        assert 'system_health' in summary
        assert 'model_health' in summary
        assert 'api_health' in summary
        assert 'data_health' in summary
        
        # Check status determination
        assert summary['overall_status'] in ['healthy', 'warning', 'critical', 'no_data']

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_export_metrics(self, config_file, temp_dir):
        """Test metrics export"""
        monitor = PerformanceMonitor(config_file, local_mode=True)
        
        # Add some test metrics
        monitor.metrics_history = {
            'system': [{'cpu_percent': 45.0}],
            'api': [{'total_requests': 100}]
        }
        monitor.alert_history = [{'type': 'test_alert', 'severity': 'warning'}]
        
        output_path = os.path.join(temp_dir, 'test_export.json')
        result_path = monitor.export_metrics(output_path)
        
        assert result_path == output_path
        assert os.path.exists(output_path)
        
        # Check export content
        with open(output_path, 'r') as f:
            export_data = json.load(f)
            assert 'export_timestamp' in export_data
            assert 'metrics_history' in export_data
            assert 'alert_history' in export_data


class TestDriftDetector:
    """Unit tests for DriftDetector class"""

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_drift_detector_initialization(self, config_file):
        """Test DriftDetector initialization"""
        detector = DriftDetector(config_file, local_mode=True)
        assert detector.config is not None
        assert detector.local_mode is True
        assert detector.aws_enabled is False
        assert hasattr(detector, 'drift_threshold')
        assert hasattr(detector, 'performance_threshold')

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_calculate_reference_statistics(self, config_file, sample_processed_data):
        """Test reference statistics calculation"""
        detector = DriftDetector(config_file, local_mode=True)
        detector.reference_data = sample_processed_data.head(100)
        
        detector.calculate_reference_statistics()
        
        assert len(detector.reference_stats) > 0
        
        # Check structure of reference stats
        for feature, stats in detector.reference_stats.items():
            assert 'mean' in stats
            assert 'std' in stats
            assert 'median' in stats
            assert 'skewness' in stats
            assert 'kurtosis' in stats
            assert 'missing_rate' in stats
            
            # Check that all values are Python native types
            for key, value in stats.items():
                assert isinstance(value, (int, float, bool))
                assert not isinstance(value, (np.integer, np.floating, np.bool_))

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_detect_statistical_drift(self, config_file, sample_processed_data):
        """Test statistical drift detection"""
        detector = DriftDetector(config_file, local_mode=True)
        
        # Set up reference data
        detector.reference_data = sample_processed_data.head(200)
        detector.calculate_reference_statistics()
        
        # Create current data with some drift
        current_data = sample_processed_data.tail(100).copy()
        current_data['Avg_Price'] *= 1.5  # Simulate price drift
        
        drift_results = detector.detect_data_drift(current_data, method='statistical')
        
        assert isinstance(drift_results, dict)
        assert 'timestamp' in drift_results
        assert 'method' in drift_results
        assert 'overall_drift_detected' in drift_results
        assert 'drift_score' in drift_results
        assert 'feature_drift' in drift_results
        
        # Check that results are JSON serializable
        json.dumps(drift_results)  # Should not raise exception

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_detect_ks_drift(self, config_file, sample_processed_data):
        """Test KS test drift detection"""
        detector = DriftDetector(config_file, local_mode=True)
        
        # Set up reference data
        detector.reference_data = sample_processed_data.head(200)
        
        # Create current data
        current_data = sample_processed_data.tail(100)
        
        drift_results = detector.detect_data_drift(current_data, method='ks_test')
        
        assert isinstance(drift_results, dict)
        assert drift_results['method'] == 'ks_test'
        assert 'feature_drift' in drift_results
        
        # Check KS test specific fields
        if drift_results['feature_drift']:
            for feature, drift_info in drift_results['feature_drift'].items():
                assert 'p_value' in drift_info
                assert 'test' in drift_info
                assert drift_info['test'] == 'kolmogorov_smirnov'

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_detect_psi_drift(self, config_file, sample_processed_data):
        """Test PSI drift detection"""
        detector = DriftDetector(config_file, local_mode=True)
        
        # Set up reference data
        detector.reference_data = sample_processed_data.head(200)
        
        # Create current data
        current_data = sample_processed_data.tail(100)
        
        drift_results = detector.detect_data_drift(current_data, method='population_stability')
        
        assert isinstance(drift_results, dict)
        assert drift_results['method'] == 'population_stability'
        assert 'feature_drift' in drift_results
        
        # Check PSI specific fields
        if drift_results['feature_drift']:
            for feature, drift_info in drift_results['feature_drift'].items():
                assert 'test' in drift_info
                assert drift_info['test'] == 'population_stability_index'
                assert 'interpretation' in drift_info

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_calculate_psi(self, config_file):
        """Test PSI calculation"""
        detector = DriftDetector(config_file, local_mode=True)
        
        # Create reference and current data
        np.random.seed(42)
        reference = pd.Series(np.random.normal(10, 2, 1000))
        current = pd.Series(np.random.normal(12, 2, 500))  # Shifted distribution
        
        psi_score = detector._calculate_psi(reference, current)
        
        assert isinstance(psi_score, float)
        assert psi_score >= 0
        assert np.isfinite(psi_score)

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_detect_model_performance_drift(self, config_file):
        """Test model performance drift detection"""
        detector = DriftDetector(config_file, local_mode=True)
        
        # Create mock prediction data
        predictions = [15.2, 18.7, 12.3, 20.1, 16.8]
        actual_values = [15.0, 19.0, 12.0, 20.5, 17.0]
        
        drift_result = detector.detect_model_performance_drift(
            predictions, actual_values, "test_model"
        )
        
        assert isinstance(drift_result, dict)
        assert 'model_name' in drift_result
        assert 'performance_drift_detected' in drift_result
        assert 'current_performance' in drift_result
        assert 'sample_size' in drift_result
        
        # Check performance metrics
        current_perf = drift_result['current_performance']
        assert 'mae' in current_perf
        assert 'mse' in current_perf
        assert 'mape' in current_perf
        assert 'rmse' in current_perf

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_send_drift_alert(self, config_file, temp_dir):
        """Test drift alert sending"""
        detector = DriftDetector(config_file, local_mode=True)
        detector.reports_dir = temp_dir
        
        test_drift_results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'statistical',
            'overall_drift_detected': True,
            'drift_score': 0.45,
            'summary': {
                'total_features': 20,
                'drifted_features': 8,
                'drift_ratio': 0.4
            }
        }
        
        detector.send_drift_alert(test_drift_results, "test_drift")
        
        # Check alert file was created
        alert_files = [f for f in os.listdir(temp_dir) if f.startswith('alert_')]
        assert len(alert_files) > 0

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_save_and_load_drift_state(self, config_file, temp_dir):
        """Test drift state save/load"""
        detector = DriftDetector(config_file, local_mode=True)
        detector.state_dir = temp_dir
        
        # Set up some state
        detector.reference_stats = {'feature1': {'mean': 10.0, 'std': 2.0}}
        detector.baseline_performance = {'model1': {'mape': 12.5}}
        
        # Save state
        state_path = os.path.join(temp_dir, 'test_state.json')
        detector.save_drift_state(state_path)
        
        assert os.path.exists(state_path)
        
        # Create new detector and load state
        new_detector = DriftDetector(config_file, local_mode=True)
        success = new_detector.load_drift_state(state_path)
        
        assert success is True
        assert len(new_detector.reference_stats) > 0
        assert 'feature1' in new_detector.reference_stats

    @pytest.mark.unit
    @pytest.mark.monitoring
    def test_export_drift_state(self, config_file, temp_dir):
        """Test drift state export"""
        detector = DriftDetector(config_file, local_mode=True)
        
        # Set up some state
        detector.reference_stats = {'feature1': {'mean': 10.0, 'std': 2.0}}
        detector.baseline_performance = {'model1': {'mape': 12.5}}
        
        output_path = os.path.join(temp_dir, 'drift_export.json')
        result_path = detector.export_drift_state(output_path)
        
        assert result_path == output_path
        assert os.path.exists(output_path)
        
        # Check export content
        with open(output_path, 'r') as f:
            export_data = json.load(f)
            assert 'export_timestamp' in export_data
            assert 'reference_stats' in export_data
            assert 'baseline_performance' in export_data
            assert 'configuration' in export_data


class TestUtilityFunctions:
    """Unit tests for utility functions"""

    @pytest.mark.unit
    def test_ensure_python_type(self):
        """Test ensure_python_type function"""
        # Test numpy types
        assert ensure_python_type(np.int64(42)) == 42
        assert ensure_python_type(np.float64(3.14)) == 3.14
        assert ensure_python_type(np.bool_(True)) is True
        
        # Test numpy arrays
        result = ensure_python_type(np.array([1, 2, 3]))
        assert result == [1, 2, 3]
        
        # Test dictionaries
        input_dict = {'a': np.int64(1), 'b': np.float64(2.5)}
        result = ensure_python_type(input_dict)
        assert result == {'a': 1, 'b': 2.5}
        
        # Test lists
        input_list = [np.int64(1), np.float64(2.5), np.bool_(False)]
        result = ensure_python_type(input_list)
        assert result == [1, 2.5, False]
        
        # Test None and NaN
        assert ensure_python_type(None) is None
        assert ensure_python_type(np.nan) is None
        
        # Test regular Python types (should pass through)
        assert ensure_python_type(42) == 42
        assert ensure_python_type(3.14) == 3.14
        assert ensure_python_type("hello") == "hello"

    @pytest.mark.unit
    def test_safe_float(self):
        """Test safe_float function"""
        # Test valid conversions
        assert safe_float(42) == 42.0
        assert safe_float(3.14) == 3.14
        assert safe_float("2.5") == 2.5
        assert safe_float(np.float64(1.23)) == 1.23
        
        # Test invalid values
        assert safe_float("invalid") == 0.0
        assert safe_float(None) == 0.0
        assert safe_float(np.inf) == 0.0
        assert safe_float(-np.inf) == 0.0
        assert safe_float(np.nan) == 0.0
        
        # Test with custom default
        assert safe_float("invalid", default=99.9) == 99.9

    @pytest.mark.unit
    def test_safe_bool(self):
        """Test safe_bool function"""
        # Test valid conversions
        assert safe_bool(True) is True
        assert safe_bool(False) is False
        assert safe_bool(1) is True
        assert safe_bool(0) is False
        assert safe_bool(np.bool_(True)) is True
        assert safe_bool(np.int64(1)) is True
        assert safe_bool(2.5) is True
        assert safe_bool(0.0) is False
        
        # Test invalid values
        assert safe_bool(np.inf) is False
        assert safe_bool(np.nan) is False
        assert safe_bool(None) is False
        
        # Test strings
        assert safe_bool("true") is True
        assert safe_bool("") is False
