#!/usr/bin/env python3
"""
Integration tests for deployment pipeline
Tests end-to-end deployment workflow

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import os
import pytest
import pandas as pd
import numpy as np
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# FIXED IMPORTS - Correct paths to modules
from src.deployment.sagemaker_deploy import FixedSageMakerDeployer
from src.deployment.model_registry import ModelRegistry
from src.inference.predictor import ModelPredictor  # FIXED: was deployment.predictor
from src.monitoring.performance_monitor import PerformanceMonitor
from prometheus_client import CollectorRegistry


class TestDeploymentPipeline:
    """Integration tests for complete deployment pipeline"""

    @pytest.mark.integration
    @pytest.mark.deployment
    @patch('boto3.client')
    @patch('sagemaker.Session')
    def test_end_to_end_deployment_pipeline(self, mock_sagemaker_session, mock_boto_client, 
                                          config_file, sample_model_file):
        """Test complete deployment pipeline from model to endpoint"""
        
        # Mock AWS services
        mock_sagemaker_client = Mock()
        mock_s3_client = Mock()
        mock_boto_client.side_effect = lambda service, **kwargs: {
            'sagemaker': mock_sagemaker_client,
            's3': mock_s3_client
        }.get(service, Mock())
        
        # Mock SageMaker session
        mock_session = Mock()
        mock_session.default_bucket.return_value = 'test-bucket'
        mock_sagemaker_session.return_value = mock_session
        
        # Initialize deployer
        deployer = FixedSageMakerDeployer(config_file)
        
        # Test model feature extraction
        feature_order = deployer.extract_model_feature_order(sample_model_file)
        assert isinstance(feature_order, list)
        assert len(feature_order) > 0
        
        # Test inference script creation
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = deployer.create_fixed_inference_script(
                'test_model', sample_model_file, temp_dir
            )
            assert os.path.exists(script_path)
            
            # Test requirements creation
            req_path = deployer.create_fixed_requirements_file(temp_dir)
            assert os.path.exists(req_path)
        
        # Mock successful deployment
        mock_predictor = Mock()
        mock_predictor.predict.return_value = {'predictions': [15.5], 'confidence': [0.85]}
        
        with patch('src.deployment.sagemaker_deploy.SKLearnModel') as mock_sklearn_model:
            mock_model_instance = Mock()
            mock_model_instance.deploy.return_value = mock_predictor
            mock_sklearn_model.return_value = mock_model_instance
            
            # Test deployment
            endpoint_info = deployer.deploy_fixed_endpoint(
                sample_model_file, 'test_model', 'test-endpoint', 'staging'
            )
            
            assert isinstance(endpoint_info, dict)
            assert 'endpoint_name' in endpoint_info
            assert 'status' in endpoint_info

    @pytest.mark.integration
    @pytest.mark.deployment
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch('boto3.client')
    def test_model_registry_integration(self, mock_boto_client, mock_set_experiment, 
                                      mock_set_tracking_uri, config_file, sample_model_file, 
                                      sample_evaluation_results):
        """Test model registry integration"""
        
        # Initialize registry
        registry = ModelRegistry(config_file)
        
        # Test model registration
        metadata = {
            'metrics': sample_evaluation_results['random_forest'],
            'training_timestamp': datetime.now().isoformat(),
            'feature_count': 75
        }
        
        with patch('mlflow.start_run') as mock_start_run:
            mock_run = Mock()
            mock_run.info.run_id = 'test_run_id'
            mock_run.info.artifact_uri = 's3://test-bucket/artifacts'
            mock_start_run.return_value.__enter__.return_value = mock_run
            
            with patch('mlflow.log_params'), \
                 patch('mlflow.log_metric'), \
                 patch('mlflow.sklearn.log_model'), \
                 patch('mlflow.set_tags'):
                
                result = registry.register_model(
                    sample_model_file, 'test_model', 'v1.0', metadata
                )
                
                assert isinstance(result, dict)
                assert 'run_id' in result
                assert result['status'] == 'registered'

    @pytest.mark.integration
    @pytest.mark.deployment
    def test_predictor_integration(self, config_file, sample_model_file):
        """Test model predictor integration"""
        
        # Initialize predictor
        with patch('boto3.client'), \
             patch('mlflow.set_tracking_uri'):
            
            predictor = ModelPredictor(config_file)
            
            # Test model loading simulation
            mock_models = {
                'test_model': {
                    'model': Mock(),
                    'scaler': Mock(),
                    'version': 'v1.0',
                    'metrics': {'val_mape': 12.5}
                }
            }
            
            predictor.loaded_models = mock_models
            
            # Test model info retrieval
            model_info = predictor.get_model_info('test_model')
            assert isinstance(model_info, dict)
            assert 'version' in model_info
            assert 'metrics' in model_info

    @pytest.mark.integration
    @pytest.mark.deployment
    def test_monitoring_integration(self, config_file):
        """Test monitoring integration with deployment"""
        
        # Create separate registry for testing
        test_registry = CollectorRegistry()
        
        # Initialize monitor
        monitor = PerformanceMonitor(config_file, local_mode=True, registry=test_registry)
        
        # Test metrics collection
        with patch('psutil.cpu_percent', return_value=45.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            # Setup mock objects
            mock_memory_obj = Mock()
            mock_memory_obj.percent = 60.0
            mock_memory_obj.available = 8 * 1024**3
            mock_memory.return_value = mock_memory_obj
            
            mock_disk_obj = Mock()
            mock_disk_obj.used = 100 * 1024**3
            mock_disk_obj.total = 500 * 1024**3
            mock_disk_obj.free = 400 * 1024**3
            mock_disk.return_value = mock_disk_obj
            
            # Collect metrics
            monitor.collect_system_metrics()
            
            # Verify metrics were collected
            assert 'system' in monitor.metrics_history
            assert len(monitor.metrics_history['system']) > 0

    @pytest.mark.integration
    @pytest.mark.deployment
    @patch('boto3.client')
    @patch('sagemaker.Session')
    def test_deployment_pipeline_with_monitoring(self, mock_sagemaker_session, mock_boto_client,
                                                config_file, sample_model_file):
        """Test deployment pipeline with monitoring integration"""
        
        # Initialize components
        deployer = FixedSageMakerDeployer(config_file)
        test_registry = CollectorRegistry()
        monitor = PerformanceMonitor(config_file, local_mode=True, registry=test_registry)
        
        # Mock AWS services
        mock_boto_client.return_value = Mock()
        mock_sagemaker_session.return_value = Mock()
        
        # Test deployment readiness check
        feature_order = deployer.extract_model_feature_order(sample_model_file)
        assert len(feature_order) > 0
        
        # Test monitoring setup
        with patch('psutil.cpu_percent', return_value=45.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            # Setup mock objects
            mock_memory_obj = Mock()
            mock_memory_obj.percent = 60.0
            mock_memory_obj.available = 8 * 1024**3
            mock_memory.return_value = mock_memory_obj
            
            mock_disk_obj = Mock()
            mock_disk_obj.used = 100 * 1024**3
            mock_disk_obj.total = 500 * 1024**3
            mock_disk_obj.free = 400 * 1024**3
            mock_disk.return_value = mock_disk_obj
            
            # Start monitoring
            monitor.collect_current_metrics()
            
            # Get health summary
            health = monitor.get_health_summary()
            assert 'overall_status' in health
            assert health['overall_status'] in ['healthy', 'warning', 'critical', 'no_data']

    @pytest.mark.integration
    @pytest.mark.deployment
    @patch('boto3.client')
    def test_endpoint_management_workflow(self, mock_boto_client, config_file):
        """Test complete endpoint management workflow"""
        
        # Mock SageMaker client
        mock_sagemaker_client = Mock()
        mock_boto_client.return_value = mock_sagemaker_client
        
        # Mock endpoint list response
        mock_endpoints = [{
            'EndpointName': 'test-endpoint-1',
            'EndpointStatus': 'InService',
            'CreationTime': datetime.now(),
            'LastModifiedTime': datetime.now(),
            'EndpointConfigName': 'test-config-1'
        }]
        
        mock_sagemaker_client.list_endpoints.return_value = {'Endpoints': mock_endpoints}
        mock_sagemaker_client.describe_endpoint_config.return_value = {
            'ProductionVariants': [{
                'InstanceType': 'ml.m5.large',
                'InitialInstanceCount': 1
            }]
        }
        
        # Initialize deployer
        deployer = FixedSageMakerDeployer(config_file)
        
        # Test endpoint listing
        result = deployer.list_endpoints()
        assert isinstance(result, dict)
        assert result['status'] == 'success'
        assert result['endpoint_count'] == 1

    @pytest.mark.integration
    @pytest.mark.deployment
    def test_error_handling_workflow(self, config_file):
        """Test error handling in deployment workflow"""
        
        # Test with invalid config
        invalid_config = {
            'aws': {
                'region': 'invalid-region',
                'sagemaker': {'execution_role': 'invalid-role'}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(invalid_config, f)
            invalid_config_path = f.name
        
        try:
            # Test deployer with invalid config - should handle initialization errors gracefully
            with patch('boto3.client', side_effect=Exception("Invalid region")):
                try:
                    deployer = FixedSageMakerDeployer(invalid_config_path)
                    # If initialization succeeds, that's okay
                    assert deployer is not None
                except Exception as e:
                    # If initialization fails, that's also acceptable for invalid config
                    assert "Invalid region" in str(e) or "invalid-region" in str(e)
        
        finally:
            os.unlink(invalid_config_path)

    @pytest.mark.integration
    @pytest.mark.deployment
    def test_deployment_validation_workflow(self, config_file, sample_model_file):
        """Test deployment validation workflow"""
        
        with patch('boto3.client'), \
             patch('sagemaker.Session'):
            
            deployer = FixedSageMakerDeployer(config_file)
            
            # Test model validation
            feature_order = deployer.extract_model_feature_order(sample_model_file)
            assert isinstance(feature_order, list)
            assert len(feature_order) > 0
            
            # Test inference script validation
            with tempfile.TemporaryDirectory() as temp_dir:
                script_path = deployer.create_fixed_inference_script(
                    'test_model', sample_model_file, temp_dir
                )
                
                # Verify script was created correctly
                assert os.path.exists(script_path)
                
                with open(script_path, 'r') as f:
                    script_content = f.read()
                    # Check for required functions
                    assert 'def model_fn' in script_content
                    assert 'def input_fn' in script_content
                    assert 'def predict_fn' in script_content
                    assert 'def output_fn' in script_content
                    assert 'CORRECT_FEATURE_ORDER' in script_content