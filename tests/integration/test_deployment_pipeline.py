#!/usr/bin/env python3
"""
Integration Tests for Deployment Pipeline
Tests the interaction between model deployment, SageMaker endpoints, and API components
"""

import os
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np
import yaml
import json
import joblib
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
import asyncio

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from deployment.sagemaker_deploy import FixedSageMakerDeployer
from inference.api import SageMakerSyncAPI
from deployment.predictor import ModelPredictor


@pytest.fixture
def temp_config():
    """Create temporary config file for testing"""
    config_data = {
        'project': {
            'name': 'test-project',
            'version': '1.0.0'
        },
        'aws': {
            'region': 'us-east-1',
            's3': {'bucket_name': 'test-bucket'},
            'sagemaker': {
                'execution_role': 'arn:aws:iam::123456789:role/test-role'
            }
        },
        'deployment': {
            'environments': {
                'dev': {
                    'instance_type': 'ml.t2.medium',
                    'initial_instance_count': 1,
                    'auto_scaling_enabled': False
                },
                'staging': {
                    'instance_type': 'ml.m5.large',
                    'initial_instance_count': 1,
                    'auto_scaling_enabled': True
                }
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        return f.name


@pytest.fixture
def temp_model():
    """Create a temporary trained model for testing"""
    temp_dir = tempfile.mkdtemp()
    
    # Create a simple mock model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    model = RandomForestRegressor(n_estimators=5, random_state=42)
    scaler = StandardScaler()
    
    # Create dummy training data
    X_dummy = np.random.randn(50, 10)
    y_dummy = np.random.randn(50)
    
    # Fit the model
    X_scaled = scaler.fit_transform(X_dummy)
    model.fit(X_scaled, y_dummy)
    
    # Save model artifact
    model_artifact = {
        'model': model,
        'scaler': scaler
    }
    
    model_path = os.path.join(temp_dir, 'test_model.pkl')
    joblib.dump(model_artifact, model_path)
    
    yield model_path, temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_prediction_data():
    """Create sample data for prediction testing"""
    return {
        'Total_Quantity': 150.0,
        'Avg_Price': 18.5,
        'Transaction_Count': 25,
        'Month': 7,
        'DayOfWeek': 1,
        'IsWeekend': 0,
        'Price_Volatility': 1.2,
        'Wholesale_Price': 14.0,
        'Loss_Rate': 8.5
    }


class TestDeploymentPipelineIntegration:
    """Integration tests for the complete deployment pipeline"""

    @patch('boto3.Session')
    @patch('sagemaker.Session')
    def test_sagemaker_deployer_initialization(self, mock_sagemaker_session, mock_boto_session, temp_config):
        """Test SageMaker deployer can be initialized with proper configuration"""
        
        # Mock AWS clients
        mock_s3 = MagicMock()
        mock_sagemaker = MagicMock()
        mock_boto_session.return_value.client.side_effect = lambda service, **kwargs: {
            's3': mock_s3,
            'sagemaker': mock_sagemaker
        }.get(service, MagicMock())
        
        mock_sagemaker_session.return_value.default_bucket.return_value = 'test-bucket'
        
        # Initialize deployer
        deployer = FixedSageMakerDeployer(temp_config)
        
        # Verify initialization
        assert deployer.aws_config is not None
        assert deployer.deployment_config is not None
        assert deployer.region == 'us-east-1'
        assert deployer.bucket == 'test-bucket'

    @patch('boto3.client')
    def test_endpoint_listing_integration(self, mock_boto_client, temp_config):
        """Test endpoint listing functionality"""
        
        # Mock SageMaker client
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        
        # Mock list_endpoints response
        mock_sagemaker.list_endpoints.return_value = {
            'Endpoints': [
                {
                    'EndpointName': 'test-endpoint-1',
                    'EndpointStatus': 'InService',
                    'CreationTime': datetime.now(),
                    'LastModifiedTime': datetime.now(),
                    'EndpointConfigName': 'test-config'
                }
            ]
        }
        
        # Mock describe_endpoint_config response
        mock_sagemaker.describe_endpoint_config.return_value = {
            'ProductionVariants': [{
                'InstanceType': 'ml.t2.medium',
                'InitialInstanceCount': 1
            }]
        }
        
        # Test listing
        deployer = FixedSageMakerDeployer(temp_config)
        result = deployer.list_endpoints()
        
        # Verify results
        assert result['status'] == 'success'
        assert result['endpoint_count'] == 1
        assert len(result['endpoints']) == 1
        assert result['endpoints'][0]['name'] == 'test-endpoint-1'

    def test_api_model_loading_integration(self, temp_config, temp_model):
        """Test API can load and work with models"""
        
        model_path, model_dir = temp_model
        
        # Copy model to expected location
        models_dir = os.path.join(os.path.dirname(model_dir), 'models')
        os.makedirs(models_dir, exist_ok=True)
        shutil.copy(model_path, os.path.join(models_dir, 'best_model.pkl'))
        
        # Initialize API with model directory context
        with patch('os.path.exists') as mock_exists:
            mock_exists.side_effect = lambda path: 'models' in path and 'pkl' in path
            
            with patch('os.listdir') as mock_listdir:
                mock_listdir.return_value = ['best_model.pkl']
                
                with patch('joblib.load') as mock_load:
                    # Load the actual model for testing
                    actual_artifact = joblib.load(model_path)
                    mock_load.return_value = actual_artifact
                    
                    api = SageMakerSyncAPI(temp_config)
                    
                    # Verify models were loaded
                    assert len(api.models) > 0
                    assert 'best_model' in api.models
                    assert len(api.model_feature_orders) > 0

    def test_api_prediction_integration(self, temp_config, temp_model, sample_prediction_data):
        """Test API prediction functionality with real model"""
        
        model_path, model_dir = temp_model
        
        # Initialize API with mocked model loading
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=['test_model.pkl']), \
             patch('joblib.load') as mock_load:
            
            # Load actual model
            actual_artifact = joblib.load(model_path)
            mock_load.return_value = actual_artifact
            
            api = SageMakerSyncAPI(temp_config)
            
            # Create feature input
            from inference.api import FeatureInput
            features = FeatureInput(**sample_prediction_data)
            
            # Make prediction
            result = api.predict_single(features, 'test_model')
            
            # Verify prediction result
            assert result.predicted_price > 0
            assert 0 <= result.confidence <= 1
            assert result.model_used == 'test_model'
            assert result.features_engineered > 0

    def test_api_batch_prediction_integration(self, temp_config, temp_model, sample_prediction_data):
        """Test API batch prediction functionality"""
        
        model_path, model_dir = temp_model
        
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=['test_model.pkl']), \
             patch('joblib.load') as mock_load:
            
            actual_artifact = joblib.load(model_path)
            mock_load.return_value = actual_artifact
            
            api = SageMakerSyncAPI(temp_config)
            
            # Create batch input
            from inference.api import FeatureInput, BatchFeatureInput
            
            # Create multiple feature instances
            features_list = []
            for i in range(3):
                modified_data = sample_prediction_data.copy()
                modified_data['Total_Quantity'] = sample_prediction_data['Total_Quantity'] + i * 10
                features_list.append(FeatureInput(**modified_data))
            
            batch_input = BatchFeatureInput(
                instances=features_list,
                model_name='test_model'
            )
            
            # Make batch prediction
            result = api.predict_batch(batch_input)
            
            # Verify batch result
            assert len(result.predictions) == 3
            assert result.batch_id.startswith('batch_')
            assert result.processing_time_ms > 0
            assert all(pred.predicted_price > 0 for pred in result.predictions)

    @patch('boto3.client')
    def test_predictor_model_loading_integration(self, mock_boto_client, temp_config, temp_model):
        """Test ModelPredictor integration with model loading"""
        
        model_path, model_dir = temp_model
        
        # Mock AWS clients
        mock_boto_client.return_value = MagicMock()
        
        # Mock MLflow components
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'), \
             patch('mlflow.tracking.MlflowClient') as mock_client:
            
            # Initialize predictor
            predictor = ModelPredictor(temp_config)
            
            # Test local model loading
            models = predictor.load_local_models()
            
            # Mock local models directory
            with patch('os.path.exists', return_value=True), \
                 patch('os.listdir', return_value=['test_model.pkl']), \
                 patch('joblib.load') as mock_load:
                
                actual_artifact = joblib.load(model_path)
                mock_load.return_value = actual_artifact
                
                models = predictor.load_local_models()
                
                # Verify models loaded
                assert len(models) > 0

    def test_deployment_configuration_validation(self, temp_config):
        """Test deployment configuration is properly validated"""
        
        # Load config and validate structure
        with open(temp_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verify deployment configuration exists
        assert 'deployment' in config
        assert 'environments' in config['deployment']
        
        # Verify environment configurations
        envs = config['deployment']['environments']
        assert 'dev' in envs
        assert 'staging' in envs
        
        # Verify each environment has required fields
        for env_name, env_config in envs.items():
            assert 'instance_type' in env_config
            assert 'initial_instance_count' in env_config
            assert isinstance(env_config['auto_scaling_enabled'], bool)

    @patch('requests.get')
    def test_api_health_check_integration(self, mock_get, temp_config, temp_model):
        """Test API health check functionality"""
        
        model_path, model_dir = temp_model
        
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=['test_model.pkl']), \
             patch('joblib.load') as mock_load:
            
            actual_artifact = joblib.load(model_path)
            mock_load.return_value = actual_artifact
            
            api = SageMakerSyncAPI(temp_config)
            
            # Mock successful health response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'status': 'healthy',
                'models_loaded': len(api.models),
                'uptime_seconds': 100
            }
            mock_get.return_value = mock_response
            
            # Test model list endpoint
            model_list = api.get_model_list()
            
            # Verify response
            assert len(model_list) > 0
            assert 'test_model' in model_list or 'best_model' in model_list

    def test_feature_engineering_compatibility(self, temp_config, sample_prediction_data):
        """Test that deployment components work with feature engineering output"""
        
        # Create mock feature engineering output
        feature_columns = [
            'Total_Quantity', 'Avg_Price', 'Transaction_Count', 'Month', 'DayOfWeek',
            'IsWeekend', 'Price_Volatility', 'Month_Sin', 'Month_Cos', 'Revenue',
            'Price_Quantity_Interaction', 'Wholesale_Price', 'Loss_Rate'
        ]
        
        # Test feature order extraction function
        from inference.api import get_default_feature_order
        default_order = get_default_feature_order()
        
        # Verify default order contains essential features
        essential_features = ['Total_Quantity', 'Avg_Price', 'Month', 'DayOfWeek']
        for feature in essential_features:
            assert feature in default_order

    def test_error_handling_integration(self, temp_config):
        """Test error handling across deployment components"""
        
        # Test API with no models
        with patch('os.path.exists', return_value=False):
            api = SageMakerSyncAPI(temp_config)
            
            # Should gracefully handle missing models
            assert len(api.models) > 0  # Should create mock model
            assert 'mock_model' in api.models

    @patch('boto3.Session')
    def test_deployment_aws_integration_mock(self, mock_session, temp_config, temp_model):
        """Test deployment components integrate properly with AWS (mocked)"""
        
        model_path, model_dir = temp_model
        
        # Mock AWS services
        mock_s3 = MagicMock()
        mock_sagemaker = MagicMock()
        mock_session.return_value.client.side_effect = lambda service, **kwargs: {
            's3': mock_s3,
            'sagemaker': mock_sagemaker
        }.get(service, MagicMock())
        
        mock_session.return_value.default_bucket.return_value = 'test-bucket'
        
        # Test deployer initialization
        deployer = FixedSageMakerDeployer(temp_config)
        
        # Test feature order extraction
        feature_order = deployer.extract_model_feature_order(model_path)
        
        # Verify feature order is valid
        assert isinstance(feature_order, list)
        assert len(feature_order) > 0

    def test_model_version_compatibility(self, temp_config, temp_model):
        """Test that different components handle model versioning consistently"""
        
        model_path, model_dir = temp_model
        
        # Test API version handling
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=['test_model.pkl']), \
             patch('joblib.load') as mock_load:
            
            actual_artifact = joblib.load(model_path)
            mock_load.return_value = actual_artifact
            
            api = SageMakerSyncAPI(temp_config)
            
            # Verify version information is handled
            assert hasattr(api, 'app_version')
            assert api.app_version is not None

    def test_inference_script_generation(self, temp_config, temp_model):
        """Test that inference script generation works correctly"""
        
        model_path, model_dir = temp_model
        
        # Mock deployer
        with patch('boto3.Session'):
            deployer = FixedSageMakerDeployer(temp_config)
            
            # Test inference script creation
            output_dir = tempfile.mkdtemp()
            try:
                script_path = deployer.create_fixed_inference_script(
                    'test_model', model_path, output_dir
                )
                
                # Verify script was created
                assert os.path.exists(script_path)
                
                # Verify script contains expected components
                with open(script_path, 'r') as f:
                    script_content = f.read()
                
                assert 'def model_fn' in script_content
                assert 'def input_fn' in script_content
                assert 'def predict_fn' in script_content
                assert 'def output_fn' in script_content
                assert 'CORRECT_FEATURE_ORDER' in script_content
                
            finally:
                shutil.rmtree(output_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
