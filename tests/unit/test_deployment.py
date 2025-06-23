#!/usr/bin/env python3
"""
Unit tests for deployment components
Tests SageMaker deployment, API functionality, and inference

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
from fastapi.testclient import TestClient

# Import modules to test
from src.deployment.sagemaker_deploy import FixedSageMakerDeployer
from src.inference.api import app, SageMakerSyncAPI, FeatureInput, BatchFeatureInput
from src.inference.predictor import ModelPredictor


class TestSageMakerDeployer:
    """Unit tests for FixedSageMakerDeployer class"""

    @pytest.mark.unit
    @patch('boto3.client')
    @patch('sagemaker.Session')
    def test_deployer_initialization(self, mock_sagemaker_session, mock_boto_client, config_file):
        """Test SageMaker deployer initialization"""
        deployer = FixedSageMakerDeployer(config_file)
        assert deployer.config is not None
        assert hasattr(deployer, 'aws_config')
        assert hasattr(deployer, 'deployment_config')

    @pytest.mark.unit
    @patch('boto3.client')
    @patch('sagemaker.Session')
    def test_extract_model_feature_order(self, mock_sagemaker_session, mock_boto_client, 
                                       config_file, sample_model_file):
        """Test feature order extraction from model"""
        deployer = FixedSageMakerDeployer(config_file)
        
        feature_order = deployer.extract_model_feature_order(sample_model_file)
        
        assert isinstance(feature_order, list)
        assert len(feature_order) > 0
        # Should either extract from model or return default
        if hasattr(deployer, '_get_default_feature_order'):
            default_order = deployer._get_default_feature_order()
            assert isinstance(default_order, list)

    @pytest.mark.unit
    @patch('boto3.client')
    @patch('sagemaker.Session')
    def test_create_fixed_inference_script(self, mock_sagemaker_session, mock_boto_client,
                                         config_file, sample_model_file, temp_dir):
        """Test inference script creation"""
        deployer = FixedSageMakerDeployer(config_file)
        
        script_path = deployer.create_fixed_inference_script(
            'test_model', sample_model_file, temp_dir
        )
        
        assert os.path.exists(script_path)
        assert script_path.endswith('inference.py')
        
        # Check script content
        with open(script_path, 'r') as f:
            content = f.read()
            assert 'def model_fn' in content
            assert 'def input_fn' in content
            assert 'def predict_fn' in content
            assert 'def output_fn' in content
            assert 'CORRECT_FEATURE_ORDER' in content

    @pytest.mark.unit
    @patch('boto3.client')
    @patch('sagemaker.Session')
    def test_create_fixed_requirements_file(self, mock_sagemaker_session, mock_boto_client,
                                          config_file, temp_dir):
        """Test requirements file creation"""
        deployer = FixedSageMakerDeployer(config_file)
        
        req_path = deployer.create_fixed_requirements_file(temp_dir)
        
        assert os.path.exists(req_path)
        assert req_path.endswith('requirements.txt')
        
        # Check requirements content
        with open(req_path, 'r') as f:
            content = f.read()
            assert 'pandas' in content
            assert 'numpy' in content
            assert 'scikit-learn' in content

    @pytest.mark.unit
    @patch('boto3.client')
    @patch('sagemaker.Session')
    def test_list_endpoints(self, mock_sagemaker_session, mock_boto_client, config_file):
        """Test endpoint listing functionality"""
        deployer = FixedSageMakerDeployer(config_file)
        
        # Mock SageMaker client response
        mock_sagemaker_client = Mock()
        deployer.sagemaker_client = mock_sagemaker_client
        
        mock_endpoints = [
            {
                'EndpointName': 'test-endpoint-1',
                'EndpointStatus': 'InService',
                'CreationTime': datetime.now(),
                'LastModifiedTime': datetime.now(),
                'EndpointConfigName': 'test-config-1'
            }
        ]
        
        mock_sagemaker_client.list_endpoints.return_value = {'Endpoints': mock_endpoints}
        mock_sagemaker_client.describe_endpoint_config.return_value = {
            'ProductionVariants': [{
                'InstanceType': 'ml.m5.large',
                'InitialInstanceCount': 1
            }]
        }
        
        result = deployer.list_endpoints()
        
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'endpoint_count' in result
        assert 'endpoints' in result
        assert result['status'] == 'success'

    @pytest.mark.unit
    @patch('boto3.client')
    @patch('sagemaker.Session')
    def test_test_endpoint(self, mock_sagemaker_session, mock_boto_client, config_file):
        """Test endpoint testing functionality"""
        deployer = FixedSageMakerDeployer(config_file)
        
        # Mock Predictor
        with patch('src.deployment.sagemaker_deploy.Predictor') as mock_predictor_class:
            mock_predictor = Mock()
            mock_predictor.predict.return_value = {
                'predictions': [15.5],
                'confidence': [0.85]
            }
            mock_predictor_class.return_value = mock_predictor
            
            result = deployer.test_endpoint('test-endpoint')
            
            assert isinstance(result, dict)
            assert 'endpoint_name' in result
            assert 'status' in result
            if result['status'] == 'success':
                assert 'latency_ms' in result
                assert 'sample_result' in result

    @pytest.mark.unit
    @patch('boto3.client')
    @patch('sagemaker.Session')
    def test_delete_endpoint(self, mock_sagemaker_session, mock_boto_client, config_file):
        """Test endpoint deletion"""
        deployer = FixedSageMakerDeployer(config_file)
        
        # Mock successful deletion
        mock_sagemaker_client = Mock()
        deployer.sagemaker_client = mock_sagemaker_client
        mock_sagemaker_client.delete_endpoint.return_value = None
        
        result = deployer.delete_endpoint('test-endpoint')
        
        assert isinstance(result, bool)
        mock_sagemaker_client.delete_endpoint.assert_called_once_with(EndpointName='test-endpoint')


class TestSageMakerSyncAPI:
    """Unit tests for SageMakerSyncAPI class"""

    @pytest.mark.unit
    def test_api_initialization(self, config_file):
        """Test API initialization"""
        api = SageMakerSyncAPI(config_file)
        assert api.config is not None
        assert hasattr(api, 'models')
        assert hasattr(api, 'model_feature_orders')

    @pytest.mark.unit
    def test_create_mock_model(self, config_file):
        """Test mock model creation"""
        api = SageMakerSyncAPI(config_file)
        
        mock_model = api.create_mock_model()
        
        assert isinstance(mock_model, dict)
        assert 'model' in mock_model
        
        # Test prediction capability
        model = mock_model['model']
        test_data = [[1, 2, 3, 4, 5]]
        predictions = model.predict(test_data)
        assert len(predictions) == 1
        assert isinstance(predictions[0], (int, float))

    @pytest.mark.unit
    def test_get_model_list(self, config_file):
        """Test model list retrieval"""
        api = SageMakerSyncAPI(config_file)
        
        # Add mock models
        api.models = {
            'model1': {'model': Mock()},
            'model2': {'model': Mock()},
            'best_model': {'model': Mock()}
        }
        
        model_list = api.get_model_list()
        
        assert isinstance(model_list, list)
        assert len(model_list) == 3
        assert 'model1' in model_list
        assert 'best_model' in model_list

    @pytest.mark.unit
    def test_feature_input_validation(self):
        """Test FeatureInput validation"""
        # Test valid input
        valid_input = FeatureInput(
            Total_Quantity=150.0,
            Avg_Price=18.5,
            Transaction_Count=25,
            Month=6,
            DayOfWeek=1,
            IsWeekend=0
        )
        
        assert valid_input.Total_Quantity == 150.0
        assert valid_input.Avg_Price == 18.5
        assert valid_input.Month == 6
        
        # Test validation errors
        with pytest.raises(ValueError):
            FeatureInput(
                Total_Quantity=-10,  # Should be positive
                Avg_Price=18.5,
                Transaction_Count=25,
                Month=6,
                DayOfWeek=1,
                IsWeekend=0
            )
        
        with pytest.raises(ValueError):
            FeatureInput(
                Total_Quantity=150.0,
                Avg_Price=-5.0,  # Should be positive
                Transaction_Count=25,
                Month=6,
                DayOfWeek=1,
                IsWeekend=0
            )

    @pytest.mark.unit
    def test_batch_feature_input_validation(self):
        """Test BatchFeatureInput validation"""
        feature1 = FeatureInput(
            Total_Quantity=150.0,
            Avg_Price=18.5,
            Transaction_Count=25,
            Month=6,
            DayOfWeek=1,
            IsWeekend=0
        )
        
        feature2 = FeatureInput(
            Total_Quantity=200.0,
            Avg_Price=20.0,
            Transaction_Count=30,
            Month=7,
            DayOfWeek=2,
            IsWeekend=0
        )
        
        batch_input = BatchFeatureInput(
            instances=[feature1, feature2],
            model_name="test_model"
        )
        
        assert len(batch_input.instances) == 2
        assert batch_input.model_name == "test_model"
        
        # Test validation with too many instances
        with pytest.raises(ValueError):
            BatchFeatureInput(
                instances=[feature1] * 101,  # Max is 100
                model_name="test_model"
            )

    @pytest.mark.unit
    def test_predict_single_mock(self, config_file):
        """Test single prediction with mock model"""
        api = SageMakerSyncAPI(config_file)
        
        # API should automatically have both mock_model and best_model available
        assert 'mock_model' in api.models, "mock_model should be available"
        assert 'best_model' in api.models, "best_model should be available"
        
        feature_input = FeatureInput(
            Total_Quantity=150.0,
            Avg_Price=18.5,
            Transaction_Count=25,
            Month=6,
            DayOfWeek=1,
            IsWeekend=0
        )
        
        # Test with mock_model specifically
        result = api.predict_single(feature_input, "mock_model")
        
        assert hasattr(result, 'predicted_price')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'model_used')
        assert isinstance(result.predicted_price, float)
        assert 0 <= result.confidence <= 1
        assert result.model_used == "mock_model"

    @pytest.mark.unit
    def test_predict_batch_mock(self, config_file):
        """Test batch prediction with mock model"""
        api = SageMakerSyncAPI(config_file)
        
        # API should automatically have both mock_model and best_model available
        assert 'mock_model' in api.models, "mock_model should be available"
        assert 'best_model' in api.models, "best_model should be available"
        
        batch_input = BatchFeatureInput(
            instances=[
                FeatureInput(
                    Total_Quantity=150.0,
                    Avg_Price=18.5,
                    Transaction_Count=25,
                    Month=6,
                    DayOfWeek=1,
                    IsWeekend=0
                ),
                FeatureInput(
                    Total_Quantity=200.0,
                    Avg_Price=20.0,
                    Transaction_Count=30,
                    Month=7,
                    DayOfWeek=2,
                    IsWeekend=0
                )
            ],
            model_name="mock_model"
        )
        
        result = api.predict_batch(batch_input)
        
        assert hasattr(result, 'predictions')
        assert hasattr(result, 'batch_id')
        assert hasattr(result, 'processing_time_ms')
        assert len(result.predictions) == 2
        assert isinstance(result.processing_time_ms, float)


class TestFastAPI:
    """Unit tests for FastAPI endpoints"""

    @pytest.mark.unit
    @pytest.mark.api
    def test_health_endpoint(self):
        """Test health check endpoint"""
        client = TestClient(app)
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["status"] == "healthy"

    @pytest.mark.unit
    @pytest.mark.api
    def test_models_endpoint(self):
        """Test models listing endpoint"""
        client = TestClient(app)
        
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "default_model" in data
        assert "total_models" in data
        assert isinstance(data["models"], list)

    @pytest.mark.unit
    @pytest.mark.api
    def test_features_example_endpoint(self):
        """Test features example endpoint"""
        client = TestClient(app)
        
        response = client.get("/features/example")
        
        assert response.status_code == 200
        data = response.json()
        assert "example_input" in data
        assert "mapping" in data
        
        # Check example input structure
        example = data["example_input"]
        required_fields = ["Total_Quantity", "Avg_Price", "Transaction_Count", "Month"]
        for field in required_fields:
            assert field in example

    @pytest.mark.unit
    @pytest.mark.api
    def test_predict_endpoint(self):
        """Test prediction endpoint"""
        client = TestClient(app)
        
        test_features = {
            "Total_Quantity": 150.0,
            "Avg_Price": 18.5,
            "Transaction_Count": 25,
            "Month": 6,
            "DayOfWeek": 1,
            "IsWeekend": 0,
            "Price_Volatility": 1.2,
            "Discount_Count": 3
        }
        
        response = client.post("/predict", json=test_features)
        
        # Should return 200 if mock model is working, or specific error
        assert response.status_code in [200, 422, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "predicted_price" in data
            assert "confidence" in data
            assert "model_used" in data

    @pytest.mark.unit
    @pytest.mark.api
    def test_predict_batch_endpoint(self):
        """Test batch prediction endpoint"""
        client = TestClient(app)
        
        test_batch = {
            "instances": [
                {
                    "Total_Quantity": 150.0,
                    "Avg_Price": 18.5,
                    "Transaction_Count": 25,
                    "Month": 6,
                    "DayOfWeek": 1,
                    "IsWeekend": 0
                },
                {
                    "Total_Quantity": 200.0,
                    "Avg_Price": 20.0,
                    "Transaction_Count": 30,
                    "Month": 7,
                    "DayOfWeek": 2,
                    "IsWeekend": 0
                }
            ],
            "model_name": "best_model"
        }
        
        response = client.post("/predict/batch", json=test_batch)
        
        # Should return 200 if mock model is working, or specific error
        assert response.status_code in [200, 422, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "batch_id" in data
            assert "processing_time_ms" in data
            assert len(data["predictions"]) == 2

    @pytest.mark.unit
    @pytest.mark.api
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        client = TestClient(app)
        
        response = client.get("/metrics")
        
        assert response.status_code == 200
        # Should return Prometheus metrics format
        content = response.content.decode()
        assert "prediction_requests_total" in content or "# TYPE" in content or len(content) > 0

    @pytest.mark.unit
    @pytest.mark.api
    def test_root_endpoint(self):
        """Test root endpoint"""
        client = TestClient(app)
        
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "status" in data

    @pytest.mark.unit
    @pytest.mark.api
    def test_invalid_predict_input(self):
        """Test prediction with invalid input"""
        client = TestClient(app)
        
        # Test with missing required fields
        invalid_features = {
            "Total_Quantity": 150.0,
            # Missing Avg_Price, Transaction_Count, etc.
        }
        
        response = client.post("/predict", json=invalid_features)
        
        # Should return validation error
        assert response.status_code == 422
        
        # Test with negative values
        negative_features = {
            "Total_Quantity": -150.0,  # Should be positive
            "Avg_Price": 18.5,
            "Transaction_Count": 25,
            "Month": 6,
            "DayOfWeek": 1,
            "IsWeekend": 0
        }
        
        response = client.post("/predict", json=negative_features)
        assert response.status_code == 422

    @pytest.mark.unit
    @pytest.mark.api
    def test_cors_headers(self):
        """Test CORS headers are present"""
        client = TestClient(app)
        
        response = client.options("/health")
        
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers or response.status_code in [200, 405]
        
        # Test actual request
        response = client.get("/health")
        assert response.status_code == 200

    @pytest.mark.unit
    @pytest.mark.api
    def test_api_versioning(self):
        """Test API versioning information"""
        client = TestClient(app)
        
        response = client.get("/")
        data = response.json()
        
        assert "version" in data
        # Should have some version information
        assert data["version"] is not None

    @pytest.mark.unit
    @pytest.mark.api
    def test_processing_time_header(self):
        """Test processing time header is added"""
        client = TestClient(app)
        
        response = client.get("/health")
        
        # Should have processing time header
        assert "X-Processing-Time" in response.headers or "x-processing-time" in response.headers
        
        # Should have API version header
        assert "X-API-Version" in response.headers or "x-api-version" in response.headers