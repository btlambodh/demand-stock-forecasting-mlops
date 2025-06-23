#!/usr/bin/env python3
"""
Integration Tests for Training Pipeline
Tests the interaction between model training, evaluation, and model registry components
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
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from training.train_model import ModelTrainer
from deployment.model_registry import ModelRegistry
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
        'evaluation': {
            'thresholds': {
                'mape_threshold': 20.0,
                'rmse_threshold': 5.0,
                'r2_threshold': 0.7
            }
        },
        'models': {
            'random_forest': {
                'n_estimators': 10,  # Small for testing
                'max_depth': 3
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        return f.name


@pytest.fixture
def temp_processed_data():
    """Create temporary processed data for training"""
    temp_dir = tempfile.mkdtemp()
    
    # Create sample processed data with features and targets
    np.random.seed(42)
    n_samples = 200
    
    base_data = {
        'Date': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
        'Item Code': np.random.choice([101, 102, 103], n_samples),
        'Total_Quantity': np.random.uniform(50, 200, n_samples),
        'Avg_Price': np.random.uniform(10, 30, n_samples),
        'Transaction_Count': np.random.randint(5, 50, n_samples),
        'Price_Volatility': np.random.uniform(0.1, 2.0, n_samples),
        'Month': np.random.randint(1, 13, n_samples),
        'DayOfWeek': np.random.randint(0, 7, n_samples),
        'IsWeekend': np.random.choice([0, 1], n_samples),
        'Revenue': np.random.uniform(500, 6000, n_samples),
    }
    
    # Add some derived features
    base_data['Month_Sin'] = np.sin(2 * np.pi * np.array(base_data['Month']) / 12)
    base_data['Month_Cos'] = np.cos(2 * np.pi * np.array(base_data['Month']) / 12)
    base_data['Price_Quantity_Interaction'] = np.array(base_data['Avg_Price']) * np.array(base_data['Total_Quantity'])
    
    # Add target variables (next day price prediction)
    base_data['Avg_Price_Target_1d'] = np.array(base_data['Avg_Price']) * (1 + np.random.normal(0, 0.05, n_samples))
    
    df = pd.DataFrame(base_data)
    
    # Create train/validation/test splits
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    
    # Save datasets
    train_df.to_parquet(os.path.join(temp_dir, 'train.parquet'), index=False)
    val_df.to_parquet(os.path.join(temp_dir, 'validation.parquet'), index=False)
    test_df.to_parquet(os.path.join(temp_dir, 'test.parquet'), index=False)
    
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_model_output():
    """Create temporary directory for model outputs"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestTrainingPipelineIntegration:
    """Integration tests for the complete training pipeline"""

    def test_training_to_evaluation_flow(self, temp_config, temp_processed_data, temp_model_output):
        """Test model training produces proper evaluation results"""
        
        # Initialize trainer with logging
        model_version = f"test_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup logging directory
        logs_dir = os.path.join(temp_model_output, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Import logging setup function
        from training.train_model import setup_logging
        logger = setup_logging(model_version, logs_dir)
        
        trainer = ModelTrainer(temp_config, model_version, logger)
        
        # Run training pipeline
        training_results = trainer.run_training_pipeline(
            temp_processed_data, 
            temp_model_output, 
            forecast_horizon=1
        )
        
        # Verify training completed successfully
        assert training_results['status'] == 'success'
        assert training_results['models_trained'] > 0
        assert training_results['best_model'] is not None
        
        # Verify model files were created
        assert 'saved_files' in training_results
        saved_files = training_results['saved_files']
        
        # Check for essential files
        assert 'evaluation' in saved_files
        assert 'best_model' in saved_files
        assert os.path.exists(saved_files['evaluation'])
        assert os.path.exists(saved_files['best_model'])
        
        # Verify evaluation file structure
        with open(saved_files['evaluation'], 'r') as f:
            evaluation_data = json.load(f)
        
        # Check evaluation metrics
        for model_name, metrics in evaluation_data.items():
            assert 'val_mape' in metrics
            assert 'val_rmse' in metrics
            assert 'val_r2' in metrics
            assert isinstance(metrics['val_mape'], (int, float))

    def test_training_to_model_loading_flow(self, temp_config, temp_processed_data, temp_model_output):
        """Test trained models can be loaded and used for prediction"""
        
        # Train models first
        model_version = f"test_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup logging
        logs_dir = os.path.join(temp_model_output, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        from training.train_model import setup_logging
        logger = setup_logging(model_version, logs_dir)
        
        trainer = ModelTrainer(temp_config, model_version, logger)
        training_results = trainer.run_training_pipeline(
            temp_processed_data, 
            temp_model_output
        )
        
        # Load the best model
        best_model_path = training_results['saved_files']['best_model']
        model_artifact = joblib.load(best_model_path)
        
        # Verify model artifact structure
        if isinstance(model_artifact, dict):
            assert 'model' in model_artifact
            model = model_artifact['model']
        else:
            model = model_artifact
        
        # Test model can make predictions
        test_df = pd.read_parquet(os.path.join(temp_processed_data, 'test.parquet'))
        
        # Prepare features (exclude target and metadata columns)
        exclude_cols = ['Date', 'Item Code'] + [col for col in test_df.columns if 'Target' in col]
        feature_cols = [col for col in test_df.columns if col not in exclude_cols]
        
        X_test = test_df[feature_cols].fillna(0)
        
        # Make predictions
        predictions = model.predict(X_test.head(5))  # Test with small sample
        
        # Verify predictions are reasonable
        assert len(predictions) == 5
        assert all(isinstance(pred, (int, float)) for pred in predictions)
        assert all(pred > 0 for pred in predictions)  # Prices should be positive

    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metric')
    @patch('mlflow.sklearn.log_model')
    def test_training_to_registry_integration(self, mock_log_model, mock_log_metric, 
                                            mock_log_params, mock_start_run, 
                                            temp_config, temp_processed_data, temp_model_output):
        """Test integration between training and model registry"""
        
        # Mock MLflow run context
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_123'
        mock_run.info.artifact_uri = 's3://test-bucket/artifacts'
        mock_start_run.return_value.__enter__ = lambda x: mock_run
        mock_start_run.return_value.__exit__ = lambda x, y, z, w: None
        
        # Train models
        model_version = f"test_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logs_dir = os.path.join(temp_model_output, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        from training.train_model import setup_logging
        logger = setup_logging(model_version, logs_dir)
        
        trainer = ModelTrainer(temp_config, model_version, logger)
        training_results = trainer.run_training_pipeline(
            temp_processed_data, 
            temp_model_output
        )
        
        # Initialize model registry with mocked MLflow
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'), \
             patch('mlflow.get_experiment_by_name'), \
             patch('mlflow.create_experiment'):
            
            registry = ModelRegistry(temp_config)
            
            # Test model registration
            evaluation_file = training_results['saved_files']['evaluation']
            registration_results = registry.register_models_from_training(
                temp_model_output, 
                evaluation_file
            )
            
            # Verify registration was attempted
            assert len(registration_results) > 0
            
            # Verify MLflow calls were made
            assert mock_start_run.called
            assert mock_log_params.called or mock_log_metric.called

    def test_model_validation_and_thresholds(self, temp_config, temp_processed_data, temp_model_output):
        """Test that models meet performance thresholds"""
        
        # Train models
        model_version = f"test_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logs_dir = os.path.join(temp_model_output, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        from training.train_model import setup_logging
        logger = setup_logging(model_version, logs_dir)
        
        trainer = ModelTrainer(temp_config, model_version, logger)
        training_results = trainer.run_training_pipeline(
            temp_processed_data, 
            temp_model_output
        )
        
        # Load evaluation results
        with open(training_results['saved_files']['evaluation'], 'r') as f:
            evaluation_data = json.load(f)
        
        # Check that best model meets basic thresholds
        best_model_name = training_results['best_model']
        best_model_metrics = evaluation_data[best_model_name]
        
        # Basic sanity checks for metrics
        assert 'val_mape' in best_model_metrics
        assert best_model_metrics['val_mape'] < 100  # MAPE should be reasonable
        assert best_model_metrics['val_mape'] > 0    # MAPE should be positive
        
        if 'val_r2' in best_model_metrics:
            assert best_model_metrics['val_r2'] > -1  # R² should be > -1

    def test_model_predictor_integration(self, temp_config, temp_processed_data, temp_model_output):
        """Test ModelPredictor can work with trained models"""
        
        # Train models first
        model_version = f"test_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logs_dir = os.path.join(temp_model_output, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        from training.train_model import setup_logging
        logger = setup_logging(model_version, logs_dir)
        
        trainer = ModelTrainer(temp_config, model_version, logger)
        training_results = trainer.run_training_pipeline(
            temp_processed_data, 
            temp_model_output
        )
        
        # Mock AWS clients for predictor
        with patch('boto3.client'):
            # Initialize predictor
            predictor = ModelPredictor(temp_config)
            
            # Mock the load_local_models method to use our trained models
            predictor.loaded_models = {}
            
            # Load the trained models
            best_model_path = training_results['saved_files']['best_model']
            model_artifact = joblib.load(best_model_path)
            
            predictor.loaded_models['test_model'] = {
                'model': model_artifact.get('model') if isinstance(model_artifact, dict) else model_artifact,
                'scaler': model_artifact.get('scaler') if isinstance(model_artifact, dict) else None,
                'version': 'test',
                'metrics': {}
            }
            
            # Test prediction
            test_df = pd.read_parquet(os.path.join(temp_processed_data, 'test.parquet'))
            sample_data = test_df.head(3)
            
            # Make prediction
            prediction_result = predictor.predict(sample_data, model_name='test_model')
            
            # Verify prediction structure
            assert 'predictions' in prediction_result
            assert 'confidence' in prediction_result
            assert 'model_used' in prediction_result
            assert len(prediction_result['predictions']) == 3

    def test_training_pipeline_error_handling(self, temp_config, temp_model_output):
        """Test training pipeline handles errors gracefully"""
        
        # Create empty data directory (should cause graceful failure)
        empty_dir = tempfile.mkdtemp()
        
        try:
            model_version = f"test_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logs_dir = os.path.join(temp_model_output, 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            from training.train_model import setup_logging
            logger = setup_logging(model_version, logs_dir)
            
            trainer = ModelTrainer(temp_config, model_version, logger)
            
            # This should fail gracefully
            with pytest.raises(Exception):
                trainer.run_training_pipeline(empty_dir, temp_model_output)
                
        finally:
            shutil.rmtree(empty_dir)

    def test_feature_importance_integration(self, temp_config, temp_processed_data, temp_model_output):
        """Test that feature importance is captured and usable"""
        
        # Train models
        model_version = f"test_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logs_dir = os.path.join(temp_model_output, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        from training.train_model import setup_logging
        logger = setup_logging(model_version, logs_dir)
        
        trainer = ModelTrainer(temp_config, model_version, logger)
        training_results = trainer.run_training_pipeline(
            temp_processed_data, 
            temp_model_output
        )
        
        # Load best model and check for feature importance
        best_model_path = training_results['saved_files']['best_model']
        model_artifact = joblib.load(best_model_path)
        
        if isinstance(model_artifact, dict):
            model = model_artifact['model']
        else:
            model = model_artifact
        
        # Check if model has feature importance (tree-based models)
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            assert len(feature_importance) > 0
            assert all(imp >= 0 for imp in feature_importance)  # Importance should be non-negative
            assert abs(sum(feature_importance) - 1.0) < 0.01   # Should sum to approximately 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
