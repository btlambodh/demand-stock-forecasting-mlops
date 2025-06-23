#!/usr/bin/env python3
"""
Integration tests for training pipeline
Tests end-to-end training workflow from data to deployed model

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
from src.data_processing.data_validation import DataValidator
from src.data_processing.feature_engineering import FeatureEngineer
from src.training.train_model import ModelTrainer
from src.deployment.model_registry import ModelRegistry
from src.inference.predictor import ModelPredictor  # FIXED: was deployment.predictor
from src.monitoring.drift_detector import DriftDetector


class TestTrainingPipeline:
    """Integration tests for complete training pipeline"""

    @pytest.mark.integration
    @pytest.mark.training
    def test_end_to_end_training_pipeline(self, config_file, save_sample_data):
        """Test complete training pipeline from raw data to trained model"""
        
        raw_dir = save_sample_data['raw_dir']
        processed_dir = save_sample_data['processed_dir']
        
        # Step 1: Data Validation
        validator = DataValidator(config_file)
        validation_results = validator.run_validation(raw_dir, processed_dir)
        
        assert isinstance(validation_results, dict)
        assert 'validation_passed' in validation_results
        assert validation_results['files_validated'] > 0
        
        # Step 2: Feature Engineering
        feature_engineer = FeatureEngineer(config_file)
        raw_data = feature_engineer.load_raw_data(raw_dir)
        
        assert isinstance(raw_data, dict)
        assert len(raw_data) > 0
        
        # Step 3: Model Training
        with tempfile.TemporaryDirectory() as temp_model_dir:
            trainer = ModelTrainer(config_file, "test_pipeline_v1", temp_model_dir)
            
            # Load processed data
            processed_data = trainer.load_processed_data(processed_dir)
            assert isinstance(processed_data, dict)
            assert 'train' in processed_data
            
            # Prepare features and targets
            train_df = processed_data['train']
            if len(train_df) > 50:  # Ensure we have enough data
                X, y = trainer.prepare_features_targets(train_df, forecast_horizon=1)
                
                if len(X) > 20:  # Minimum for training
                    # Split data
                    split_idx = int(len(X) * 0.8)
                    X_train, X_val = X[:split_idx], X[split_idx:]
                    y_train, y_val = y[:split_idx], y[split_idx:]
                    
                    # Train models
                    training_results = trainer.train_models(X_train, y_train, X_val, y_val)
                    
                    assert isinstance(training_results, dict)
                    assert len(training_results) > 0
                    
                    # Check that at least one model trained successfully
                    successful_models = [name for name, result in training_results.items() 
                                       if 'error' not in result]
                    assert len(successful_models) > 0

    @pytest.mark.integration
    @pytest.mark.training
    def test_data_processing_pipeline(self, config_file, save_sample_data):
        """Test data processing pipeline"""
        
        raw_dir = save_sample_data['raw_dir']
        
        # Initialize components
        validator = DataValidator(config_file)
        feature_engineer = FeatureEngineer(config_file)
        
        # Test validation
        file_status = validator.validate_file_existence(raw_dir)
        assert isinstance(file_status, dict)
        assert all(file_status.values())  # All files should exist
        
        # Test data loading
        raw_data = feature_engineer.load_raw_data(raw_dir)
        assert 'items' in raw_data
        assert 'sales' in raw_data
        assert 'wholesale' in raw_data
        assert 'loss_rates' in raw_data
        
        # Test feature engineering steps
        sales_df = raw_data['sales']
        
        # Create daily aggregates
        daily_agg = feature_engineer.create_daily_aggregates(sales_df)
        assert isinstance(daily_agg, pd.DataFrame)
        assert len(daily_agg) > 0
        
        # Add temporal features
        with_temporal = feature_engineer.add_temporal_features(daily_agg)
        assert 'Month_Sin' in with_temporal.columns
        assert 'Month_Cos' in with_temporal.columns
        assert 'IsWeekend' in with_temporal.columns

    @pytest.mark.integration
    @pytest.mark.training
    def test_model_training_with_registry(self, config_file, sample_processed_data):
        """Test model training integrated with registry"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize trainer
            trainer = ModelTrainer(config_file, "test_registry_v1", temp_dir)
            
            # Prepare minimal training data
            if len(sample_processed_data) > 100:
                train_data = sample_processed_data.head(80)
                val_data = sample_processed_data.tail(20)
                
                X_train, y_train = trainer.prepare_features_targets(train_data, forecast_horizon=1)
                X_val, y_val = trainer.prepare_features_targets(val_data, forecast_horizon=1)
                
                if len(X_train) > 10 and len(X_val) > 5:
                    # Train a simple model
                    training_results = trainer.train_models(X_train, y_train, X_val, y_val)
                    
                    # Test registry integration
                    with patch('mlflow.set_tracking_uri'), \
                         patch('mlflow.set_experiment'), \
                         patch('boto3.client'):
                        
                        registry = ModelRegistry(config_file)
                        
                        # Test model registration
                        for model_name, result in training_results.items():
                            if 'error' not in result and 'model_artifact' in result:
                                
                                # Save model to file
                                model_path = os.path.join(temp_dir, f"{model_name}.pkl")
                                import joblib
                                joblib.dump(result['model_artifact'], model_path)
                                
                                # Test registration
                                metadata = {
                                    'metrics': result['metrics'],
                                    'training_timestamp': datetime.now().isoformat(),
                                    'feature_count': len(X_train.columns)
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
                                        
                                        reg_result = registry.register_model(
                                            model_path, f"test_{model_name}", "v1.0", metadata
                                        )
                                        
                                        assert isinstance(reg_result, dict)
                                        assert 'run_id' in reg_result
                                        break  # Test one model registration

    @pytest.mark.integration
    @pytest.mark.training
    def test_predictor_loading_pipeline(self, config_file, sample_trained_model):
        """Test predictor model loading pipeline"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model to file
            model_path = os.path.join(temp_dir, 'test_model.pkl')
            import joblib
            joblib.dump(sample_trained_model, model_path)
            
            # Create evaluation file
            evaluation_data = {
                'test_model': {
                    'val_mape': 12.5,
                    'val_rmse': 2.1,
                    'val_r2': 0.82,
                    'test_mape': 13.2
                }
            }
            
            eval_path = os.path.join(temp_dir, 'evaluation.json')
            with open(eval_path, 'w') as f:
                json.dump(evaluation_data, f)
            
            # Initialize predictor
            with patch('boto3.client'), \
                 patch('mlflow.set_tracking_uri'):
                
                predictor = ModelPredictor(config_file)
                
                # Test local model loading simulation
                mock_models = {
                    'test_model': {
                        'model': sample_trained_model['model'],
                        'scaler': sample_trained_model['scaler'],
                        'version': 'local',
                        'metrics': evaluation_data['test_model']
                    }
                }
                
                predictor.loaded_models = mock_models
                
                # Test prediction capability
                test_data = pd.DataFrame({
                    'Total_Quantity': [100.0],
                    'Avg_Price': [15.0],
                    'Month': [6],
                    'DayOfWeek': [1]
                })
                
                # Test preprocessing
                is_valid, errors = predictor.validate_input(test_data)
                assert is_valid or len(errors) == 0  # Should be valid or have manageable errors

    @pytest.mark.integration
    @pytest.mark.training
    def test_drift_detection_integration(self, config_file, sample_processed_data):
        """Test drift detection integration with training pipeline"""
        
        # Initialize drift detector
        drift_detector = DriftDetector(config_file, local_mode=True)
        
        # Split data for reference and current
        reference_data = sample_processed_data.head(200)
        current_data = sample_processed_data.tail(100)
        
        # Set reference data
        drift_detector.reference_data = reference_data
        drift_detector.calculate_reference_statistics()
        
        # Test drift detection
        drift_results = drift_detector.detect_data_drift(current_data, method='statistical')
        
        assert isinstance(drift_results, dict)
        assert 'overall_drift_detected' in drift_results
        assert 'drift_score' in drift_results
        assert 'timestamp' in drift_results

    @pytest.mark.integration
    @pytest.mark.training
    def test_training_evaluation_pipeline(self, config_file, sample_processed_data):
        """Test training and evaluation pipeline"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ModelTrainer(config_file, "test_eval_v1", temp_dir)
            
            # Prepare data splits
            if len(sample_processed_data) > 150:
                train_size = int(len(sample_processed_data) * 0.6)
                val_size = int(len(sample_processed_data) * 0.2)
                
                train_data = sample_processed_data.iloc[:train_size]
                val_data = sample_processed_data.iloc[train_size:train_size+val_size]
                test_data = sample_processed_data.iloc[train_size+val_size:]
                
                # Prepare features and targets
                X_train, y_train = trainer.prepare_features_targets(train_data, forecast_horizon=1)
                X_val, y_val = trainer.prepare_features_targets(val_data, forecast_horizon=1)
                X_test, y_test = trainer.prepare_features_targets(test_data, forecast_horizon=1)
                
                if len(X_train) > 20 and len(X_val) > 10 and len(X_test) > 10:
                    # Train models
                    training_results = trainer.train_models(X_train, y_train, X_val, y_val)
                    
                    # Evaluate on test set
                    test_results = trainer.evaluate_on_test(training_results, X_test, y_test)
                    
                    assert isinstance(test_results, dict)
                    
                    # Check test metrics were added
                    for model_name, result in test_results.items():
                        if 'error' not in result:
                            assert 'metrics' in result
                            metrics = result['metrics']
                            
                            # Check test metrics exist
                            test_metrics = ['test_mae', 'test_rmse', 'test_r2', 'test_mape']
                            for metric in test_metrics:
                                assert metric in metrics

    @pytest.mark.integration
    @pytest.mark.training
    def test_feature_importance_pipeline(self, config_file, sample_processed_data):
        """Test feature importance extraction in training pipeline"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ModelTrainer(config_file, "test_importance_v1", temp_dir)
            
            # Prepare minimal data
            if len(sample_processed_data) > 100:
                train_data = sample_processed_data.head(80)
                val_data = sample_processed_data.tail(20)
                
                X_train, y_train = trainer.prepare_features_targets(train_data, forecast_horizon=1)
                X_val, y_val = trainer.prepare_features_targets(val_data, forecast_horizon=1)
                
                if len(X_train) > 20:
                    # Train models
                    training_results = trainer.train_models(X_train, y_train, X_val, y_val)
                    
                    # Check feature importance
                    for model_name, result in training_results.items():
                        if 'error' not in result and 'model_artifact' in result:
                            model = result['model_artifact']['model']
                            
                            # Check if model has feature importance
                            if hasattr(model, 'feature_importances_'):
                                importance = model.feature_importances_
                                assert len(importance) == len(X_train.columns)
                                assert np.sum(importance) > 0  # Should have some importance

    @pytest.mark.integration
    @pytest.mark.training
    def test_training_pipeline_error_handling(self, config_file):
        """Test training pipeline error handling"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ModelTrainer(config_file, "test_error_v1", temp_dir)
            
            # Test with invalid data
            invalid_data = pd.DataFrame()  # Empty DataFrame
            
            try:
                X, y = trainer.prepare_features_targets(invalid_data, forecast_horizon=1)
                # Should handle empty data gracefully
                assert len(X) == 0 or len(y) == 0
            except Exception:
                # Exception is acceptable for invalid data
                pass
            
            # Test with minimal data
            minimal_data = pd.DataFrame({
                'Total_Quantity': [100, 150],
                'Avg_Price': [15, 18],
                'Avg_Price_Target_1d': [16, 19]
            })
            
            X, y = trainer.prepare_features_targets(minimal_data, forecast_horizon=1)
            
            # Should handle minimal data
            if len(X) > 0 and len(y) > 0:
                # Try training with very small dataset
                training_results = trainer.train_models(X, y, X, y)
                assert isinstance(training_results, dict)
                
                # Some models might fail with tiny datasets, which is expected
                total_models = len(training_results)
                failed_models = len([r for r in training_results.values() if 'error' in r])
                
                # At least one model type should be attempted
                assert total_models > 0

    @pytest.mark.integration
    @pytest.mark.training
    def test_pipeline_data_quality_integration(self, config_file, save_sample_data):
        """Test training pipeline integration with data quality checks"""
        
        raw_dir = save_sample_data['raw_dir']
        
        # Initialize validator
        validator = DataValidator(config_file)
        
        # Test comprehensive validation
        with tempfile.TemporaryDirectory() as validation_output:
            validation_results = validator.run_validation(raw_dir, validation_output)
            
            assert isinstance(validation_results, dict)
            assert 'total_quality_score' in validation_results
            
            # Quality score should be reasonable for test data
            quality_score = validation_results['total_quality_score']
            assert 0 <= quality_score <= 100
            
            # Check validation report was created
            validation_files = os.listdir(validation_output)
            report_files = [f for f in validation_files if 'validation' in f.lower()]
            assert len(report_files) > 0