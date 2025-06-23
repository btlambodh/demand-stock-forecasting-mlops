#!/usr/bin/env python3
"""
Unit tests for model training components
Tests model training, evaluation, and registry functionality

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import os
import pytest
import pandas as pd
import numpy as np
import json
import joblib
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import modules to test
from src.training.train_model import ModelTrainer
from src.deployment.model_registry import ModelRegistry
from src.inference.predictor import ModelPredictor


class TestModelTrainer:
    """Unit tests for ModelTrainer class"""

    @pytest.mark.unit
    def test_trainer_initialization(self, config_file):
        """Test ModelTrainer initialization"""
        trainer = ModelTrainer(config_file, "test_v1", None)
        assert trainer.config is not None
        assert trainer.model_version == "test_v1"

    @pytest.mark.unit
    def test_load_processed_data(self, config_file, save_sample_data):
        """Test loading processed data"""
        trainer = ModelTrainer(config_file, "test_v1", None)
        processed_dir = save_sample_data['processed_dir']
        
        data = trainer.load_processed_data(processed_dir)
        
        assert isinstance(data, dict)
        assert 'train' in data
        assert 'validation' in data
        assert 'test' in data
        
        # Check data types
        for split_name, df in data.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'Date' in df.columns

    @pytest.mark.unit
    def test_prepare_features_targets(self, config_file, sample_processed_data):
        """Test feature and target preparation"""
        trainer = ModelTrainer(config_file, "test_v1", None)
        
        # Add target column to sample data
        df_with_targets = sample_processed_data.copy()
        
        X, y = trainer.prepare_features_targets(df_with_targets, forecast_horizon=1)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X.columns) > 0
        
        # Check that target columns and non-feature columns are excluded
        excluded_cols = ['Date', 'Item Code', 'Category Name']
        for col in excluded_cols:
            if col in df_with_targets.columns:
                assert col not in X.columns

    @pytest.mark.unit
    def test_add_derived_features(self, config_file):
        """Test derived feature creation"""
        trainer = ModelTrainer(config_file, "test_v1", None)
        
        # Create sample data without derived features
        sample_df = pd.DataFrame({
            'Total_Quantity': [100, 150, 200],
            'Avg_Price': [15.0, 18.0, 12.0],
            'Transaction_Count': [10, 15, 20],
            'Max_Price': [18.0, 22.0, 15.0],
            'Min_Price': [12.0, 14.0, 9.0],
            'Month': [6, 7, 8],
            'DayOfWeek': [1, 2, 3]
        })
        
        result_df = trainer.add_derived_features(sample_df)
        
        # Check that derived features were added
        derived_features = ['Revenue', 'Price_Range', 'Month_Sin', 'Month_Cos', 
                           'DayOfWeek_Sin', 'DayOfWeek_Cos', 'Quarter']
        for feature in derived_features:
            assert feature in result_df.columns
        
        # Check calculated values are reasonable
        assert (result_df['Revenue'] == result_df['Total_Quantity'] * result_df['Avg_Price']).all()
        assert (result_df['Price_Range'] == result_df['Max_Price'] - result_df['Min_Price']).all()

    @pytest.mark.unit
    def test_handle_missing_values(self, config_file):
        """Test missing value handling in trainer"""
        trainer = ModelTrainer(config_file, "test_v1", None)
        
        # Create data with missing values
        df_with_missing = pd.DataFrame({
            'Item Code': [101, 102, 103],
            'Price_Volatility': [1.2, np.nan, 1.8],
            'Discount_Rate': [np.nan, 0.1, 0.15],
            'Avg_Price_Lag_1': [15.0, np.nan, 17.0],
            'Avg_Price_MA_7': [np.nan, 16.0, 18.0],
            'Category_Avg_Price': [14.0, 15.0, np.nan]
        })
        
        result_df = trainer.handle_missing_values(df_with_missing)
        
        # Check that missing values are handled
        assert result_df.isnull().sum().sum() == 0
        
        # Check that default values are reasonable
        assert result_df['Price_Volatility'].notna().all()
        assert result_df['Discount_Rate'].notna().all()

    @pytest.mark.unit
    def test_train_models(self, config_file, sample_processed_data):
        """Test model training"""
        trainer = ModelTrainer(config_file, "test_v1", None)
        
        # Prepare data
        X_train = sample_processed_data[['Total_Quantity', 'Avg_Price', 'Month', 'DayOfWeek']].head(50)
        y_train = sample_processed_data['Avg_Price_Target_1d'].head(50).fillna(15.0)
        X_val = sample_processed_data[['Total_Quantity', 'Avg_Price', 'Month', 'DayOfWeek']].tail(20)
        y_val = sample_processed_data['Avg_Price_Target_1d'].tail(20).fillna(15.0)
        
        results = trainer.train_models(X_train, y_train, X_val, y_val)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that models were trained
        for model_name, result in results.items():
            if 'error' not in result:
                assert 'model_artifact' in result
                assert 'metrics' in result
                assert 'training_time' in result
                
                # Check model artifact structure
                artifact = result['model_artifact']
                assert 'model' in artifact
                
                # Check metrics
                metrics = result['metrics']
                required_metrics = ['train_mae', 'train_rmse', 'val_mae', 'val_rmse', 'val_mape']
                for metric in required_metrics:
                    assert metric in metrics
                    assert isinstance(metrics[metric], (int, float))

    @pytest.mark.unit
    def test_calculate_metrics(self, config_file):
        """Test metrics calculation"""
        trainer = ModelTrainer(config_file, "test_v1", None)
        
        # Create sample predictions and targets
        y_true_train = np.array([10, 15, 20, 25, 30])
        y_pred_train = np.array([11, 14, 19, 26, 29])
        y_true_val = np.array([12, 18, 22])
        y_pred_val = np.array([13, 17, 21])
        
        metrics = trainer.calculate_metrics(
            pd.Series(y_true_train), y_pred_train,
            pd.Series(y_true_val), y_pred_val
        )
        
        assert isinstance(metrics, dict)
        
        # Check all required metrics are present
        required_metrics = [
            'train_mae', 'train_mse', 'train_rmse', 'train_r2', 'train_mape',
            'val_mae', 'val_mse', 'val_rmse', 'val_r2', 'val_mape', 'val_smape'
        ]
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])

    @pytest.mark.unit
    def test_evaluate_on_test(self, config_file, sample_processed_data):
        """Test model evaluation on test set"""
        trainer = ModelTrainer(config_file, "test_v1", None)
        
        # Create a simple trained model result
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        scaler = StandardScaler()
        
        # Fit with sample data
        X_sample = sample_processed_data[['Total_Quantity', 'Avg_Price', 'Month', 'DayOfWeek']].head(50)
        y_sample = sample_processed_data['Avg_Price_Target_1d'].head(50).fillna(15.0)
        X_scaled = scaler.fit_transform(X_sample)
        model.fit(X_scaled, y_sample)
        
        # Create results dict
        results = {
            'test_model': {
                'model_artifact': {'model': model, 'scaler': scaler},
                'metrics': {'val_mape': 12.5, 'val_rmse': 2.1},
                'training_time': 10.5
            }
        }
        
        # Test data
        X_test = sample_processed_data[['Total_Quantity', 'Avg_Price', 'Month', 'DayOfWeek']].tail(20)
        y_test = sample_processed_data['Avg_Price_Target_1d'].tail(20).fillna(15.0)
        
        test_results = trainer.evaluate_on_test(results, X_test, y_test)
        
        assert isinstance(test_results, dict)
        assert 'test_model' in test_results
        
        test_result = test_results['test_model']
        assert 'metrics' in test_result
        
        # Check test metrics were added
        test_metrics = ['test_mae', 'test_rmse', 'test_r2', 'test_mape']
        for metric in test_metrics:
            assert metric in test_result['metrics']

    @pytest.mark.unit
    def test_save_models_and_results(self, config_file, temp_dir, sample_trained_model, sample_evaluation_results):
        """Test saving models and results"""
        trainer = ModelTrainer(config_file, "test_v1", None)
        
        # Create results dict with model artifacts
        results = {
            'test_model': {
                'model_artifact': sample_trained_model,
                'metrics': {'val_mape': 12.5},
                'training_time': 10.0
            }
        }
        
        feature_cols = ['feature_1', 'feature_2', 'feature_3']
        
        saved_files = trainer.save_models_and_results(
            results, sample_evaluation_results, feature_cols, temp_dir
        )
        
        assert isinstance(saved_files, dict)
        assert 'test_model' in saved_files
        assert 'evaluation' in saved_files
        assert 'feature_info' in saved_files
        
        # Check files exist
        for file_path in saved_files.values():
            assert os.path.exists(file_path)
        
        # Check model file can be loaded
        model_path = saved_files['test_model']
        loaded_model = joblib.load(model_path)
        assert 'model' in loaded_model
        assert 'scaler' in loaded_model


class TestModelRegistry:
    """Unit tests for ModelRegistry class"""

    @pytest.mark.unit
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment') 
    @patch('boto3.client')
    def test_registry_initialization(self, mock_boto_client, mock_set_experiment, mock_set_tracking_uri, config_file):
        """Test ModelRegistry initialization"""
        registry = ModelRegistry(config_file)
        assert registry.config is not None
        assert hasattr(registry, 'aws_config')
        mock_set_tracking_uri.assert_called()

    @pytest.mark.unit
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metric')
    @patch('mlflow.sklearn.log_model')
    @patch('boto3.client')
    def test_register_model(self, mock_boto_client, mock_log_model, mock_log_metric, 
                           mock_log_params, mock_start_run, config_file, sample_model_file):
        """Test model registration"""
        # Mock MLflow run context
        mock_run = Mock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.artifact_uri = 's3://test-bucket/artifacts'
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        registry = ModelRegistry(config_file)
        
        metadata = {
            'metrics': {'val_mape': 12.5, 'val_rmse': 2.1},
            'training_timestamp': datetime.now().isoformat(),
            'feature_count': 10
        }
        
        result = registry.register_model(
            model_path=sample_model_file,
            model_name='test_model',
            model_version='v1.0',
            metadata=metadata
        )
        
        assert isinstance(result, dict)
        assert 'run_id' in result
        assert 'model_name' in result
        assert 'status' in result
        assert result['status'] == 'registered'

    @pytest.mark.unit
    @patch('boto3.client')
    def test_validate_model_performance(self, mock_boto_client, config_file):
        """Test model performance validation"""
        registry = ModelRegistry(config_file)
        
        # Test valid metrics
        valid_metrics = {
            'val_mape': 12.0,  # Below 15% threshold
            'val_rmse': 4.0,   # Below 5.0 threshold  
            'val_r2': 0.8      # Above 0.7 threshold
        }
        
        assert registry.validate_model_performance(valid_metrics) is True
        
        # Test invalid metrics
        invalid_metrics = {
            'val_mape': 25.0,  # Above threshold
            'val_rmse': 6.0,   # Above threshold
            'val_r2': 0.5      # Below threshold
        }
        
        assert registry.validate_model_performance(invalid_metrics) is False

    @pytest.mark.unit
    @patch('boto3.client')
    def test_compare_models(self, mock_boto_client, config_file):
        """Test model comparison functionality"""
        registry = ModelRegistry(config_file)
        
        # Mock MLflow client methods
        mock_client = Mock()
        registry.client = mock_client
        
        # Mock model version info
        v1_info = Mock()
        v1_info.current_stage = 'Staging'
        v1_info.run_id = 'run_1'
        
        v2_info = Mock()
        v2_info.current_stage = 'Production'
        v2_info.run_id = 'run_2'
        
        mock_client.get_model_version.side_effect = [v1_info, v2_info]
        
        # Mock run info
        v1_run = Mock()
        v1_run.data.metrics = {'val_mape': 15.0, 'val_rmse': 2.5, 'val_r2': 0.75}
        
        v2_run = Mock()
        v2_run.data.metrics = {'val_mape': 12.0, 'val_rmse': 2.1, 'val_r2': 0.82}
        
        mock_client.get_run.side_effect = [v1_run, v2_run]
        
        comparison = registry.compare_models('test_model', '1', '2')
        
        assert isinstance(comparison, dict)
        assert 'model_name' in comparison
        assert 'version1' in comparison
        assert 'version2' in comparison
        assert 'comparison' in comparison

    @pytest.mark.unit
    @patch('boto3.client')
    def test_cleanup_old_models(self, mock_boto_client, config_file):
        """Test old model cleanup"""
        registry = ModelRegistry(config_file)
        
        # Mock MLflow client
        mock_client = Mock()
        registry.client = mock_client
        
        # Mock model versions (simulate 10 versions)
        mock_versions = []
        for i in range(10):
            version = Mock()
            version.version = str(i + 1)
            version.current_stage = 'None' if i < 7 else ('Production' if i == 9 else 'Staging')
            mock_versions.append(version)
        
        mock_client.search_model_versions.return_value = mock_versions
        mock_client.delete_model_version.return_value = None
        
        deleted_count = registry.cleanup_old_models('test_model', keep_versions=5)
        
        # Should delete 2 versions (versions 1 and 2, keeping 3-7 plus Production/Staging)
        assert deleted_count >= 0
        assert mock_client.delete_model_version.call_count >= 0


class TestModelPredictor:
    """Unit tests for ModelPredictor class"""

    @pytest.mark.unit
    @patch('boto3.client')
    @patch('mlflow.set_tracking_uri')
    def test_predictor_initialization(self, mock_mlflow, mock_boto_client, config_file):
        """Test ModelPredictor initialization"""
        predictor = ModelPredictor(config_file)
        assert predictor.config is not None
        assert hasattr(predictor, 'loaded_models')
        assert hasattr(predictor, 'prediction_stats')

    @pytest.mark.unit
    @patch('boto3.client')
    @patch('mlflow.set_tracking_uri')
    def test_load_local_models(self, mock_mlflow, mock_boto_client, config_file, temp_dir, sample_evaluation_results):
        """Test loading models from local directory"""
        predictor = ModelPredictor(config_file)
        
        # Create mock models directory
        models_dir = os.path.join(temp_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Create sample model file
        model_path = create_temp_model_file(models_dir, 'test_model')
        
        # Create evaluation file
        eval_path = os.path.join(models_dir, 'evaluation.json')
        with open(eval_path, 'w') as f:
            json.dump(sample_evaluation_results, f)
        
        # Mock models directory location
        with patch.object(predictor, 'load_local_models') as mock_load:
            mock_load.return_value = {
                'chinese_produce_test_model': {
                    'model': RandomForestRegressor(),
                    'scaler': StandardScaler(),
                    'version': 'local',
                    'metrics': sample_evaluation_results['random_forest']
                }
            }
            
            models = mock_load()
            
            assert isinstance(models, dict)
            assert len(models) > 0

    @pytest.mark.unit
    @patch('boto3.client')
    @patch('mlflow.set_tracking_uri')
    def test_preprocess_features(self, mock_mlflow, mock_boto_client, config_file):
        """Test feature preprocessing"""
        predictor = ModelPredictor(config_file)
        
        # Set up feature columns
        predictor.feature_columns = ['Total_Quantity', 'Avg_Price', 'Month', 'DayOfWeek', 'Revenue']
        
        # Create sample input data
        input_df = pd.DataFrame({
            'Total_Quantity': [100, 150],
            'Avg_Price': [15.0, 18.0],
            'Month': [6, 7],
            'DayOfWeek': [1, 2],
            'Extra_Column': ['A', 'B']  # Should be removed
        })
        
        result_df = predictor.preprocess_features(input_df)
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
        
        # Check that non-feature columns are handled
        assert 'Extra_Column' not in result_df.columns or len(result_df.columns) >= 4

    @pytest.mark.unit
    @patch('boto3.client')
    @patch('mlflow.set_tracking_uri')
    def test_add_derived_features(self, mock_mlflow, mock_boto_client, config_file):
        """Test derived feature addition in predictor"""
        predictor = ModelPredictor(config_file)
        
        input_df = pd.DataFrame({
            'Total_Quantity': [100],
            'Avg_Price': [15.0],
            'Transaction_Count': [10],
            'Month': [6]
        })
        
        result_df = predictor.add_derived_features(input_df)
        
        # Check that Revenue was calculated
        if 'Total_Quantity' in result_df.columns and 'Avg_Price' in result_df.columns:
            expected_revenue = result_df['Total_Quantity'].iloc[0] * result_df['Avg_Price'].iloc[0]
            if 'Revenue' in result_df.columns:
                assert result_df['Revenue'].iloc[0] == expected_revenue

    @pytest.mark.unit
    @patch('boto3.client')
    @patch('mlflow.set_tracking_uri')
    def test_validate_input(self, mock_mlflow, mock_boto_client, config_file):
        """Test input validation"""
        predictor = ModelPredictor(config_file)
        
        # Test valid input
        valid_df = pd.DataFrame({
            'Avg_Price': [15.0, 18.0],
            'Total_Quantity': [100, 150],
            'Month': [6, 7],
            'DayOfWeek': [1, 2]
        })
        
        is_valid, errors = predictor.validate_input(valid_df)
        assert is_valid is True
        assert len(errors) == 0
        
        # Test invalid input (empty DataFrame)
        empty_df = pd.DataFrame()
        is_valid, errors = predictor.validate_input(empty_df)
        assert is_valid is False
        assert len(errors) > 0
        
        # Test invalid input (negative prices)
        invalid_df = pd.DataFrame({
            'Avg_Price': [-5.0, 18.0],
            'Total_Quantity': [100, 150],
            'Month': [6, 7],
            'DayOfWeek': [1, 2]
        })
        
        is_valid, errors = predictor.validate_input(invalid_df)
        assert is_valid is False
        assert any('negative' in error.lower() for error in errors)

    @pytest.mark.unit
    @patch('boto3.client')
    @patch('mlflow.set_tracking_uri')
    def test_calculate_prediction_confidence(self, mock_mlflow, mock_boto_client, config_file):
        """Test prediction confidence calculation"""
        predictor = ModelPredictor(config_file)
        
        # Create mock model and data
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        X_scaled = np.random.randn(10, 5)
        y = np.random.randn(10)
        model.fit(X_scaled, y)
        
        predictions = model.predict(X_scaled)
        model_info = {'metrics': {'val_mape': 12.5}}
        
        confidence = predictor.calculate_prediction_confidence(
            model, X_scaled, predictions, model_info
        )
        
        assert isinstance(confidence, np.ndarray)
        assert len(confidence) == len(predictions)
        assert all(0.1 <= c <= 0.95 for c in confidence)

    @pytest.mark.unit
    @patch('boto3.client')
    @patch('mlflow.set_tracking_uri')
    def test_get_prediction_stats(self, mock_mlflow, mock_boto_client, config_file):
        """Test prediction statistics retrieval"""
        predictor = ModelPredictor(config_file)
        
        # Simulate some predictions
        predictor.prediction_stats['total_predictions'] = 100
        predictor.prediction_stats['successful_predictions'] = 95
        predictor.prediction_stats['average_latency'] = 0.05
        
        stats = predictor.get_prediction_stats()
        
        assert isinstance(stats, dict)
        assert 'total_predictions' in stats
        assert 'successful_predictions' in stats
        assert 'average_latency' in stats
        assert stats['total_predictions'] == 100
        assert stats['successful_predictions'] == 95

    @pytest.mark.unit
    @patch('boto3.client')
    @patch('mlflow.set_tracking_uri')
    def test_get_model_info(self, mock_mlflow, mock_boto_client, config_file):
        """Test model information retrieval"""
        predictor = ModelPredictor(config_file)
        
        # Add mock model to loaded models
        predictor.loaded_models['test_model'] = {
            'model': RandomForestRegressor(),
            'scaler': StandardScaler(),
            'version': 'v1.0',
            'metrics': {'val_mape': 12.5},
            'model_type': 'RandomForestRegressor'
        }
        
        # Test getting specific model info
        model_info = predictor.get_model_info('test_model')
        assert isinstance(model_info, dict)
        assert 'version' in model_info
        assert 'metrics' in model_info
        assert 'model_type' in model_info
        # Model and scaler objects should be removed for serialization
        assert 'model' not in model_info
        assert 'scaler' not in model_info
        
        # Test getting all models info
        all_info = predictor.get_model_info()
        assert isinstance(all_info, dict)
        assert 'test_model' in all_info
