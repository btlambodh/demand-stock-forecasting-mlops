#!/usr/bin/env python3
"""
End-to-End Integration Tests for MLOps Pipeline
Tests the complete workflow from raw data to deployed model and monitoring
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
import time

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data_processing.data_validation import DataValidator
from data_processing.feature_engineering import FeatureEngineer
from training.train_model import ModelTrainer
from inference.api import SageMakerSyncAPI
from monitoring.performance_monitor import PerformanceMonitor
from monitoring.drift_detector import DriftDetector


@pytest.fixture
def temp_config():
    """Create comprehensive config for end-to-end testing"""
    config_data = {
        'project': {
            'name': 'e2e-test-project',
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
                'mape_threshold': 25.0,  # Relaxed for testing
                'rmse_threshold': 10.0,
                'r2_threshold': 0.5
            }
        },
        'monitoring': {
            'performance': {
                'drift_threshold': 0.3,
                'performance_degradation_threshold': 0.2,
                'cpu_threshold': 80,
                'memory_threshold': 85,
                'disk_threshold': 90
            },
            'alerts': {
                'enabled': True,
                'cooldown_minutes': 30
            }
        },
        'deployment': {
            'environments': {
                'dev': {
                    'instance_type': 'ml.t2.medium',
                    'initial_instance_count': 1,
                    'auto_scaling_enabled': False
                }
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        return f.name


@pytest.fixture
def comprehensive_test_data():
    """Create comprehensive test data for end-to-end testing"""
    temp_dir = tempfile.mkdtemp()
    
    # Create realistic test data
    np.random.seed(42)
    n_samples = 500  # Larger dataset for realistic testing
    
    # Generate dates
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')
    
    # Generate items and categories
    items = [101, 102, 103, 104, 105]
    categories = [1, 1, 2, 2, 3]
    
    # Create comprehensive datasets
    item_master = pd.DataFrame({
        'Item Code': items,
        'Item Name': ['Apple', 'Banana', 'Orange', 'Lemon', 'Grape'],
        'Category Code': categories,
        'Category Name': ['Fruit', 'Fruit', 'Citrus', 'Citrus', 'Berry']
    })
    
    # Generate sales transactions with realistic patterns
    sales_data = []
    for date in dates:
        daily_items = np.random.choice(items, size=np.random.randint(10, 30), replace=True)
        for item in daily_items:
            # Add seasonality and trends
            base_price = 10 + (item - 100) * 2
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
            price = base_price * seasonal_factor * np.random.uniform(0.8, 1.2)
            
            sales_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Time': f"{np.random.randint(8, 18):02d}:{np.random.randint(0, 60):02d}",
                'Item Code': item,
                'Quantity Sold (kilo)': np.random.uniform(5, 100),
                'Unit Selling Price (RMB/kg)': price,
                'Sale or Return': 'Sale',
                'Discount (Yes/No)': np.random.choice(['Yes', 'No'], p=[0.2, 0.8])
            })
    
    sales_df = pd.DataFrame(sales_data)
    
    # Generate wholesale prices
    wholesale_data = []
    for date in dates:
        for item in items:
            base_wholesale = 7 + (item - 100) * 1.5
            seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * date.dayofyear / 365)
            wholesale_price = base_wholesale * seasonal_factor * np.random.uniform(0.9, 1.1)
            
            wholesale_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Item Code': item,
                'Wholesale Price (RMB/kg)': wholesale_price
            })
    
    wholesale_df = pd.DataFrame(wholesale_data)
    
    # Generate loss rates
    loss_rates = pd.DataFrame({
        'Item Code': items,
        'Item Name': ['Apple', 'Banana', 'Orange', 'Lemon', 'Grape'],
        'Loss Rate (%)': [8.5, 12.0, 6.3, 7.8, 9.2]
    })
    
    # Save all datasets
    item_master.to_csv(os.path.join(temp_dir, 'annex1.csv'), index=False)
    sales_df.to_csv(os.path.join(temp_dir, 'annex2.csv'), index=False)
    wholesale_df.to_csv(os.path.join(temp_dir, 'annex3.csv'), index=False)
    loss_rates.to_csv(os.path.join(temp_dir, 'annex4.csv'), index=False)
    
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for all outputs"""
    temp_dir = tempfile.mkdtemp()
    
    # Create subdirectories
    subdirs = ['validation', 'processed', 'models', 'monitoring', 'logs']
    for subdir in subdirs:
        os.makedirs(os.path.join(temp_dir, subdir), exist_ok=True)
    
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestEndToEndMLOpsPipeline:
    """End-to-end integration tests for the complete MLOps pipeline"""

    def test_complete_data_to_model_pipeline(self, temp_config, comprehensive_test_data, temp_workspace):
        """Test complete pipeline from raw data to trained model"""
        
        validation_dir = os.path.join(temp_workspace, 'validation')
        processed_dir = os.path.join(temp_workspace, 'processed')
        models_dir = os.path.join(temp_workspace, 'models')
        
        # Step 1: Data Validation
        validator = DataValidator(temp_config)
        validation_results = validator.run_validation(comprehensive_test_data, validation_dir)
        
        assert validation_results['validation_passed'], "Data validation should pass"
        assert validation_results['files_validated'] == 4, "All 4 files should be validated"
        
        # Step 2: Feature Engineering
        feature_engineer = FeatureEngineer(temp_config)
        feature_results = feature_engineer.run_feature_engineering(comprehensive_test_data, processed_dir)
        
        assert 'train' in feature_results, "Training data should be created"
        assert 'validation' in feature_results, "Validation data should be created"
        assert 'test' in feature_results, "Test data should be created"
        
        # Verify feature data quality
        train_df = pd.read_parquet(feature_results['train'])
        assert len(train_df) > 100, "Training data should have sufficient samples"
        assert 'Avg_Price_Target_1d' in train_df.columns, "Target variable should exist"
        
        # Step 3: Model Training
        model_version = f"e2e_test_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup logging
        logs_dir = os.path.join(temp_workspace, 'logs')
        from training.train_model import setup_logging
        logger = setup_logging(model_version, logs_dir)
        
        trainer = ModelTrainer(temp_config, model_version, logger)
        training_results = trainer.run_training_pipeline(processed_dir, models_dir)
        
        assert training_results['status'] == 'success', "Training should complete successfully"
        assert training_results['models_trained'] > 0, "At least one model should be trained"
        assert training_results['best_model'] is not None, "Best model should be selected"
        
        # Verify model files exist
        assert os.path.exists(training_results['saved_files']['best_model']), "Best model file should exist"
        assert os.path.exists(training_results['saved_files']['evaluation']), "Evaluation file should exist"
        
        return training_results

    @patch('boto3.Session')
    def test_model_to_api_pipeline(self, mock_session, temp_config, comprehensive_test_data, temp_workspace):
        """Test pipeline from trained model to API deployment"""
        
        # First train a model
        training_results = self.test_complete_data_to_model_pipeline(temp_config, comprehensive_test_data, temp_workspace)
        
        # Mock AWS for API
        mock_session.return_value.client.return_value = MagicMock()
        
        # Step 4: API Integration with improved mocking
        models_dir = os.path.join(temp_workspace, 'models')
        
        # Create a more comprehensive mock that allows the API to find the models directory
        def mock_exists_side_effect(path):
            # Allow the models directory to be found
            if path in ['models', '../models', '../../models']:
                return False  # No actual models directory in standard locations
            if models_dir in path or 'best_model.pkl' in path:
                return True
            return 'models' in path
        
        def mock_listdir_side_effect(path):
            if path == models_dir or 'models' in path:
                return ['best_model.pkl']
            return []
        
        with patch('os.path.exists', side_effect=mock_exists_side_effect), \
             patch('os.listdir', side_effect=mock_listdir_side_effect), \
             patch('joblib.load') as mock_load:
            
            # Load the actual trained model
            best_model_path = training_results['saved_files']['best_model']
            actual_model = joblib.load(best_model_path)
            mock_load.return_value = actual_model
            
            # Initialize API
            api = SageMakerSyncAPI(temp_config)
            
            # The API should have loaded the model or created best_model alias
            assert len(api.models) > 0, "API should load models"
            assert 'best_model' in api.models, "Best model should be available in API"
            
            # Test prediction
            from inference.api import FeatureInput
            test_features = FeatureInput(
                Total_Quantity=150.0,
                Avg_Price=18.5,
                Transaction_Count=25,
                Month=7,
                DayOfWeek=1,
                IsWeekend=0,
                Price_Volatility=1.2,
                Wholesale_Price=14.0,
                Loss_Rate=8.5
            )
            
            prediction_result = api.predict_single(test_features, 'best_model')
            
            # Verify prediction
            assert prediction_result.predicted_price > 0, "Prediction should be positive"
            assert 0 <= prediction_result.confidence <= 1, "Confidence should be valid"

    def test_monitoring_integration_pipeline(self, temp_config, comprehensive_test_data, temp_workspace):
        """Test monitoring integration with trained models"""
        
        # First train a model
        training_results = self.test_complete_data_to_model_pipeline(temp_config, comprehensive_test_data, temp_workspace)
        
        monitoring_dir = os.path.join(temp_workspace, 'monitoring')
        
        # Step 5: Performance Monitoring
        with patch('boto3.client'):
            monitor = PerformanceMonitor(temp_config, local_mode=True)
            
            # Test current metrics collection
            monitor.collect_current_metrics()
            
            # Verify monitoring setup
            assert hasattr(monitor, 'metrics_history'), "Monitor should have metrics history"
            assert hasattr(monitor, 'config'), "Monitor should have config"
            
            # Test health summary
            health_summary = monitor.get_health_summary()
            
            assert 'overall_status' in health_summary, "Health summary should have overall status"
            assert 'system_health' in health_summary, "Health summary should include system health"

    def test_drift_detection_pipeline(self, temp_config, comprehensive_test_data, temp_workspace):
        """Test drift detection with data pipeline outputs"""
        
        # First create processed data
        processed_dir = os.path.join(temp_workspace, 'processed')
        
        feature_engineer = FeatureEngineer(temp_config)
        feature_results = feature_engineer.run_feature_engineering(comprehensive_test_data, processed_dir)
        
        # Step 6: Drift Detection
        with patch('boto3.client'):
            drift_detector = DriftDetector(temp_config, local_mode=True)
            
            # Load reference and current data
            reference_data_loaded = drift_detector.load_reference_data(feature_results['train'])
            assert reference_data_loaded, "Reference data should load successfully"
            
            # Load current data for comparison
            current_data = pd.read_parquet(feature_results['validation'])
            
            # Detect drift
            drift_results = drift_detector.detect_data_drift(current_data, method='statistical')
            
            # Verify drift detection results
            assert 'overall_drift_detected' in drift_results, "Should have overall drift status"
            assert 'drift_score' in drift_results, "Should have drift score"
            assert 'feature_drift' in drift_results, "Should have feature-level drift"
            assert isinstance(drift_results['overall_drift_detected'], bool), "Drift status should be boolean"

    def test_complete_mlops_workflow(self, temp_config, comprehensive_test_data, temp_workspace):
        """Test the complete MLOps workflow end-to-end"""
        
        # Track pipeline execution times
        pipeline_times = {}
        
        # Step 1: Data Pipeline (Validation + Feature Engineering)
        start_time = time.time()
        
        validation_dir = os.path.join(temp_workspace, 'validation')
        processed_dir = os.path.join(temp_workspace, 'processed')
        
        validator = DataValidator(temp_config)
        validation_results = validator.run_validation(comprehensive_test_data, validation_dir)
        
        feature_engineer = FeatureEngineer(temp_config)
        feature_results = feature_engineer.run_feature_engineering(comprehensive_test_data, processed_dir)
        
        pipeline_times['data_pipeline'] = time.time() - start_time
        
        # Step 2: Training Pipeline
        start_time = time.time()
        
        models_dir = os.path.join(temp_workspace, 'models')
        model_version = f"e2e_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logs_dir = os.path.join(temp_workspace, 'logs')
        from training.train_model import setup_logging
        logger = setup_logging(model_version, logs_dir)
        
        trainer = ModelTrainer(temp_config, model_version, logger)
        training_results = trainer.run_training_pipeline(processed_dir, models_dir)
        
        pipeline_times['training_pipeline'] = time.time() - start_time
        
        # Step 3: Deployment Pipeline (API Setup)
        start_time = time.time()
        
        # Use better mocking for the API
        def mock_exists_side_effect(path):
            if path in ['models', '../models', '../../models']:
                return False  # No models in standard locations
            return models_dir in path or 'best_model.pkl' in path
        
        def mock_listdir_side_effect(path):
            if models_dir in path or path == models_dir:
                return ['best_model.pkl']
            return []
        
        with patch('os.path.exists', side_effect=mock_exists_side_effect), \
             patch('os.listdir', side_effect=mock_listdir_side_effect), \
             patch('joblib.load') as mock_load:
            
            best_model_path = training_results['saved_files']['best_model']
            actual_model = joblib.load(best_model_path)
            mock_load.return_value = actual_model
            
            api = SageMakerSyncAPI(temp_config)
            
            # Test API functionality
            from inference.api import FeatureInput
            test_features = FeatureInput(
                Total_Quantity=150.0,
                Avg_Price=18.5,
                Transaction_Count=25,
                Month=7,
                DayOfWeek=1,
                IsWeekend=0,
                Price_Volatility=1.2,
                Wholesale_Price=14.0,
                Loss_Rate=8.5
            )
            
            prediction_result = api.predict_single(test_features, 'best_model')
        
        pipeline_times['deployment_pipeline'] = time.time() - start_time
        
        # Step 4: Monitoring Pipeline
        start_time = time.time()
        
        with patch('boto3.client'):
            # Performance monitoring
            monitor = PerformanceMonitor(temp_config, local_mode=True)
            health_summary = monitor.get_health_summary()
            
            # Drift detection
            drift_detector = DriftDetector(temp_config, local_mode=True)
            drift_detector.load_reference_data(feature_results['train'])
            current_data = pd.read_parquet(feature_results['validation'])
            drift_results = drift_detector.detect_data_drift(current_data)
        
        pipeline_times['monitoring_pipeline'] = time.time() - start_time
        
        # Verify complete workflow
        workflow_results = {
            'data_validation_passed': validation_results['validation_passed'],
            'features_created': len(pd.read_parquet(feature_results['train']).columns),
            'models_trained': training_results['models_trained'],
            'best_model_mape': None,
            'api_prediction_successful': prediction_result.predicted_price > 0,
            'monitoring_healthy': health_summary['overall_status'] in ['healthy', 'no_data'],
            'drift_detection_completed': 'drift_score' in drift_results,
            'pipeline_times': pipeline_times
        }
        
        # Extract best model performance
        if training_results['best_performance']:
            workflow_results['best_model_mape'] = training_results['best_performance'].get('val_mape')
        
        # Verify all major components completed successfully
        assert workflow_results['data_validation_passed'], "Data validation should pass"
        assert workflow_results['features_created'] > 50, "Should create many features"
        assert workflow_results['models_trained'] > 0, "Should train models"
        assert workflow_results['api_prediction_successful'], "API prediction should work"
        assert workflow_results['monitoring_healthy'], "Monitoring should be healthy"
        assert workflow_results['drift_detection_completed'], "Drift detection should complete"
        
        # Performance checks
        assert pipeline_times['data_pipeline'] < 60, "Data pipeline should complete in reasonable time"
        assert pipeline_times['training_pipeline'] < 120, "Training should complete in reasonable time"
        
        return workflow_results

    def test_error_recovery_workflow(self, temp_config, temp_workspace):
        """Test MLOps pipeline error handling and recovery"""
        
        # Test with invalid data
        empty_dir = tempfile.mkdtemp()
        validation_dir = os.path.join(temp_workspace, 'validation')
        
        try:
            # Should handle missing data gracefully
            validator = DataValidator(temp_config)
            validation_results = validator.run_validation(empty_dir, validation_dir)
            
            assert validation_results['validation_passed'] is False, "Should fail with missing data"
            assert 'recommendations' in validation_results, "Should provide recommendations"
            
        finally:
            shutil.rmtree(empty_dir)

    def test_configuration_impact_on_pipeline(self, comprehensive_test_data, temp_workspace):
        """Test how different configurations affect the pipeline"""
        
        # Test with strict thresholds
        strict_config = {
            'project': {'name': 'strict-test', 'version': '1.0.0'},
            'aws': {'region': 'us-east-1'},
            'evaluation': {
                'thresholds': {
                    'mape_threshold': 5.0,  # Very strict
                    'rmse_threshold': 1.0,
                    'r2_threshold': 0.95
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(strict_config, f)
            strict_config_path = f.name
        
        try:
            processed_dir = os.path.join(temp_workspace, 'processed')
            models_dir = os.path.join(temp_workspace, 'models')
            
            # Run feature engineering
            feature_engineer = FeatureEngineer(strict_config_path)
            feature_results = feature_engineer.run_feature_engineering(comprehensive_test_data, processed_dir)
            
            # Run training with strict thresholds
            model_version = f"strict_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logs_dir = os.path.join(temp_workspace, 'logs')
            from training.train_model import setup_logging
            logger = setup_logging(model_version, logs_dir)
            
            trainer = ModelTrainer(strict_config_path, model_version, logger)
            training_results = trainer.run_training_pipeline(processed_dir, models_dir)
            
            # Should still complete but may not meet strict thresholds
            assert training_results['status'] == 'success', "Training should complete even with strict config"
            
        finally:
            os.unlink(strict_config_path)

    def test_scalability_considerations(self, temp_config, temp_workspace):
        """Test pipeline behavior with different data sizes"""
        
        # Create small dataset
        small_data_dir = tempfile.mkdtemp()
        
        try:
            # Create minimal dataset
            small_sales = pd.DataFrame({
                'Date': ['2024-01-01'] * 10,
                'Time': ['10:00'] * 10,
                'Item Code': [101] * 10,
                'Quantity Sold (kilo)': np.random.uniform(10, 50, 10),
                'Unit Selling Price (RMB/kg)': np.random.uniform(10, 20, 10),
                'Sale or Return': ['Sale'] * 10,
                'Discount (Yes/No)': ['No'] * 10
            })
            
            small_sales.to_csv(os.path.join(small_data_dir, 'annex2.csv'), index=False)
            
            # Create other required files with minimal data
            pd.DataFrame({
                'Item Code': [101],
                'Item Name': ['Apple'],
                'Category Code': [1],
                'Category Name': ['Fruit']
            }).to_csv(os.path.join(small_data_dir, 'annex1.csv'), index=False)
            
            pd.DataFrame({
                'Date': ['2024-01-01'],
                'Item Code': [101],
                'Wholesale Price (RMB/kg)': [10.0]
            }).to_csv(os.path.join(small_data_dir, 'annex3.csv'), index=False)
            
            pd.DataFrame({
                'Item Code': [101],
                'Item Name': ['Apple'],
                'Loss Rate (%)': [8.5]
            }).to_csv(os.path.join(small_data_dir, 'annex4.csv'), index=False)
            
            # Test pipeline with small dataset
            validator = DataValidator(temp_config)
            validation_results = validator.run_validation(small_data_dir, temp_workspace)
            
            # Should handle small datasets gracefully
            assert validation_results['files_validated'] == 4, "Should validate all files even with small data"
            
        finally:
            shutil.rmtree(small_data_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])