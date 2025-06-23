#!/usr/bin/env python3
"""
Integration Tests for Data Pipeline
Tests the interaction between data validation, feature engineering, and feature store components
"""

import os
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import yaml
import json

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data_processing.data_validation import DataValidator
from data_processing.feature_engineering import FeatureEngineer
from data_processing.feature_store_integration import FeatureStoreManager


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
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        return f.name


@pytest.fixture
def temp_data_dir():
    """Create temporary directory with sample data files"""
    temp_dir = tempfile.mkdtemp()
    
    # Create sample CSV files
    sample_data = {
        'annex1.csv': pd.DataFrame({
            'Item Code': [101, 102, 103],
            'Item Name': ['Apple', 'Banana', 'Orange'],
            'Category Code': [1, 1, 2],
            'Category Name': ['Fruit', 'Fruit', 'Citrus']
        }),
        'annex2.csv': pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'] * 10,
            'Time': ['10:00', '11:00', '12:00'] * 10,
            'Item Code': [101, 102, 103] * 10,
            'Quantity Sold (kilo)': np.random.uniform(10, 100, 30),
            'Unit Selling Price (RMB/kg)': np.random.uniform(5, 25, 30),
            'Sale or Return': ['Sale'] * 30,
            'Discount (Yes/No)': ['No'] * 25 + ['Yes'] * 5
        }),
        'annex3.csv': pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'] * 10,
            'Item Code': [101, 102, 103] * 10,
            'Wholesale Price (RMB/kg)': np.random.uniform(3, 20, 30)
        }),
        'annex4.csv': pd.DataFrame({
            'Item Code': [101, 102, 103],
            'Item Name': ['Apple', 'Banana', 'Orange'],
            'Loss Rate (%)': [8.5, 12.0, 6.3]
        })
    }
    
    for filename, df in sample_data.items():
        filepath = os.path.join(temp_dir, filename)
        df.to_csv(filepath, index=False)
    
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestDataPipelineIntegration:
    """Integration tests for the complete data pipeline"""

    def test_validation_to_feature_engineering_flow(self, temp_config, temp_data_dir, temp_output_dir):
        """Test data flows correctly from validation to feature engineering"""
        
        # Step 1: Run data validation
        validator = DataValidator(temp_config)
        validation_results = validator.run_validation(temp_data_dir, temp_output_dir)
        
        # Verify validation completed successfully
        assert validation_results['files_validated'] == 4
        assert validation_results['validation_passed'] is True
        assert validation_results['total_quality_score'] > 70.0
        
        # Step 2: Run feature engineering using validated data
        feature_engineer = FeatureEngineer(temp_config)
        feature_results = feature_engineer.run_feature_engineering(temp_data_dir, temp_output_dir)
        
        # Verify feature engineering outputs
        assert 'train' in feature_results
        assert 'validation' in feature_results
        assert 'test' in feature_results
        assert 'metadata' in feature_results
        
        # Verify files were created
        for output_type, filepath in feature_results.items():
            assert os.path.exists(filepath), f"Missing output file: {filepath}"
        
        # Verify feature data structure
        train_df = pd.read_parquet(feature_results['train'])
        assert len(train_df) > 0, "Training data is empty"
        assert 'Date' in train_df.columns, "Missing Date column"
        assert 'Item Code' in train_df.columns, "Missing Item Code column"
        
        # Verify temporal features were created
        temporal_features = [col for col in train_df.columns if any(x in col for x in ['_Sin', '_Cos', 'Lag_', 'MA_'])]
        assert len(temporal_features) > 0, "No temporal features created"

    @patch('boto3.Session')
    def test_feature_engineering_to_feature_store_flow(self, mock_session, temp_config, temp_data_dir, temp_output_dir):
        """Test flow from feature engineering to feature store integration"""
        
        # Mock AWS clients
        mock_s3 = MagicMock()
        mock_sagemaker = MagicMock()
        mock_session.return_value.client.side_effect = lambda service, **kwargs: {
            's3': mock_s3,
            'sagemaker': mock_sagemaker,
            'sagemaker-featurestore-runtime': MagicMock(),
            'glue': MagicMock(),
            'athena': MagicMock()
        }.get(service, MagicMock())
        
        # Step 1: Create features
        feature_engineer = FeatureEngineer(temp_config)
        feature_results = feature_engineer.run_feature_engineering(temp_data_dir, temp_output_dir)
        
        # Step 2: Initialize feature store manager
        fs_manager = FeatureStoreManager(temp_config)
        
        # Step 3: Test data upload functionality
        upload_results = fs_manager.upload_data_to_s3(temp_output_dir)
        
        # Verify upload was attempted
        assert mock_s3.upload_file.called, "S3 upload was not attempted"
        assert len(upload_results) > 0, "No upload results returned"

    def test_end_to_end_data_pipeline(self, temp_config, temp_data_dir, temp_output_dir):
        """Test complete data pipeline from raw data to processed features"""
        
        # Step 1: Validation
        validator = DataValidator(temp_config)
        validation_results = validator.run_validation(temp_data_dir, temp_output_dir)
        assert validation_results['validation_passed'], "Data validation failed"
        
        # Step 2: Feature Engineering
        feature_engineer = FeatureEngineer(temp_config)
        feature_results = feature_engineer.run_feature_engineering(temp_data_dir, temp_output_dir)
        
        # Step 3: Verify feature metadata
        metadata_file = feature_results['metadata']
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Verify metadata structure
        assert 'total_features' in metadata
        assert 'feature_columns' in metadata
        assert 'train_records' in metadata
        assert 'validation_records' in metadata
        assert 'test_records' in metadata
        
        # Verify data splits are reasonable
        total_records = metadata['train_records'] + metadata['validation_records'] + metadata['test_records']
        assert total_records > 0, "No records in processed data"
        
        train_ratio = metadata['train_records'] / total_records
        assert 0.6 <= train_ratio <= 0.8, f"Training ratio {train_ratio} is not reasonable"

    def test_data_pipeline_error_handling(self, temp_config, temp_output_dir):
        """Test data pipeline handles missing/invalid data gracefully"""
        
        # Create empty data directory
        empty_dir = tempfile.mkdtemp()
        
        try:
            # Test validation with missing files
            validator = DataValidator(temp_config)
            validation_results = validator.run_validation(empty_dir, temp_output_dir)
            
            # Should fail gracefully
            assert validation_results['validation_passed'] is False
            assert validation_results['files_validated'] == 0
            
        finally:
            shutil.rmtree(empty_dir)

    def test_data_pipeline_configuration_handling(self, temp_data_dir, temp_output_dir):
        """Test pipeline handles different configuration scenarios"""
        
        # Test with minimal config
        minimal_config = {
            'project': {'name': 'test'},
            'aws': {'region': 'us-east-1'},
            'evaluation': {'thresholds': {}}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(minimal_config, f)
            minimal_config_path = f.name
        
        try:
            # Should handle minimal config gracefully
            validator = DataValidator(minimal_config_path)
            validation_results = validator.run_validation(temp_data_dir, temp_output_dir)
            
            # Basic validation should still work
            assert validation_results['files_validated'] > 0
            
        finally:
            os.unlink(minimal_config_path)

    def test_feature_data_quality_checks(self, temp_config, temp_data_dir, temp_output_dir):
        """Test that feature engineering produces quality data"""
        
        feature_engineer = FeatureEngineer(temp_config)
        feature_results = feature_engineer.run_feature_engineering(temp_data_dir, temp_output_dir)
        
        # Load and validate each dataset
        for dataset_name in ['train', 'validation', 'test']:
            if dataset_name in feature_results:
                df = pd.read_parquet(feature_results[dataset_name])
                
                # Check for basic data quality
                assert len(df) > 0, f"{dataset_name} dataset is empty"
                
                # Check for excessive missing values
                missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
                assert missing_ratio < 0.5, f"{dataset_name} has too many missing values: {missing_ratio}"
                
                # Check for infinite values
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                infinite_count = np.isinf(df[numeric_cols]).sum().sum()
                assert infinite_count == 0, f"{dataset_name} contains infinite values"
                
                # Verify target columns exist (for forecasting)
                target_cols = [col for col in df.columns if 'Target' in col]
                assert len(target_cols) > 0, f"{dataset_name} missing target columns"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
