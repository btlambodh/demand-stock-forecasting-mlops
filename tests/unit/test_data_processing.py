#!/usr/bin/env python3
"""
Unit Tests for Data Processing Components
Chinese Produce Market Forecasting MLOps Project

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import json
import yaml
from unittest.mock import patch, Mock, mock_open
from datetime import datetime, timedelta

# Import modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# For now, we'll create mock classes since the actual modules might not exist yet
# These can be replaced with actual imports once the modules are implemented


class MockDataValidator:
    """Mock DataValidator for testing purposes"""
    
    def __init__(self, config_path):
        self.config_path = config_path
        self.validation_results = {}
    
    def validate_file_existence(self, data_path):
        """Mock file existence validation"""
        return {
            'item_master': True,
            'sales_transactions': True,
            'wholesale_prices': True,
            'loss_rates': True
        }
    
    def validate_schema(self, df, expected_schema, file_name):
        """Mock schema validation"""
        return {
            'file_name': file_name,
            'columns_match': True,
            'dtypes_valid': True,
            'missing_columns': [],
            'extra_columns': [],
            'dtype_issues': []
        }
    
    def validate_data_quality(self, df, file_name):
        """Mock data quality validation"""
        return {
            'file_name': file_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data': {},
            'duplicate_rows': 0,
            'outliers': {},
            'date_range': {},
            'negative_values': {},
            'quality_score': 85.0
        }
    
    def run_validation(self, data_path, output_path):
        """Mock validation run"""
        return {
            'timestamp': datetime.now().isoformat(),
            'data_path': data_path,
            'files_validated': 4,
            'total_quality_score': 85.0,
            'validation_passed': True,
            'recommendations': []
        }


class MockFeatureEngineer:
    """Mock FeatureEngineer for testing purposes"""
    
    def __init__(self, config):
        self.config = config
        self.feature_columns = []
    
    def create_temporal_features(self, df):
        """Mock temporal feature creation"""
        result = df.copy()
        if 'Date' in df.columns:
            result['Year'] = 2024
            result['Month'] = 6
            result['DayOfWeek'] = 1
            result['DayOfMonth'] = 15
            result['WeekOfYear'] = 24
            result['Quarter'] = 2
            result['IsWeekend'] = False
        return result
    
    def create_statistical_features(self, df):
        """Mock statistical feature creation"""
        result = df.copy()
        if 'Total_Quantity' in df.columns:
            result['Total_Quantity_log'] = np.log1p(df['Total_Quantity'])
            result['Total_Quantity_squared'] = df['Total_Quantity'] ** 2
        return result
    
    def process_features(self, df):
        """Mock feature processing"""
        result = self.create_temporal_features(df)
        result = self.create_statistical_features(result)
        return result


# Fixtures
@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'data': {
            'raw_data_path': '/tmp/raw_data',
            'processed_data_path': '/tmp/processed_data'
        },
        'validation': {
            'quality_threshold': 70.0
        },
        'features': {
            'temporal_features': True,
            'statistical_features': True
        }
    }


@pytest.fixture
def temp_directory():
    """Create temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_raw_data():
    """Sample raw data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    data = {
        'Date': dates,
        'Product_Name': np.random.choice(['Apple', 'Orange', 'Banana', 'Grape'], 100),
        'Location': np.random.choice(['Beijing', 'Shanghai', 'Guangzhou'], 100),
        'Category': np.random.choice(['Fruit', 'Vegetable'], 100),
        'Total_Quantity': np.random.normal(100, 20, 100).clip(1, None),
        'Avg_Price': np.random.normal(15, 3, 100).clip(1, None),
        'Transaction_Count': np.random.poisson(25, 100)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_processed_data():
    """Sample processed data for testing"""
    np.random.seed(42)
    
    data = {
        'Total_Quantity': np.random.normal(0, 1, 100),  # Standardized
        'Avg_Price': np.random.normal(0, 1, 100),       # Standardized
        'Transaction_Count': np.random.normal(0, 1, 100),  # Standardized
        'Year': np.repeat(2024, 100),
        'Month': np.random.randint(1, 13, 100),
        'DayOfWeek': np.random.randint(0, 7, 100)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def config_file(temp_directory):
    """Create a temporary config file"""
    config = {
        'data': {
            'raw_data_path': temp_directory,
            'processed_data_path': temp_directory
        },
        'validation': {
            'quality_threshold': 70.0
        }
    }
    
    config_path = os.path.join(temp_directory, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


class TestDataValidator:
    """Test cases for DataValidator class"""
    
    def test_validator_initialization(self, config_file):
        """Test DataValidator initialization"""
        validator = MockDataValidator(config_file)
        assert validator.config_path == config_file
        assert hasattr(validator, 'validation_results')
    
    @pytest.mark.unit
    def test_validate_file_existence(self, config_file, temp_directory):
        """Test file existence validation"""
        validator = MockDataValidator(config_file)
        
        # Create test files
        test_files = ['annex1.csv', 'annex2.csv', 'annex3.csv', 'annex4.csv']
        for filename in test_files:
            filepath = os.path.join(temp_directory, filename)
            pd.DataFrame({'test': [1, 2, 3]}).to_csv(filepath, index=False)
        
        result = validator.validate_file_existence(temp_directory)
        
        assert isinstance(result, dict)
        assert len(result) == 4
        assert all(result.values())  # All should be True
    
    @pytest.mark.unit
    def test_validate_schema_valid(self, config_file, sample_raw_data):
        """Test schema validation with valid data"""
        validator = MockDataValidator(config_file)
        
        expected_schema = {
            'columns': list(sample_raw_data.columns),
            'dtypes': {col: str(dtype) for col, dtype in sample_raw_data.dtypes.items()}
        }
        
        result = validator.validate_schema(sample_raw_data, expected_schema, 'test_file')
        
        assert result['file_name'] == 'test_file'
        assert result['columns_match'] is True
        assert result['dtypes_valid'] is True
        assert len(result['missing_columns']) == 0
        assert len(result['extra_columns']) == 0
    
    @pytest.mark.unit
    def test_validate_data_quality(self, config_file, sample_raw_data):
        """Test data quality validation"""
        validator = MockDataValidator(config_file)
        
        result = validator.validate_data_quality(sample_raw_data, 'test_file')
        
        assert result['file_name'] == 'test_file'
        assert result['total_rows'] == len(sample_raw_data)
        assert result['total_columns'] == len(sample_raw_data.columns)
        assert 'missing_data' in result
        assert 'duplicate_rows' in result
        assert 'quality_score' in result
        assert 0 <= result['quality_score'] <= 100
    
    @pytest.mark.unit
    def test_run_validation_success(self, config_file, temp_directory):
        """Test successful validation run"""
        validator = MockDataValidator(config_file)
        
        # Create test data files
        test_data = pd.DataFrame({
            'Item Code': [1, 2, 3],
            'Item Name': ['Apple', 'Orange', 'Banana'],
            'Category Code': [1, 1, 1],
            'Category Name': ['Fruit', 'Fruit', 'Fruit']
        })
        
        test_files = ['annex1.csv', 'annex2.csv', 'annex3.csv', 'annex4.csv']
        for filename in test_files:
            filepath = os.path.join(temp_directory, filename)
            test_data.to_csv(filepath, index=False)
        
        result = validator.run_validation(temp_directory, temp_directory)
        
        assert result['validation_passed'] is True
        assert result['files_validated'] > 0
        assert result['total_quality_score'] > 0
        assert 'timestamp' in result


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class"""
    
    def test_feature_engineer_initialization(self, sample_config):
        """Test FeatureEngineer initialization"""
        engineer = MockFeatureEngineer(sample_config)
        assert engineer.config == sample_config
        assert hasattr(engineer, 'feature_columns')
    
    @pytest.mark.unit
    def test_create_temporal_features(self, sample_raw_data):
        """Test temporal feature creation"""
        engineer = MockFeatureEngineer({})
        
        # Ensure Date column is datetime
        sample_raw_data['Date'] = pd.to_datetime(sample_raw_data['Date'])
        
        result = engineer.create_temporal_features(sample_raw_data)
        
        # Check that temporal features are created
        temporal_features = [
            'Year', 'Month', 'DayOfWeek', 'DayOfMonth', 
            'WeekOfYear', 'Quarter', 'IsWeekend'
        ]
        
        for feature in temporal_features:
            assert feature in result.columns
    
    @pytest.mark.unit
    def test_create_statistical_features(self, sample_raw_data):
        """Test statistical feature creation"""
        engineer = MockFeatureEngineer({})
        
        result = engineer.create_statistical_features(sample_raw_data)
        
        # Check for statistical features
        expected_features = ['Total_Quantity_log', 'Total_Quantity_squared']
        
        for feature in expected_features:
            if 'Total_Quantity' in sample_raw_data.columns:
                assert feature in result.columns
    
    @pytest.mark.unit
    def test_process_features_end_to_end(self, sample_raw_data, temp_directory):
        """Test complete feature processing pipeline"""
        config = {
            'data': {
                'processed_data_path': temp_directory
            }
        }
        
        engineer = MockFeatureEngineer(config)
        
        # Ensure Date column is datetime
        sample_raw_data['Date'] = pd.to_datetime(sample_raw_data['Date'])
        
        result = engineer.process_features(sample_raw_data)
        
        # Check that result is not empty
        assert not result.empty
        
        # Check that we have more features than we started with
        assert len(result.columns) >= len(sample_raw_data.columns)
        
        # Check that target variable is preserved
        assert 'Total_Quantity' in result.columns


class TestDataProcessingIntegration:
    """Integration tests for data processing components"""
    
    @pytest.mark.integration
    def test_validation_then_feature_engineering(self, config_file, sample_raw_data, temp_directory):
        """Test integration between validation and feature engineering"""
        # First validate
        validator = MockDataValidator(config_file)
        
        validation_result = validator.run_validation(temp_directory, temp_directory)
        assert validation_result['validation_passed'] is True
        
        # Then engineer features
        config = {
            'data': {
                'processed_data_path': temp_directory
            }
        }
        
        engineer = MockFeatureEngineer(config)
        
        # Ensure Date column is datetime
        sample_raw_data['Date'] = pd.to_datetime(sample_raw_data['Date'])
        
        processed_data = engineer.process_features(sample_raw_data)
        
        # Processed data should have same or more features
        assert len(processed_data.columns) >= len(sample_raw_data.columns)
        
        # Should have same number of rows
        assert len(processed_data) == len(sample_raw_data)


class TestDataProcessingUtils:
    """Test utility functions for data processing"""
    
    @pytest.mark.unit
    def test_missing_value_detection(self, sample_raw_data):
        """Test missing value detection"""
        # Add some missing values
        data_with_missing = sample_raw_data.copy()
        data_with_missing.loc[0:5, 'Total_Quantity'] = np.nan
        
        missing_counts = data_with_missing.isnull().sum()
        
        assert missing_counts['Total_Quantity'] == 6
        assert missing_counts['Product_Name'] == 0
    
    @pytest.mark.unit
    def test_outlier_detection(self, sample_raw_data):
        """Test outlier detection using IQR method"""
        quantity_col = sample_raw_data['Total_Quantity']
        
        Q1 = quantity_col.quantile(0.25)
        Q3 = quantity_col.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((quantity_col < lower_bound) | (quantity_col > upper_bound)).sum()
        
        # Should be a reasonable number of outliers
        assert outliers >= 0
        assert outliers < len(sample_raw_data) * 0.2  # Less than 20%
    
    @pytest.mark.unit
    def test_duplicate_detection(self):
        """Test duplicate detection"""
        # Create data with duplicates
        data = pd.DataFrame({
            'col1': [1, 2, 1, 3],
            'col2': ['a', 'b', 'a', 'c']
        })
        
        duplicate_count = data.duplicated().sum()
        
        assert duplicate_count == 1  # One duplicate row
    
    @pytest.mark.unit
    def test_data_type_validation(self, sample_raw_data):
        """Test data type validation"""
        expected_types = {
            'Date': 'datetime64[ns]',
            'Product_Name': 'object',
            'Total_Quantity': 'float64',
            'Transaction_Count': 'int64'
        }
        
        # Convert Date to datetime for testing
        sample_raw_data['Date'] = pd.to_datetime(sample_raw_data['Date'])
        
        for col, expected_type in expected_types.items():
            if col in sample_raw_data.columns:
                actual_type = str(sample_raw_data[col].dtype)
                # For object types, just check if it's object
                if expected_type == 'object':
                    assert actual_type == 'object'
                elif 'datetime' in expected_type:
                    assert 'datetime' in actual_type
                # For numeric types, check general category
                elif 'float' in expected_type:
                    assert 'float' in actual_type
                elif 'int' in expected_type:
                    assert 'int' in actual_type


if __name__ == "__main__":
    pytest.main([__file__])