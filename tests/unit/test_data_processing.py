#!/usr/bin/env python3
"""
Unit tests for data processing components
Tests data validation, feature engineering, and feature store integration

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import os
import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import modules to test
from src.data_processing.data_validation import DataValidator
from src.data_processing.feature_engineering import FeatureEngineer
from src.data_processing.feature_store_integration import FeatureStoreManager


class TestDataValidator:
    """Unit tests for DataValidator class"""

    @pytest.mark.unit
    def test_validator_initialization(self, config_file):
        """Test DataValidator initialization"""
        validator = DataValidator(config_file)
        assert validator.config is not None
        assert hasattr(validator, 'validation_results')
        assert hasattr(validator, 'expected_schemas')

    @pytest.mark.unit
    def test_file_existence_validation_success(self, config_file, save_sample_data):
        """Test successful file existence validation"""
        validator = DataValidator(config_file)
        raw_dir = save_sample_data['raw_dir']
        
        file_status = validator.validate_file_existence(raw_dir)
        
        assert isinstance(file_status, dict)
        assert all(file_status.values()), "All files should exist"
        assert 'item_master' in file_status
        assert 'sales_transactions' in file_status
        assert 'wholesale_prices' in file_status
        assert 'loss_rates' in file_status

    @pytest.mark.unit
    def test_file_existence_validation_missing_files(self, config_file, temp_dir):
        """Test file existence validation with missing files"""
        validator = DataValidator(config_file)
        
        file_status = validator.validate_file_existence(temp_dir)
        
        assert isinstance(file_status, dict)
        assert all(not status for status in file_status.values()), "All files should be missing"

    @pytest.mark.unit
    def test_schema_validation_valid_data(self, config_file, sample_raw_data):
        """Test schema validation with valid data"""
        validator = DataValidator(config_file)
        items_df = sample_raw_data['items']
        expected_schema = validator.expected_schemas['annex1']
        
        result = validator.validate_schema(items_df, expected_schema, 'test_file')
        
        assert result['columns_match'] is True
        assert result['dtypes_valid'] is True
        assert len(result['missing_columns']) == 0
        assert len(result['extra_columns']) == 0

    @pytest.mark.unit
    def test_schema_validation_missing_columns(self, config_file):
        """Test schema validation with missing columns"""
        validator = DataValidator(config_file)
        # Create DataFrame with missing columns
        incomplete_df = pd.DataFrame({'Item Code': [1, 2, 3]})
        expected_schema = validator.expected_schemas['annex1']
        
        result = validator.validate_schema(incomplete_df, expected_schema, 'test_file')
        
        assert result['columns_match'] is False
        assert len(result['missing_columns']) > 0
        assert 'Item Name' in result['missing_columns']

    @pytest.mark.unit
    def test_data_quality_validation(self, config_file, sample_raw_data):
        """Test data quality validation"""
        validator = DataValidator(config_file)
        sales_df = sample_raw_data['sales']
        
        quality_result = validator.validate_data_quality(sales_df, 'sales_test')
        
        assert quality_result['file_name'] == 'sales_test'
        assert quality_result['total_rows'] > 0
        assert quality_result['total_columns'] > 0
        assert 'missing_data' in quality_result
        assert 'quality_score' in quality_result
        assert 0 <= quality_result['quality_score'] <= 100

    @pytest.mark.unit
    def test_business_rules_validation(self, config_file, sample_raw_data):
        """Test business rules validation"""
        validator = DataValidator(config_file)
        data_dict = {
            'annex1': sample_raw_data['items'],
            'annex2': sample_raw_data['sales'],
            'annex3': sample_raw_data['wholesale'],
            'annex4': sample_raw_data['loss_rates']
        }
        
        business_result = validator.validate_business_rules(data_dict)
        
        assert 'rules_passed' in business_result
        assert 'rules_failed' in business_result
        assert 'rule_results' in business_result
        assert isinstance(business_result['rule_results'], list)
        assert business_result['rules_passed'] >= 0
        assert business_result['rules_failed'] >= 0

    @pytest.mark.unit
    def test_dtype_compatibility_check(self, config_file):
        """Test data type compatibility checking"""
        validator = DataValidator(config_file)
        
        # Test compatible types
        assert validator._dtype_compatible('int64', 'int64') is True
        assert validator._dtype_compatible('int32', 'int64') is True
        assert validator._dtype_compatible('float64', 'float64') is True
        assert validator._dtype_compatible('object', 'object') is True
        
        # Test incompatible types
        assert validator._dtype_compatible('float64', 'int64') is False
        assert validator._dtype_compatible('object', 'int64') is False

    @pytest.mark.unit
    def test_quality_score_calculation(self, config_file):
        """Test quality score calculation"""
        validator = DataValidator(config_file)
        
        # Test with perfect data
        perfect_metrics = {
            'missing_data': {'col1': {'percentage': 0}, 'col2': {'percentage': 0}},
            'duplicate_rows': 0,
            'total_rows': 100,
            'outliers': {'col1': {'percentage': 0}},
            'negative_values': {'price': {'percentage': 0}}
        }
        
        score = validator._calculate_quality_score(perfect_metrics)
        assert score == 100.0
        
        # Test with some issues
        imperfect_metrics = {
            'missing_data': {'col1': {'percentage': 10}, 'col2': {'percentage': 5}},
            'duplicate_rows': 5,
            'total_rows': 100,
            'outliers': {'col1': {'percentage': 8}},
            'negative_values': {'price': {'percentage': 2}}
        }
        
        score = validator._calculate_quality_score(imperfect_metrics)
        assert 0 <= score < 100

    @pytest.mark.unit
    def test_run_validation_complete(self, config_file, save_sample_data, temp_dir):
        """Test complete validation run"""
        validator = DataValidator(config_file)
        raw_dir = save_sample_data['raw_dir']
        output_dir = os.path.join(temp_dir, 'validation_output')
        
        results = validator.run_validation(raw_dir, output_dir)
        
        assert isinstance(results, dict)
        assert 'timestamp' in results
        assert 'files_validated' in results
        assert 'total_quality_score' in results
        assert 'validation_passed' in results
        assert results['files_validated'] > 0
        
        # Check output file was created
        assert os.path.exists(os.path.join(output_dir, 'validation_summary.json'))


class TestFeatureEngineer:
    """Unit tests for FeatureEngineer class"""

    @pytest.mark.unit
    def test_feature_engineer_initialization(self, config_file):
        """Test FeatureEngineer initialization"""
        fe = FeatureEngineer(config_file)
        assert fe.config is not None

    @pytest.mark.unit
    def test_load_raw_data(self, config_file, save_sample_data):
        """Test loading raw data"""
        fe = FeatureEngineer(config_file)
        raw_dir = save_sample_data['raw_dir']
        
        data_dict = fe.load_raw_data(raw_dir)
        
        assert isinstance(data_dict, dict)
        assert 'items' in data_dict
        assert 'sales' in data_dict
        assert 'wholesale' in data_dict
        assert 'loss_rates' in data_dict
        
        # Check data types
        for key, df in data_dict.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0

    @pytest.mark.unit
    def test_create_daily_aggregates(self, config_file, sample_raw_data):
        """Test daily aggregation creation"""
        fe = FeatureEngineer(config_file)
        sales_df = sample_raw_data['sales']
        
        daily_agg = fe.create_daily_aggregates(sales_df)
        
        assert isinstance(daily_agg, pd.DataFrame)
        assert len(daily_agg) > 0
        
        # Check required columns
        required_cols = ['Date', 'Item Code', 'Total_Quantity', 'Avg_Price', 
                        'Transaction_Count', 'Revenue', 'Discount_Rate']
        for col in required_cols:
            assert col in daily_agg.columns
        
        # Check data integrity
        assert daily_agg['Total_Quantity'].min() >= 0
        assert daily_agg['Avg_Price'].min() >= 0
        assert daily_agg['Transaction_Count'].min() >= 0

    @pytest.mark.unit
    def test_add_temporal_features(self, config_file):
        """Test temporal feature creation"""
        fe = FeatureEngineer(config_file)
        
        # Create sample data with dates
        sample_df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=10),
            'Item Code': [101] * 10,
            'Avg_Price': [15.0] * 10
        })
        
        result_df = fe.add_temporal_features(sample_df)
        
        # Check temporal features were added
        temporal_features = ['Year', 'Month', 'Quarter', 'DayOfYear', 'DayOfWeek',
                           'Month_Sin', 'Month_Cos', 'IsWeekend', 'Season']
        for feature in temporal_features:
            assert feature in result_df.columns
        
        # Check value ranges
        assert result_df['Month'].min() >= 1
        assert result_df['Month'].max() <= 12
        assert result_df['DayOfWeek'].min() >= 0
        assert result_df['DayOfWeek'].max() <= 6
        assert result_df['IsWeekend'].isin([0, 1]).all()

    @pytest.mark.unit
    def test_add_lag_features(self, config_file):
        """Test lag feature creation"""
        fe = FeatureEngineer(config_file)
        
        # Create sample data
        sample_df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=50),
            'Item Code': [101] * 25 + [102] * 25,
            'Avg_Price': np.random.uniform(10, 20, 50),
            'Total_Quantity': np.random.uniform(50, 150, 50),
            'Revenue': np.random.uniform(500, 3000, 50)
        })
        
        result_df = fe.add_lag_features(sample_df)
        
        # Check lag features were added
        lag_features = ['Avg_Price_Lag_1', 'Avg_Price_Lag_7', 'Total_Quantity_Lag_1', 'Revenue_Lag_1']
        for feature in lag_features:
            assert feature in result_df.columns
        
        # Check that lag features have some NaN values (as expected)
        assert result_df['Avg_Price_Lag_1'].isnull().sum() >= 2  # At least 1 per item

    @pytest.mark.unit
    def test_add_rolling_features(self, config_file):
        """Test rolling window feature creation"""
        fe = FeatureEngineer(config_file)
        
        # Create sample data
        sample_df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=50),
            'Item Code': [101] * 50,
            'Avg_Price': np.random.uniform(10, 20, 50),
            'Total_Quantity': np.random.uniform(50, 150, 50),
            'Revenue': np.random.uniform(500, 3000, 50)
        })
        
        result_df = fe.add_rolling_features(sample_df)
        
        # Check rolling features were added
        rolling_features = ['Avg_Price_MA_7', 'Avg_Price_MA_14', 'Total_Quantity_MA_7', 
                          'Avg_Price_Std_7', 'Revenue_MA_7']
        for feature in rolling_features:
            assert feature in result_df.columns
        
        # Check that moving averages are reasonable
        assert result_df['Avg_Price_MA_7'].notna().sum() > 0

    @pytest.mark.unit
    def test_handle_missing_values(self, config_file):
        """Test missing value handling"""
        fe = FeatureEngineer(config_file)
        
        # Create data with missing values
        sample_df = pd.DataFrame({
            'Item Code': [101, 102, 103, 101, 102],
            'Avg_Price': [15.0, np.nan, 18.0, 16.0, np.nan],
            'Total_Quantity': [100, 120, np.nan, 110, 125],
            'Category': ['A', 'B', None, 'A', 'B']
        })
        
        result_df = fe.handle_missing_values(sample_df)
        
        # Check that numeric missing values are filled
        assert result_df['Avg_Price'].isnull().sum() == 0
        assert result_df['Total_Quantity'].isnull().sum() == 0
        
        # Check that categorical missing values are filled
        assert result_df['Category'].isnull().sum() == 0

    @pytest.mark.unit
    def test_encode_categorical_features(self, config_file):
        """Test categorical feature encoding"""
        fe = FeatureEngineer(config_file)
        
        # Create data with categorical features
        sample_df = pd.DataFrame({
            'Item Code': [101, 102, 103],
            'Season': ['Spring', 'Summer', 'Winter'],
            'Category Name': ['Fruits', 'Vegetables', 'Fruits']
        })
        
        result_df = fe.encode_categorical_features(sample_df)
        
        # Check that season dummies were created
        season_cols = [col for col in result_df.columns if col.startswith('Season_')]
        assert len(season_cols) > 0
        
        # Check that category encoding was created
        assert 'Category Name_Encoded' in result_df.columns
        
        # Original categorical columns should be removed/encoded
        assert 'Season' not in result_df.columns or result_df.columns.tolist().count('Season') == 0

    @pytest.mark.unit
    def test_create_target_variables(self, config_file):
        """Test target variable creation"""
        fe = FeatureEngineer(config_file)
        
        # Create sample data
        sample_df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=50),
            'Item Code': [101] * 25 + [102] * 25,
            'Avg_Price': np.random.uniform(10, 20, 50)
        })
        
        result_df = fe.create_target_variables(sample_df)
        
        # Check target features were added
        target_features = ['Avg_Price_Target_1d', 'Avg_Price_Target_7d', 
                          'Price_Change_Target_1d', 'Quantity_Target_1d']
        for feature in target_features:
            assert feature in result_df.columns
        
        # Check that target features have some NaN values at the end
        assert result_df['Avg_Price_Target_1d'].isnull().sum() >= 2  # At least 1 per item


class TestFeatureStoreManager:
    """Unit tests for FeatureStoreManager class"""

    @pytest.mark.unit
    def test_feature_store_manager_initialization(self, config_file):
        """Test FeatureStoreManager initialization"""
        with patch('boto3.Session'):
            fsm = FeatureStoreManager(config_file)
            assert fsm.config is not None
            assert hasattr(fsm, 'project_name')
            assert hasattr(fsm, 'bucket_name')

    @pytest.mark.unit
    def test_clean_column_name(self, config_file):
        """Test column name cleaning for Feature Store"""
        with patch('boto3.Session'):
            fsm = FeatureStoreManager(config_file)
            
            # Test various problematic column names
            test_cases = [
                ('Wholesale Price (RMB/kg)', 'Wholesale_Price_RMB_kg'),
                ('Loss Rate (%)', 'Loss_Rate_pct'),
                ('Price/Quantity Ratio', 'Price_Quantity_Ratio'),
                ('Item-Code', 'Item_Code'),
                ('Price$$$', 'Price_dollar_dollar_dollar'),
                ('', 'unnamed_feature'),
                ('123invalid', 'feature_123invalid')
            ]
            
            for input_name, expected_output in test_cases:
                result = fsm.clean_column_name(input_name)
                assert isinstance(result, str)
                assert len(result) <= 64
                # Check that result is valid identifier-like
                assert result.replace('_', '').replace('feature', '').replace('pct', '').replace('dollar', '').replace('kg', '').isalnum() or result == 'unnamed_feature'

    @pytest.mark.unit
    def test_map_pandas_to_athena_type(self, config_file):
        """Test pandas to Athena type mapping"""
        with patch('boto3.Session'):
            fsm = FeatureStoreManager(config_file)
            
            # Test type mappings
            test_cases = [
                ('Date', pd.Timestamp('2024-01-01'), 'timestamp'),
                ('Item Code', 101, 'bigint'),
                ('Price', 15.5, 'double'),
                ('Name', 'Apple', 'string'),
                ('Is_Weekend', True, 'boolean'),
                ('Category Code', np.int64(1), 'bigint'),
                ('Unknown', None, 'string')
            ]
            
            for col_name, sample_value, expected_type in test_cases:
                result = fsm.map_pandas_to_athena_type(col_name, sample_value)
                assert result == expected_type

    @pytest.mark.unit
    @patch('boto3.Session')
    def test_create_feature_definitions(self, mock_session, config_file, sample_processed_data):
        """Test feature definition creation"""
        fsm = FeatureStoreManager(config_file)
        
        # Use subset of processed data
        df_subset = sample_processed_data[['Total_Quantity', 'Avg_Price', 'Month', 'IsWeekend']].head(10)
        
        feature_definitions = fsm.create_feature_definitions(df_subset)
        
        assert isinstance(feature_definitions, list)
        assert len(feature_definitions) > 0
        
        # Check structure of feature definitions
        for feature_def in feature_definitions:
            assert 'FeatureName' in feature_def
            assert 'FeatureType' in feature_def
            assert feature_def['FeatureType'] in ['Integral', 'Fractional', 'String']

    @pytest.mark.unit
    @patch('boto3.Session')
    def test_upload_data_to_s3_mock(self, mock_session, config_file, save_sample_data):
        """Test S3 data upload (mocked)"""
        # Mock S3 client
        mock_s3_client = Mock()
        mock_session.return_value.client.return_value = mock_s3_client
        
        fsm = FeatureStoreManager(config_file)
        fsm.s3_client = mock_s3_client
        
        processed_dir = save_sample_data['processed_dir']
        
        # Mock successful upload
        mock_s3_client.upload_file.return_value = None
        
        results = fsm.upload_data_to_s3(processed_dir)
        
        assert isinstance(results, dict)
        # Should have attempted to upload several files
        assert mock_s3_client.upload_file.call_count >= 1

    @pytest.mark.unit
    @patch('boto3.Session')
    def test_create_glue_database_mock(self, mock_session, config_file):
        """Test Glue database creation (mocked)"""
        # Mock Glue client
        mock_glue_client = Mock()
        mock_session.return_value.client.return_value = mock_glue_client
        
        fsm = FeatureStoreManager(config_file)
        fsm.glue_client = mock_glue_client
        
        # Mock successful database creation
        mock_glue_client.create_database.return_value = {'DatabaseArn': 'test-arn'}
        
        result = fsm.create_glue_database()
        
        assert result is True
        mock_glue_client.create_database.assert_called_once()
