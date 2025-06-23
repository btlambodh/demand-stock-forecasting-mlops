#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for MLOps testing
Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import json

import pytest
import pandas as pd
import numpy as np
import yaml
import boto3
from moto import mock_s3, mock_sagemaker, mock_athena
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ===== PYTEST CONFIGURATION =====

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for component interactions")
    config.addinivalue_line("markers", "deployment: Deployment and infrastructure tests")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")
    config.addinivalue_line("markers", "aws: Tests that require AWS services")

# ===== FIXTURES =====

@pytest.fixture(scope="session")
def test_config():
    """Test configuration"""
    return {
        'project': {
            'name': 'test-demand-stock-forecasting',
            'version': '1.0.0-test'
        },
        'aws': {
            'region': 'us-east-1',
            'account_id': '123456789012',
            's3': {
                'bucket_name': 'test-bucket'
            },
            'sagemaker': {
                'execution_role': 'arn:aws:iam::123456789012:role/test-role'
            }
        },
        'evaluation': {
            'thresholds': {
                'mape_threshold': 15.0,
                'rmse_threshold': 5.0,
                'r2_threshold': 0.7
            }
        },
        'monitoring': {
            'performance': {
                'drift_threshold': 0.25,
                'performance_degradation_threshold': 0.15,
                'cpu_threshold': 80,
                'memory_threshold': 85
            }
        }
    }

@pytest.fixture
def temp_dir():
    """Temporary directory for testing"""
    temp_dir = tempfile.mkdtemp(prefix="mlops_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def config_file(temp_dir, test_config):
    """Create temporary config file"""
    config_path = os.path.join(temp_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    return config_path

@pytest.fixture
def sample_raw_data():
    """Sample raw data for testing"""
    # Item master data (annex1)
    items_data = pd.DataFrame({
        'Item Code': [101, 102, 103, 104, 105],
        'Item Name': ['Apple', 'Banana', 'Orange', 'Grape', 'Pear'],
        'Category Code': [1, 1, 1, 2, 2],
        'Category Name': ['Fruits', 'Fruits', 'Fruits', 'Citrus', 'Citrus']
    })
    
    # Sales transactions (annex2)
    dates = pd.date_range('2024-01-01', periods=100)
    sales_data = []
    for i, date in enumerate(dates):
        for item_code in [101, 102, 103]:
            sales_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Time': '10:00:00',
                'Item Code': item_code,
                'Quantity Sold (kilo)': np.random.uniform(10, 100),
                'Unit Selling Price (RMB/kg)': np.random.uniform(5, 25),
                'Sale or Return': 'Sale',
                'Discount (Yes/No)': 'No' if np.random.random() > 0.2 else 'Yes'
            })
    
    sales_df = pd.DataFrame(sales_data)
    
    # Wholesale prices (annex3)
    wholesale_data = []
    for date in dates:
        for item_code in [101, 102, 103]:
            wholesale_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Item Code': item_code,
                'Wholesale Price (RMB/kg)': np.random.uniform(3, 15)
            })
    
    wholesale_df = pd.DataFrame(wholesale_data)
    
    # Loss rates (annex4)
    loss_rates_data = pd.DataFrame({
        'Item Code': [101, 102, 103, 104, 105],
        'Item Name': ['Apple', 'Banana', 'Orange', 'Grape', 'Pear'],
        'Loss Rate (%)': [5.5, 8.2, 6.1, 7.8, 5.9]
    })
    
    return {
        'items': items_data,
        'sales': sales_df,
        'wholesale': wholesale_df,
        'loss_rates': loss_rates_data
    }

@pytest.fixture
def sample_processed_data():
    """Sample processed data for model training"""
    np.random.seed(42)
    
    # Create sample feature data
    n_samples = 1000
    feature_data = {
        'Date': pd.date_range('2024-01-01', periods=n_samples),
        'Item Code': np.random.choice([101, 102, 103], n_samples),
        'Total_Quantity': np.random.uniform(10, 200, n_samples),
        'Avg_Price': np.random.uniform(8, 30, n_samples),
        'Transaction_Count': np.random.randint(1, 50, n_samples),
        'Price_Volatility': np.random.uniform(0.1, 2.0, n_samples),
        'Month': np.random.randint(1, 13, n_samples),
        'DayOfWeek': np.random.randint(0, 7, n_samples),
        'IsWeekend': np.random.choice([0, 1], n_samples),
        'Revenue': np.random.uniform(100, 5000, n_samples),
        'Wholesale Price (RMB/kg)': np.random.uniform(5, 20, n_samples),
        'Loss Rate (%)': np.random.uniform(3, 15, n_samples)
    }
    
    df = pd.DataFrame(feature_data)
    
    # Add engineered features
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Price_Markup'] = df['Avg_Price'] - df['Wholesale Price (RMB/kg)']
    df['Avg_Price_Target_1d'] = df.groupby('Item Code')['Avg_Price'].shift(-1)
    
    # Remove rows with NaN targets
    df = df.dropna()
    
    return df

@pytest.fixture
def sample_trained_model():
    """Sample trained model for testing"""
    # Create a simple model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    scaler = StandardScaler()
    
    # Generate sample training data
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    
    # Fit the scaler and model
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    
    # Create model artifact
    model_artifact = {
        'model': model,
        'scaler': scaler,
        'feature_names': [f'feature_{i}' for i in range(10)],
        'training_timestamp': datetime.now().isoformat()
    }
    
    return model_artifact

@pytest.fixture
def mock_aws_services():
    """Mock AWS services for testing"""
    with mock_s3(), mock_sagemaker(), mock_athena():
        yield

@pytest.fixture
def sample_model_file(temp_dir, sample_trained_model):
    """Save sample model to file"""
    model_path = os.path.join(temp_dir, 'test_model.pkl')
    joblib.dump(sample_trained_model, model_path)
    return model_path

@pytest.fixture
def sample_evaluation_results():
    """Sample model evaluation results"""
    return {
        'random_forest': {
            'train_mae': 1.2,
            'train_rmse': 1.8,
            'train_r2': 0.85,
            'val_mae': 1.5,
            'val_rmse': 2.1,
            'val_r2': 0.78,
            'val_mape': 12.5,
            'test_mae': 1.6,
            'test_rmse': 2.2,
            'test_r2': 0.76,
            'test_mape': 13.2
        },
        'linear_regression': {
            'train_mae': 2.1,
            'train_rmse': 2.9,
            'train_r2': 0.65,
            'val_mae': 2.3,
            'val_rmse': 3.1,
            'val_r2': 0.62,
            'val_mape': 18.7,
            'test_mae': 2.4,
            'test_rmse': 3.2,
            'test_r2': 0.61,
            'test_mape': 19.1
        }
    }

@pytest.fixture
def save_sample_data(temp_dir, sample_raw_data, sample_processed_data):
    """Save sample data to files"""
    # Create directories
    raw_dir = os.path.join(temp_dir, 'raw')
    processed_dir = os.path.join(temp_dir, 'processed')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save raw data
    sample_raw_data['items'].to_csv(os.path.join(raw_dir, 'annex1.csv'), index=False)
    sample_raw_data['sales'].to_csv(os.path.join(raw_dir, 'annex2.csv'), index=False)
    sample_raw_data['wholesale'].to_csv(os.path.join(raw_dir, 'annex3.csv'), index=False)
    sample_raw_data['loss_rates'].to_csv(os.path.join(raw_dir, 'annex4.csv'), index=False)
    
    # Save processed data
    train_size = int(len(sample_processed_data) * 0.7)
    val_size = int(len(sample_processed_data) * 0.15)
    
    train_df = sample_processed_data.iloc[:train_size]
    val_df = sample_processed_data.iloc[train_size:train_size+val_size]
    test_df = sample_processed_data.iloc[train_size+val_size:]
    
    train_df.to_parquet(os.path.join(processed_dir, 'train.parquet'), index=False)
    val_df.to_parquet(os.path.join(processed_dir, 'validation.parquet'), index=False)
    test_df.to_parquet(os.path.join(processed_dir, 'test.parquet'), index=False)
    sample_processed_data.to_parquet(os.path.join(processed_dir, 'features_complete.parquet'), index=False)
    
    # Save metadata
    metadata = {
        'feature_columns': list(sample_processed_data.columns),
        'total_features': len(sample_processed_data.columns),
        'train_records': len(train_df),
        'validation_records': len(val_df),
        'test_records': len(test_df)
    }
    
    with open(os.path.join(processed_dir, 'feature_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        'raw_dir': raw_dir,
        'processed_dir': processed_dir
    }

# ===== HELPER FUNCTIONS =====

def assert_file_exists(file_path: str, message: str = ""):
    """Assert that a file exists"""
    assert os.path.exists(file_path), f"File does not exist: {file_path}. {message}"

def assert_dataframe_valid(df: pd.DataFrame, min_rows: int = 1, min_cols: int = 1):
    """Assert that DataFrame is valid"""
    assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
    assert len(df) >= min_rows, f"DataFrame has fewer than {min_rows} rows"
    assert len(df.columns) >= min_cols, f"DataFrame has fewer than {min_cols} columns"
    assert not df.empty, "DataFrame is empty"

def assert_metrics_valid(metrics: Dict[str, float], required_metrics: list = None):
    """Assert that metrics dictionary is valid"""
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    
    if required_metrics:
        for metric in required_metrics:
            assert metric in metrics, f"Required metric '{metric}' not found"
            assert isinstance(metrics[metric], (int, float)), f"Metric '{metric}' should be numeric"

def create_temp_model_file(temp_dir: str, model_name: str = "test_model") -> str:
    """Create a temporary model file for testing"""
    model = RandomForestRegressor(n_estimators=5, random_state=42)
    scaler = StandardScaler()
    
    # Fit with dummy data
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    
    model_artifact = {'model': model, 'scaler': scaler}
    model_path = os.path.join(temp_dir, f"{model_name}.pkl")
    joblib.dump(model_artifact, model_path)
    
    return model_path
