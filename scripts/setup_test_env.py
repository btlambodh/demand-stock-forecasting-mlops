#!/usr/bin/env python3
"""
Test Environment Setup Script
Creates test configuration and sample data for unit tests
"""

import os
import yaml
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_test_config():
    """Create test configuration file"""
    print('Creating test configuration...')
    
    test_config = {
        'project': {
            'name': 'demand-stock-forecasting-mlops', 
            'version': '1.0.0'
        },
        'aws': {
            'region': 'us-east-1',
            's3': {
                'bucket_name': 'test-bucket', 
                'data_prefix': 'test-data/'
            },
            'sagemaker': {
                'execution_role': 'arn:aws:iam::123456789012:role/test-role'
            }
        },
        'deployment': {
            'environments': {
                'dev': {
                    'initial_instance_count': 1, 
                    'instance_type': 'ml.t2.medium'
                },
                'staging': {
                    'initial_instance_count': 1, 
                    'instance_type': 'ml.m5.large'
                },
                'prod': {
                    'initial_instance_count': 2, 
                    'instance_type': 'ml.m5.xlarge'
                }
            }
        },
        'models': {
            'default_model': 'random_forest',
            'model_types': ['linear_regression', 'ridge', 'random_forest'],
            'hyperparameters': {
                'random_forest': {
                    'n_estimators': 10, 
                    'random_state': 42
                }
            }
        }
    }
    
    os.makedirs('tests/config', exist_ok=True)
    with open('tests/config/test_config.yaml', 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False)
    
    print('✓ Test configuration created: tests/config/test_config.yaml')


def create_test_data():
    """Create sample test data with correct column names"""
    print('Creating test data...')
    
    np.random.seed(42)
    n_records = 100
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_records)]
    
    data = []
    for i in range(n_records):
        record = {
            'Date': dates[i],
            'Item Code': np.random.choice([101, 102, 103]),
            'Total_Quantity': np.random.uniform(50, 200),
            'Avg_Quantity': np.random.uniform(5, 15),
            'Transaction_Count': np.random.randint(5, 30),
            'Avg_Price': np.random.uniform(10, 25),
            'Price_Volatility': np.random.uniform(0.1, 2.0),
            'Min_Price': np.random.uniform(8, 15),
            'Max_Price': np.random.uniform(15, 30),
            'Discount_Count': np.random.randint(0, 5),
            'Revenue': 0,
            'Discount_Rate': 0,
            'Month': np.random.randint(1, 13),
            'DayOfWeek': np.random.randint(0, 7),
            'IsWeekend': np.random.choice([0, 1]),
            'Category Code': np.random.randint(1, 4),
            'Category Name': np.random.choice(['Vegetables', 'Fruits', 'Herbs']),
            'Wholesale Price (RMB/kg)': np.random.uniform(8, 20),
            'Loss Rate (%)': np.random.uniform(5, 15)
        }
        
        # Calculate derived fields
        record['Revenue'] = record['Total_Quantity'] * record['Avg_Price']
        record['Discount_Rate'] = record['Discount_Count'] / record['Transaction_Count']
        data.append(record)
    
    df = pd.DataFrame(data)
    
    # Add target variables
    for horizon in [1, 7, 14, 30]:
        df[f'Avg_Price_Target_{horizon}d'] = df.groupby('Item Code')['Avg_Price'].shift(-horizon)
        df[f'Quantity_Target_{horizon}d'] = df.groupby('Item Code')['Total_Quantity'].shift(-horizon)
    
    # Add additional features that tests expect
    df['Quarter'] = ((df['Month'] - 1) // 3) + 1
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.year
    df['Price_Range'] = df['Max_Price'] - df['Min_Price']
    
    # Temporal features
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    # Chinese holidays
    df['IsNationalDay'] = ((df['Month'] == 10) & (df['DayOfYear'].between(274, 280))).astype(int)
    df['IsLaborDay'] = ((df['Month'] == 5) & (df['DayOfYear'].between(121, 125))).astype(int)
    
    # Price features
    df['Retail_Wholesale_Ratio'] = df['Avg_Price'] / df['Wholesale Price (RMB/kg)']
    df['Price_Markup'] = df['Avg_Price'] - df['Wholesale Price (RMB/kg)']
    df['Price_Markup_Pct'] = (df['Price_Markup'] / df['Wholesale Price (RMB/kg)']) * 100
    
    # Lag features (simplified for testing)
    for lag in [1, 7, 14, 30]:
        df[f'Avg_Price_Lag_{lag}'] = df['Avg_Price'] * np.random.uniform(0.95, 1.05, len(df))
        df[f'Total_Quantity_Lag_{lag}'] = df['Total_Quantity'] * np.random.uniform(0.9, 1.1, len(df))
        df[f'Revenue_Lag_{lag}'] = df[f'Avg_Price_Lag_{lag}'] * df[f'Total_Quantity_Lag_{lag}']
    
    # Create train/validation/test splits
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size+val_size].copy()
    test_df = df.iloc[train_size+val_size:].copy()
    
    # Create output directory
    os.makedirs('tests/data/processed', exist_ok=True)
    
    # Save datasets
    train_df.to_parquet('tests/data/processed/train.parquet', index=False)
    val_df.to_parquet('tests/data/processed/validation.parquet', index=False)
    test_df.to_parquet('tests/data/processed/test.parquet', index=False)
    df.to_parquet('tests/data/processed/features_complete.parquet', index=False)
    
    print(f'✓ Train data: {len(train_df)} records')
    print(f'✓ Validation data: {len(val_df)} records')
    print(f'✓ Test data: {len(test_df)} records')
    print(f'✓ Total features: {len(df.columns)}')
    
    # Create metadata
    metadata = {
        'feature_columns': list(df.columns),
        'total_features': len(df.columns),
        'train_records': len(train_df),
        'validation_records': len(val_df),
        'test_records': len(test_df),
        'date_range': {
            'start': df['Date'].min().strftime('%Y-%m-%d'),
            'end': df['Date'].max().strftime('%Y-%m-%d')
        },
        'items_count': df['Item Code'].nunique()
    }
    
    with open('tests/data/processed/feature_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print('✓ Feature metadata created')


def create_raw_test_data():
    """Create raw test data files for feature engineering tests"""
    print('Creating raw test data...')
    
    os.makedirs('tests/data/raw', exist_ok=True)
    
    # annex1.csv - item master
    items_df = pd.DataFrame({
        'Item Code': [101, 102, 103, 104, 105],
        'Category Code': [1, 1, 2, 2, 3],
        'Category Name': ['Vegetables', 'Vegetables', 'Fruits', 'Fruits', 'Herbs'],
        'Item Name': ['Tomato', 'Cabbage', 'Apple', 'Orange', 'Basil']
    })
    items_df.to_csv('tests/data/raw/annex1.csv', index=False)
    
    # annex2.csv - sales transactions
    sales_data = []
    for i in range(200):
        sales_data.append({
            'Date': (datetime(2023, 1, 1) + timedelta(days=i % 90)).strftime('%Y-%m-%d'),
            'Item Code': np.random.choice([101, 102, 103, 104, 105]),
            'Quantity Sold (kilo)': np.random.uniform(10, 50),
            'Unit Selling Price (RMB/kg)': np.random.uniform(10, 25),
            'Sale or Return': 'Sale',
            'Discount (Yes/No)': np.random.choice(['Yes', 'No'])
        })
    
    sales_df = pd.DataFrame(sales_data)
    sales_df.to_csv('tests/data/raw/annex2.csv', index=False)
    
    # annex3.csv - wholesale prices
    wholesale_data = []
    for item_code in [101, 102, 103, 104, 105]:
        for i in range(30):
            wholesale_data.append({
                'Date': (datetime(2023, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d'),
                'Item Code': item_code,
                'Wholesale Price (RMB/kg)': np.random.uniform(8, 20)
            })
    
    wholesale_df = pd.DataFrame(wholesale_data)
    wholesale_df.to_csv('tests/data/raw/annex3.csv', index=False)
    
    # annex4.csv - loss rates
    loss_rates_df = pd.DataFrame({
        'Item Code': [101, 102, 103, 104, 105],
        'Loss Rate (%)': [8.5, 12.3, 6.7, 9.1, 15.2]
    })
    loss_rates_df.to_csv('tests/data/raw/annex4.csv', index=False)
    
    print('✓ Raw test data files created')


def create_mock_model():
    """Create a simple mock model for API tests"""
    try:
        import joblib
        from sklearn.ensemble import RandomForestRegressor
        
        print('Creating mock model...')
        
        # Create a simple model
        mock_model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        # Create dummy training data
        X_dummy = np.random.rand(50, 10)
        y_dummy = np.random.rand(50)
        mock_model.fit(X_dummy, y_dummy)
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save as mock model
        mock_artifact = {'model': mock_model, 'scaler': None}
        joblib.dump(mock_artifact, 'models/mock_model.pkl')
        
        print('✓ Mock model created: models/mock_model.pkl')
        
    except Exception as e:
        print(f'⚠️ Could not create mock model: {e}')


def main():
    """Main function"""
    print('=' * 60)
    print('Setting up test environment for unit tests...')
    print('=' * 60)
    
    try:
        create_test_config()
        create_test_data()
        create_raw_test_data()
        create_mock_model()
        
        print('=' * 60)
        print('✓ Test environment setup completed successfully!')
        print('=' * 60)
        print('Files created:')
        print('  tests/config/test_config.yaml')
        print('  tests/data/processed/ (train, validation, test data)')
        print('  tests/data/raw/ (raw data files)')
        print('  models/mock_model.pkl (if sklearn available)')
        print('')
        print('You can now run: make test-unit')
        
    except Exception as e:
        print(f'❌ Error setting up test environment: {e}')
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())