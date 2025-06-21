#!/usr/bin/env python3
"""
Feature Engineering Pipeline for Chinese Produce Market Forecasting
Creates ML-ready features from raw market data

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import argparse
import logging
import os
from datetime import datetime
from typing import Dict

import pandas as pd
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for produce market forecasting"""
    
    def __init__(self, config_path: str):
        """Initialize feature engineer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def load_raw_data(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """Load and initial preprocessing of raw CSV files"""
        logger.info("Loading raw data files...")
        
        data_dict = {}
        
        try:
            # Load item master data (annex1)
            annex1_path = os.path.join(data_path, 'annex1.csv')
            if os.path.exists(annex1_path):
                data_dict['items'] = pd.read_csv(annex1_path)
                logger.info(f"Loaded items: {len(data_dict['items'])} records")
            
            # Load sales transactions (annex2)
            annex2_path = os.path.join(data_path, 'annex2.csv')
            if os.path.exists(annex2_path):
                sales_df = pd.read_csv(annex2_path)
                sales_df['Date'] = pd.to_datetime(sales_df['Date'], errors='coerce')
                data_dict['sales'] = sales_df
                logger.info(f"Loaded sales: {len(data_dict['sales'])} transactions")
            
            # Load wholesale prices (annex3)
            annex3_path = os.path.join(data_path, 'annex3.csv')
            if os.path.exists(annex3_path):
                wholesale_df = pd.read_csv(annex3_path)
                wholesale_df['Date'] = pd.to_datetime(wholesale_df['Date'], errors='coerce')
                data_dict['wholesale'] = wholesale_df
                logger.info(f"Loaded wholesale: {len(data_dict['wholesale'])} price records")
            
            # Load loss rates (annex4)
            annex4_path = os.path.join(data_path, 'annex4.csv')
            if os.path.exists(annex4_path):
                data_dict['loss_rates'] = pd.read_csv(annex4_path)
                logger.info(f"Loaded loss rates: {len(data_dict['loss_rates'])} items")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
        
        return data_dict

    def create_daily_aggregates(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """Create daily aggregated sales features"""
        logger.info("Creating daily sales aggregates...")
        
        # Filter out return transactions and invalid dates
        sales_clean = sales_df[
            (sales_df['Sale or Return'].str.lower() == 'sale') & 
            (sales_df['Date'].notna())
        ].copy()
        
        logger.info(f"Processing {len(sales_clean)} sale transactions...")
        
        # Daily aggregation by item
        daily_agg = sales_clean.groupby(['Date', 'Item Code']).agg({
            'Quantity Sold (kilo)': ['sum', 'mean', 'count'],
            'Unit Selling Price (RMB/kg)': ['mean', 'std', 'min', 'max'],
            'Discount (Yes/No)': lambda x: (x == 'Yes').sum()
        }).reset_index()
        
        # Flatten column names
        daily_agg.columns = [
            'Date', 'Item Code', 'Total_Quantity', 'Avg_Quantity', 'Transaction_Count',
            'Avg_Price', 'Price_Volatility', 'Min_Price', 'Max_Price', 'Discount_Count'
        ]
        
        # Fill NaN values
        daily_agg['Price_Volatility'].fillna(0, inplace=True)
        
        # Calculate additional metrics
        daily_agg['Revenue'] = daily_agg['Total_Quantity'] * daily_agg['Avg_Price']
        daily_agg['Discount_Rate'] = daily_agg['Discount_Count'] / daily_agg['Transaction_Count']
        daily_agg['Discount_Rate'].fillna(0, inplace=True)
        daily_agg['Price_Range'] = daily_agg['Max_Price'] - daily_agg['Min_Price']
        
        logger.info(f"Created daily aggregates: {len(daily_agg)} records")
        return daily_agg

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features"""
        logger.info("Adding temporal features...")
        
        df = df.copy()
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Cyclical encoding for seasonality
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfYear_Sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['DayOfYear_Cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
        # Chinese holidays (simplified)
        df['IsNationalDay'] = ((df['Month'] == 10) & (df['DayOfYear'].between(274, 280))).astype(int)
        df['IsLaborDay'] = ((df['Month'] == 5) & (df['DayOfYear'].between(121, 125))).astype(int)
        
        # Agricultural seasons
        df['Season'] = ((df['Month'] % 12 + 3) // 3).map({
            1: 'Spring', 2: 'Summer', 3: 'Autumn', 4: 'Winter'
        })
        
        # Days since epoch for trend
        epoch = pd.to_datetime('2020-01-01')
        df['Days_Since_Epoch'] = (df['Date'] - epoch).dt.days
        
        return df

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for time series"""
        logger.info("Adding lag features...")
        
        df = df.copy()
        lag_periods = [1, 7, 14, 30]
        
        for lag in lag_periods:
            df[f'Avg_Price_Lag_{lag}'] = df.groupby('Item Code')['Avg_Price'].shift(lag)
            df[f'Total_Quantity_Lag_{lag}'] = df.groupby('Item Code')['Total_Quantity'].shift(lag)
            df[f'Revenue_Lag_{lag}'] = df.groupby('Item Code')['Revenue'].shift(lag)
        
        return df

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features"""
        logger.info("Adding rolling window features...")
        
        df = df.copy()
        windows = [7, 14, 30]
        
        for window in windows:
            # Rolling averages
            df[f'Avg_Price_MA_{window}'] = df.groupby('Item Code')['Avg_Price'].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'Total_Quantity_MA_{window}'] = df.groupby('Item Code')['Total_Quantity'].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'Revenue_MA_{window}'] = df.groupby('Item Code')['Revenue'].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
            
            # Rolling volatility
            df[f'Avg_Price_Std_{window}'] = df.groupby('Item Code')['Avg_Price'].rolling(window, min_periods=1).std().reset_index(level=0, drop=True)
            df[f'Total_Quantity_Std_{window}'] = df.groupby('Item Code')['Total_Quantity'].rolling(window, min_periods=1).std().reset_index(level=0, drop=True)
            
            # Rolling min/max
            df[f'Avg_Price_Min_{window}'] = df.groupby('Item Code')['Avg_Price'].rolling(window, min_periods=1).min().reset_index(level=0, drop=True)
            df[f'Avg_Price_Max_{window}'] = df.groupby('Item Code')['Avg_Price'].rolling(window, min_periods=1).max().reset_index(level=0, drop=True)
        
        return df

    def add_price_features(self, daily_agg: pd.DataFrame, wholesale_df: pd.DataFrame) -> pd.DataFrame:
        """Add price-related features"""
        logger.info("Adding price features...")
        
        # Merge with wholesale prices
        merged_df = daily_agg.merge(
            wholesale_df[['Date', 'Item Code', 'Wholesale Price (RMB/kg)']],
            on=['Date', 'Item Code'],
            how='left'
        )
        
        # Price ratio features
        merged_df['Retail_Wholesale_Ratio'] = (
            merged_df['Avg_Price'] / merged_df['Wholesale Price (RMB/kg)']
        ).fillna(1.0)
        
        merged_df['Price_Markup'] = (
            merged_df['Avg_Price'] - merged_df['Wholesale Price (RMB/kg)']
        ).fillna(0)
        
        merged_df['Price_Markup_Pct'] = (
            merged_df['Price_Markup'] / merged_df['Wholesale Price (RMB/kg)'] * 100
        ).fillna(0)
        
        # Price change features
        merged_df['Avg_Price_Change'] = merged_df.groupby('Item Code')['Avg_Price'].pct_change()
        merged_df['Wholesale_Price_Change'] = merged_df.groupby('Item Code')['Wholesale Price (RMB/kg)'].pct_change()
        
        return merged_df

    def add_category_features(self, df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
        """Add category-based features"""
        logger.info("Adding category features...")
        
        # Merge with item categories
        df = df.merge(
            items_df[['Item Code', 'Category Code', 'Category Name']],
            on='Item Code',
            how='left'
        )
        
        # Category-level aggregations
        category_stats = df.groupby(['Date', 'Category Code']).agg({
            'Total_Quantity': 'sum',
            'Avg_Price': 'mean',
            'Revenue': 'sum'
        }).reset_index()
        
        category_stats.columns = [
            'Date', 'Category Code', 'Category_Total_Quantity',
            'Category_Avg_Price', 'Category_Revenue'
        ]
        
        # Merge back category statistics
        df = df.merge(category_stats, on=['Date', 'Category Code'], how='left')
        
        # Item's share within category
        df['Item_Quantity_Share'] = (
            df['Total_Quantity'] / df['Category_Total_Quantity']
        ).fillna(0)
        
        df['Item_Revenue_Share'] = (
            df['Revenue'] / df['Category_Revenue']
        ).fillna(0)
        
        # Price relative to category average
        df['Price_Relative_to_Category'] = (
            df['Avg_Price'] / df['Category_Avg_Price']
        ).fillna(1.0)
        
        return df

    def add_loss_rate_features(self, df: pd.DataFrame, loss_rates_df: pd.DataFrame) -> pd.DataFrame:
        """Add loss rate features"""
        logger.info("Adding loss rate features...")
        
        # Merge with loss rates
        df = df.merge(
            loss_rates_df[['Item Code', 'Loss Rate (%)']],
            on='Item Code',
            how='left'
        )
        
        # Fill missing loss rates with category average
        if 'Category Code' in df.columns:
            avg_loss_by_category = df.groupby('Category Code')['Loss Rate (%)'].mean()
            df['Loss Rate (%)'] = df['Loss Rate (%)'].fillna(
                df['Category Code'].map(avg_loss_by_category)
            ).fillna(10.0)  # Default 10% if no category data
        else:
            df['Loss Rate (%)'].fillna(df['Loss Rate (%)'].median(), inplace=True)
        
        # Create loss-adjusted features
        df['Effective_Supply'] = df['Total_Quantity'] * (1 - df['Loss Rate (%)'] / 100)
        df['Loss_Adjusted_Revenue'] = df['Effective_Supply'] * df['Avg_Price']
        
        # Loss rate categories
        df['Loss_Rate_Category'] = pd.cut(
            df['Loss Rate (%)'],
            bins=[0, 5, 15, 25, 100],
            labels=['Low', 'Medium', 'High', 'Very_High']
        )
        
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        logger.info("Creating interaction features...")
        
        # Price-quantity interactions
        df['Price_Quantity_Interaction'] = df['Avg_Price'] * df['Total_Quantity']
        df['Price_Volatility_Quantity'] = df['Price_Volatility'] * df['Total_Quantity']
        
        # Seasonal-price interactions
        df['Spring_Price'] = df['Avg_Price'] * (df['Season'] == 'Spring').astype(int)
        df['Summer_Price'] = df['Avg_Price'] * (df['Season'] == 'Summer').astype(int)
        df['Autumn_Price'] = df['Avg_Price'] * (df['Season'] == 'Autumn').astype(int)
        df['Winter_Price'] = df['Avg_Price'] * (df['Season'] == 'Winter').astype(int)
        
        # Holiday-demand interactions
        df['Holiday_Demand'] = df['Total_Quantity'] * (
            df['IsNationalDay'] + df['IsLaborDay']
        )
        
        return df

    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for forecasting"""
        logger.info("Creating target variables...")
        
        # Future price targets (1, 7, 14, 30 days ahead)
        forecast_horizons = [1, 7, 14, 30]
        
        for horizon in forecast_horizons:
            target_col = f'Avg_Price_Target_{horizon}d'
            df[target_col] = df.groupby('Item Code')['Avg_Price'].shift(-horizon)
            
            # Price change targets
            df[f'Price_Change_Target_{horizon}d'] = (
                (df[target_col] - df['Avg_Price']) / df['Avg_Price'] * 100
            )
        
        # Quantity targets
        for horizon in forecast_horizons:
            target_col = f'Quantity_Target_{horizon}d'
            df[target_col] = df.groupby('Item Code')['Total_Quantity'].shift(-horizon)
        
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        logger.info("Handling missing values...")
        
        # Numeric columns - use forward fill then median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Forward fill within each item group
            df[col] = df.groupby('Item Code')[col].fillna(method='ffill')
            # Then backward fill
            df[col] = df.groupby('Item Code')[col].fillna(method='bfill')
            # Finally use median for remaining missing values
            df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in ['Date']:
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_value, inplace=True)
        
        return df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        logger.info("Encoding categorical features...")
        
        # One-hot encoding for low cardinality categoricals
        low_cardinality_cols = ['Season', 'Loss_Rate_Category']
        for col in low_cardinality_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
        
        # Label encoding for high cardinality categoricals
        from sklearn.preprocessing import LabelEncoder
        high_cardinality_cols = ['Category Name']
        for col in high_cardinality_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_Encoded'] = le.fit_transform(df[col].astype(str))
        
        return df

    def run_feature_engineering(self, data_path: str, output_path: str) -> Dict[str, str]:
        """Run complete feature engineering pipeline"""
        logger.info("Starting feature engineering...")
        
        try:
            # Load raw data
            data_dict = self.load_raw_data(data_path)
            
            if 'sales' not in data_dict:
                raise ValueError("Sales data not found")
            
            # Create daily aggregates
            daily_agg = self.create_daily_aggregates(data_dict['sales'])
            
            # Add temporal features
            daily_agg = self.add_temporal_features(daily_agg)
            
            # Add price features (if wholesale data available)
            if 'wholesale' in data_dict:
                daily_agg = self.add_price_features(daily_agg, data_dict['wholesale'])
            
            # Add category features (if item master available)
            if 'items' in data_dict:
                daily_agg = self.add_category_features(daily_agg, data_dict['items'])
            
            # Add loss rate features (if available)
            if 'loss_rates' in data_dict:
                daily_agg = self.add_loss_rate_features(daily_agg, data_dict['loss_rates'])
            
            # Add lag and rolling features
            daily_agg = self.add_lag_features(daily_agg)
            daily_agg = self.add_rolling_features(daily_agg)
            
            # Create interaction features
            daily_agg = self.create_interaction_features(daily_agg)
            
            # Create target variables
            daily_agg = self.create_target_variables(daily_agg)
            
            # Handle missing values
            daily_agg = self.handle_missing_values(daily_agg)
            
            # Encode categorical features
            daily_agg = self.encode_categorical_features(daily_agg)
            
            # Sort by date and item
            daily_agg = daily_agg.sort_values(['Item Code', 'Date']).reset_index(drop=True)
            
            # Create output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Train/validation/test split (70/15/15)
            total_records = len(daily_agg)
            train_end = int(total_records * 0.7)
            val_end = int(total_records * 0.85)
            
            train_df = daily_agg.iloc[:train_end].copy()
            val_df = daily_agg.iloc[train_end:val_end].copy()
            test_df = daily_agg.iloc[val_end:].copy()
            
            # Remove rows with missing targets
            target_cols = [col for col in daily_agg.columns if 'Target' in col]
            if target_cols:
                train_df = train_df.dropna(subset=[target_cols[0]])  # Use first target
                val_df = val_df.dropna(subset=[target_cols[0]])
                test_df = test_df.dropna(subset=[target_cols[0]])
            
            # Save splits
            output_files = {}
            
            train_path = os.path.join(output_path, 'train.parquet')
            train_df.to_parquet(train_path, index=False)
            output_files['train'] = train_path
            
            val_path = os.path.join(output_path, 'validation.parquet')
            val_df.to_parquet(val_path, index=False)
            output_files['validation'] = val_path
            
            test_path = os.path.join(output_path, 'test.parquet')
            test_df.to_parquet(test_path, index=False)
            output_files['test'] = test_path
            
            # Save complete feature set
            features_path = os.path.join(output_path, 'features_complete.parquet')
            daily_agg.to_parquet(features_path, index=False)
            output_files['features_complete'] = features_path
            
            # Save metadata
            metadata = {
                'total_features': len(daily_agg.columns),
                'feature_columns': list(daily_agg.columns),
                'numeric_features': list(daily_agg.select_dtypes(include=[np.number]).columns),
                'categorical_features': list(daily_agg.select_dtypes(include=['object']).columns),
                'train_records': len(train_df),
                'validation_records': len(val_df),
                'test_records': len(test_df),
                'date_range': {
                    'start': daily_agg['Date'].min().strftime('%Y-%m-%d'),
                    'end': daily_agg['Date'].max().strftime('%Y-%m-%d')
                },
                'items_count': daily_agg['Item Code'].nunique(),
                'total_records': len(daily_agg)
            }
            
            metadata_path = os.path.join(output_path, 'feature_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            output_files['metadata'] = metadata_path
            
            logger.info("Feature engineering completed successfully!")
            logger.info(f"Total features created: {len(daily_agg.columns)}")
            logger.info(f"Dataset shape: {daily_agg.shape}")
            logger.info(f"Train set: {len(train_df)} records")
            logger.info(f"Validation set: {len(val_df)} records")
            logger.info(f"Test set: {len(test_df)} records")
            
            return output_files
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Feature Engineering for Chinese Produce Forecasting')
    parser.add_argument('--config', required=True, help='Path to config.yaml')
    parser.add_argument('--data-path', required=True, help='Path to raw data directory')
    parser.add_argument('--output-path', required=True, help='Path for processed features output')
    
    args = parser.parse_args()
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(args.config)
    
    # Run feature engineering
    output_files = feature_engineer.run_feature_engineering(args.data_path, args.output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    for file_type, file_path in output_files.items():
        print(f"{file_type}: {file_path}")
    
    print("\nFeature engineering completed successfully!")


if __name__ == "__main__":
    main()