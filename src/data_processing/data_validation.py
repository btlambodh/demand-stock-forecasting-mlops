#!/usr/bin/env python3
"""
Data Validation Script for Demand Stock Forecasting MLOps
Validates raw CSV files and generates data quality reports

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_validation')


class DataValidator:
    """Comprehensive data validation for demand stock forecasting market data"""
    
    def __init__(self, config_path: str):
        """Initialize validator with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.validation_results = {}
        
        # Expected schemas for each file
        self.expected_schemas = {
            'annex1': {
                'columns': ['Item Code', 'Item Name', 'Category Code', 'Category Name'],
                'dtypes': {
                    'Item Code': 'int64',
                    'Item Name': 'object',
                    'Category Code': 'int64', 
                    'Category Name': 'object'
                }
            },
            'annex2': {
                'columns': ['Date', 'Time', 'Item Code', 'Quantity Sold (kilo)', 
                           'Unit Selling Price (RMB/kg)', 'Sale or Return', 'Discount (Yes/No)'],
                'dtypes': {
                    'Date': 'object',
                    'Time': 'object',
                    'Item Code': 'int64',
                    'Quantity Sold (kilo)': 'float64',
                    'Unit Selling Price (RMB/kg)': 'float64',
                    'Sale or Return': 'object',
                    'Discount (Yes/No)': 'object'
                }
            },
            'annex3': {
                'columns': ['Date', 'Item Code', 'Wholesale Price (RMB/kg)'],
                'dtypes': {
                    'Date': 'object',
                    'Item Code': 'int64',
                    'Wholesale Price (RMB/kg)': 'float64'
                }
            },
            'annex4': {
                'columns': ['Item Code', 'Item Name', 'Loss Rate (%)'],
                'dtypes': {
                    'Item Code': 'int64',
                    'Item Name': 'object',
                    'Loss Rate (%)': 'float64'
                }
            }
        }

    def validate_file_existence(self, data_path: str) -> Dict[str, bool]:
        """Check if all required files exist"""
        logger.info("Validating file existence...")
        
        file_status = {}
        
        # Look for the actual annex files
        expected_files = ['annex1.csv', 'annex2.csv', 'annex3.csv', 'annex4.csv']
        file_mapping = {
            'annex1.csv': 'item_master',
            'annex2.csv': 'sales_transactions', 
            'annex3.csv': 'wholesale_prices',
            'annex4.csv': 'loss_rates'
        }
        
        for filename in expected_files:
            full_path = os.path.join(data_path, filename)
            exists = os.path.exists(full_path)
            file_key = file_mapping[filename]
            file_status[file_key] = exists
            
            if exists:
                logger.info(f"✓ {file_key}: {full_path}")
            else:
                logger.error(f"✗ {file_key}: {full_path} - FILE NOT FOUND")
        
        return file_status

    def validate_schema(self, df: pd.DataFrame, expected_schema: Dict, file_name: str) -> Dict:
        """Validate DataFrame schema against expected structure"""
        logger.info(f"Validating schema for {file_name}...")
        
        validation_result = {
            'file_name': file_name,
            'columns_match': False,
            'dtypes_valid': False,
            'missing_columns': [],
            'extra_columns': [],
            'dtype_issues': []
        }
        
        # Check columns
        expected_cols = set(expected_schema['columns'])
        actual_cols = set(df.columns)
        
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols
        
        validation_result['missing_columns'] = list(missing_cols)
        validation_result['extra_columns'] = list(extra_cols)
        validation_result['columns_match'] = len(missing_cols) == 0 and len(extra_cols) == 0
        
        # Check data types (for columns that exist)
        dtype_issues = []
        for col, expected_dtype in expected_schema['dtypes'].items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if not self._dtype_compatible(actual_dtype, expected_dtype):
                    dtype_issues.append({
                        'column': col,
                        'expected': expected_dtype,
                        'actual': actual_dtype
                    })
        
        validation_result['dtype_issues'] = dtype_issues
        validation_result['dtypes_valid'] = len(dtype_issues) == 0
        
        return validation_result

    def _dtype_compatible(self, actual: str, expected: str) -> bool:
        """Check if data types are compatible"""
        # Handle common type variations
        type_mapping = {
            'int64': ['int64', 'int32', 'int16', 'int8'],
            'float64': ['float64', 'float32'],
            'object': ['object', 'string']
        }
        
        if expected in type_mapping:
            return actual in type_mapping[expected]
        return actual == expected

    def validate_data_quality(self, df: pd.DataFrame, file_name: str) -> Dict:
        """Comprehensive data quality validation"""
        logger.info(f"Validating data quality for {file_name}...")
        
        quality_metrics = {
            'file_name': file_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data': {},
            'duplicate_rows': 0,
            'outliers': {},
            'date_range': {},
            'negative_values': {},
            'quality_score': 0.0
        }
        
        # Missing data analysis
        missing_data = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            missing_data[col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2)
            }
        quality_metrics['missing_data'] = missing_data
        
        # Duplicate rows
        quality_metrics['duplicate_rows'] = int(df.duplicated().sum())
        
        # Date range analysis (if date columns exist)
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        if date_columns:
            for date_col in date_columns:
                try:
                    df_temp = df[df[date_col].notna()].copy()
                    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
                    quality_metrics['date_range'][date_col] = {
                        'min_date': df_temp[date_col].min().strftime('%Y-%m-%d'),
                        'max_date': df_temp[date_col].max().strftime('%Y-%m-%d'),
                        'date_span_days': (df_temp[date_col].max() - df_temp[date_col].min()).days
                    }
                except Exception as e:
                    logger.warning(f"Could not parse dates in {date_col}: {e}")
        
        # Outlier detection for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        for col in numeric_columns:
            if df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outliers[col] = {
                    'count': int(outlier_count),
                    'percentage': round((outlier_count / len(df)) * 100, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2)
                }
        quality_metrics['outliers'] = outliers
        
        # Negative values check for price/quantity columns
        price_quantity_cols = [col for col in df.columns 
                              if any(term in col.lower() for term in ['price', 'quantity', 'rate'])]
        negative_values = {}
        for col in price_quantity_cols:
            if col in numeric_columns:
                negative_count = (df[col] < 0).sum()
                negative_values[col] = {
                    'count': int(negative_count),
                    'percentage': round((negative_count / len(df)) * 100, 2)
                }
        quality_metrics['negative_values'] = negative_values
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(quality_metrics)
        quality_metrics['quality_score'] = quality_score
        
        return quality_metrics

    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100.0
        
        # Deduct for missing data
        avg_missing_pct = np.mean([v['percentage'] for v in metrics['missing_data'].values()])
        score -= avg_missing_pct * 2  # Penalty factor
        
        # Deduct for duplicates
        duplicate_pct = (metrics['duplicate_rows'] / metrics['total_rows']) * 100
        score -= duplicate_pct * 5
        
        # Deduct for excessive outliers
        avg_outlier_pct = np.mean([v['percentage'] for v in metrics['outliers'].values()] + [0])
        if avg_outlier_pct > 5:  # More than 5% outliers
            score -= (avg_outlier_pct - 5) * 0.5
        
        # Deduct for negative values in price/quantity columns
        avg_negative_pct = np.mean([v['percentage'] for v in metrics['negative_values'].values()] + [0])
        score -= avg_negative_pct * 10
        
        return max(0.0, min(100.0, round(score, 2)))

    def validate_business_rules(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Validate business-specific rules"""
        logger.info("Validating business rules...")
        
        business_validation = {
            'rules_passed': 0,
            'rules_failed': 0,
            'rule_results': []
        }
        
        try:
            # Rule 1: All items in sales data should exist in item master
            if 'annex1' in data_dict and 'annex2' in data_dict:
                master_items = set(data_dict['annex1']['Item Code'].unique())
                sales_items = set(data_dict['annex2']['Item Code'].unique())
                orphan_items = sales_items - master_items
                
                rule_result = {
                    'rule': 'Sales items exist in master data',
                    'passed': len(orphan_items) == 0,
                    'details': f"Found {len(orphan_items)} orphan items in sales data"
                }
                business_validation['rule_results'].append(rule_result)
                
                if rule_result['passed']:
                    business_validation['rules_passed'] += 1
                else:
                    business_validation['rules_failed'] += 1
            
            # Rule 2: Wholesale prices should be reasonable
            if 'annex3' in data_dict:
                wholesale_df = data_dict['annex3']
                zero_prices = (wholesale_df['Wholesale Price (RMB/kg)'] <= 0).sum()
                high_prices = (wholesale_df['Wholesale Price (RMB/kg)'] > 1000).sum()
                
                rule_result = {
                    'rule': 'Wholesale prices are reasonable',
                    'passed': zero_prices == 0 and high_prices < len(wholesale_df) * 0.01,
                    'details': f"Zero/negative prices: {zero_prices}, Extremely high prices: {high_prices}"
                }
                business_validation['rule_results'].append(rule_result)
                
                if rule_result['passed']:
                    business_validation['rules_passed'] += 1
                else:
                    business_validation['rules_failed'] += 1
            
            # Rule 3: Loss rates should be between 0% and 100%
            if 'annex4' in data_dict:
                loss_df = data_dict['annex4']
                invalid_rates = ((loss_df['Loss Rate (%)'] < 0) | 
                               (loss_df['Loss Rate (%)'] > 100)).sum()
                
                rule_result = {
                    'rule': 'Loss rates are valid percentages (0-100%)',
                    'passed': invalid_rates == 0,
                    'details': f"Invalid loss rates: {invalid_rates}"
                }
                business_validation['rule_results'].append(rule_result)
                
                if rule_result['passed']:
                    business_validation['rules_passed'] += 1
                else:
                    business_validation['rules_failed'] += 1
            
            # Rule 4: Sales quantities should be positive for 'sale' transactions
            if 'annex2' in data_dict:
                sales_df = data_dict['annex2']
                sale_records = sales_df[sales_df['Sale or Return'].str.lower() == 'sale']
                negative_quantities = (sale_records['Quantity Sold (kilo)'] <= 0).sum()
                
                rule_result = {
                    'rule': 'Sale quantities are positive',
                    'passed': negative_quantities == 0,
                    'details': f"Negative/zero quantities in sales: {negative_quantities}"
                }
                business_validation['rule_results'].append(rule_result)
                
                if rule_result['passed']:
                    business_validation['rules_passed'] += 1
                else:
                    business_validation['rules_failed'] += 1
        
        except Exception as e:
            logger.error(f"Error in business rule validation: {e}")
        
        return business_validation

    def run_validation(self, data_path: str, output_path: str) -> Dict:
        """Run complete data validation pipeline"""
        logger.info("Starting comprehensive data validation...")
        
        validation_summary = {
            'timestamp': datetime.now().isoformat(),
            'data_path': data_path,
            'files_validated': 0,
            'total_quality_score': 0.0,
            'validation_passed': False,
            'recommendations': []
        }
        
        try:
            # 1. File existence validation
            file_status = self.validate_file_existence(data_path)
            self.validation_results['file_existence'] = file_status
            
            missing_files = [k for k, v in file_status.items() if not v]
            if missing_files:
                logger.error(f"Missing files: {missing_files}")
                validation_summary['recommendations'].append(
                    f"Upload missing files: {missing_files}"
                )
                return validation_summary
            
            # 2. Load and validate each file
            data_dict = {}
            quality_scores = []
            
            file_mapping = {
                'item_master': 'annex1.csv',
                'sales_transactions': 'annex2.csv',
                'wholesale_prices': 'annex3.csv',
                'loss_rates': 'annex4.csv'
            }
            
            annex_mapping = {
                'annex1.csv': 'annex1',
                'annex2.csv': 'annex2', 
                'annex3.csv': 'annex3',
                'annex4.csv': 'annex4'
            }
            
            for file_key, file_exists in file_status.items():
                if file_exists:
                    filename = file_mapping[file_key]
                    file_path = os.path.join(data_path, filename)
                    annex_key = annex_mapping[filename]
                    
                    try:
                        # Load data
                        df = pd.read_csv(file_path)
                        data_dict[annex_key] = df
                        validation_summary['files_validated'] += 1
                        
                        logger.info(f"Loaded {file_key}: {len(df)} rows, {len(df.columns)} columns")
                        
                        # Schema validation
                        if annex_key in self.expected_schemas:
                            schema_result = self.validate_schema(
                                df, self.expected_schemas[annex_key], file_key
                            )
                            self.validation_results[f'{file_key}_schema'] = schema_result
                        
                        # Data quality validation
                        quality_result = self.validate_data_quality(df, file_key)
                        self.validation_results[f'{file_key}_quality'] = quality_result
                        quality_scores.append(quality_result['quality_score'])
                        
                    except Exception as e:
                        logger.error(f"Error processing {file_key}: {e}")
                        self.validation_results[f'{file_key}_error'] = str(e)
            
            # 3. Business rules validation
            if data_dict:
                business_result = self.validate_business_rules(data_dict)
                self.validation_results['business_rules'] = business_result
            
            # 4. Calculate overall metrics
            if quality_scores:
                validation_summary['total_quality_score'] = round(np.mean(quality_scores), 2)
            
            # 5. Determine validation status
            validation_summary['validation_passed'] = (
                validation_summary['total_quality_score'] >= 70.0 and  # Quality threshold
                all(file_status.values())  # All files present
            )
            
            # 6. Generate recommendations
            if validation_summary['total_quality_score'] < 80:
                validation_summary['recommendations'].append(
                    "Consider data cleaning to improve quality score"
                )
            
            # 7. Save results
            os.makedirs(output_path, exist_ok=True)
            
            import json
            report_path = os.path.join(output_path, 'validation_summary.json')
            with open(report_path, 'w') as f:
                json.dump(validation_summary, f, indent=2)
            
            logger.info(f"Validation report saved: {report_path}")
            logger.info("Data validation completed successfully!")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_summary['error'] = str(e)
        
        return validation_summary


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Validate data for Demand Stock Forecasting MLOps')
    parser.add_argument('--config', required=True, help='Path to config.yaml')
    parser.add_argument('--data-path', required=True, help='Path to raw data directory')
    parser.add_argument('--output-path', required=True, help='Path for validation output')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = DataValidator(args.config)
    
    # Run validation
    results = validator.run_validation(args.data_path, args.output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("DATA VALIDATION SUMMARY")
    print("="*60)
    print(f"Files Validated: {results['files_validated']}")
    print(f"Overall Quality Score: {results['total_quality_score']}/100")
    print(f"Validation Status: {'PASSED' if results['validation_passed'] else 'FAILED'}")
    
    if results.get('recommendations'):
        print("\nRecommendations:")
        for rec in results['recommendations']:
            print(f"- {rec}")
    
    # Set exit code based on validation result
    sys.exit(0 if results['validation_passed'] else 1)


if __name__ == "__main__":
    main()