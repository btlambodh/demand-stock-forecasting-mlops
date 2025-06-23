#!/usr/bin/env python3
"""
Athena Integration Test Script
Validates the complete pipeline and tests Athena connectivity

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import os
import json
import boto3
import yaml
import pandas as pd
import time
from typing import Dict, List


def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_s3_data_availability(config: Dict) -> Dict:
    """Test if processed data is available in S3"""
    print(" Testing S3 data availability...")
    
    s3_client = boto3.client('s3', region_name=config['aws']['region'])
    bucket_name = config['aws']['s3']['bucket_name']
    data_prefix = f"{config['aws']['s3']['data_prefix']}processed/"
    
    results = {}
    expected_files = [
        'features_complete.parquet',
        'train.parquet', 
        'validation.parquet',
        'test.parquet',
        'feature_metadata.json'
    ]
    
    for filename in expected_files:
        s3_key = f"{data_prefix}{filename}"
        try:
            response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            file_size = response['ContentLength']
            results[filename] = {
                'exists': True,
                'size_mb': round(file_size / (1024*1024), 2),
                's3_uri': f"s3://{bucket_name}/{s3_key}"
            }
            print(f"  ✓ {filename}: {results[filename]['size_mb']} MB")
        except Exception as e:
            results[filename] = {'exists': False, 'error': str(e)}
            print(f"  ✗ {filename}: Not found")
    
    return results


def test_feature_metadata_validity(config: Dict) -> Dict:
    """Test if feature metadata is valid for Athena integration"""
    print(" Testing feature metadata validity...")
    
    try:
        # Check local metadata first
        local_metadata_path = 'data/processed/feature_metadata.json'
        if os.path.exists(local_metadata_path):
            with open(local_metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"  ✓ Found local metadata with {metadata['total_features']} features")
            print(f"  ✓ Data range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
            print(f"  ✓ Total records: {metadata['total_records']:,}")
            print(f"  ✓ Train/Val/Test split: {metadata['train_records']}/{metadata['validation_records']}/{metadata['test_records']}")
            
            # Check for problematic column names
            problematic_columns = []
            for col in metadata['feature_columns']:
                if any(char in col for char in ['(', ')', '%', '/', ' ']):
                    problematic_columns.append(col)
            
            if problematic_columns:
                print(f"    Found {len(problematic_columns)} columns with special characters (will be escaped)")
            else:
                print("  ✓ All column names are Athena-friendly")
            
            return {
                'valid': True,
                'metadata': metadata,
                'problematic_columns': problematic_columns
            }
        else:
            print("  ✗ Local metadata file not found")
            return {'valid': False, 'error': 'Metadata file not found'}
    
    except Exception as e:
        print(f"  ✗ Error reading metadata: {e}")
        return {'valid': False, 'error': str(e)}


def test_athena_connectivity(config: Dict) -> Dict:
    """Test Athena connectivity and permissions"""
    print("🔍 Testing Athena connectivity...")
    
    try:
        athena_client = boto3.client('athena', region_name=config['aws']['region'])
        
        # Test basic connectivity
        workgroups = athena_client.list_work_groups()
        print(f"  ✓ Connected to Athena ({len(workgroups['WorkGroups'])} workgroups available)")
        
        # Test query execution permissions
        test_query = "SELECT 1 as test_value"
        response = athena_client.start_query_execution(
            QueryString=test_query,
            ResultConfiguration={
                'OutputLocation': config['aws']['athena']['query_results_location']
            },
            WorkGroup=config['aws']['athena'].get('workgroup', 'primary')
        )
        
        query_id = response['QueryExecutionId']
        print(f"  ✓ Query execution permissions confirmed (Query ID: {query_id[:8]}...)")
        
        return {'connected': True, 'test_query_id': query_id}
    
    except Exception as e:
        print(f"  ✗ Athena connection failed: {e}")
        return {'connected': False, 'error': str(e)}


def test_database_and_tables(config: Dict) -> Dict:
    """Test if database and tables exist"""
    print("🔍 Testing database and tables...")
    
    try:
        athena_client = boto3.client('athena', region_name=config['aws']['region'])
        database_name = config['aws']['athena']['database_name']
        
        # Test database exists
        glue_client = boto3.client('glue', region_name=config['aws']['region'])
        try:
            glue_client.get_database(Name=database_name)
            print(f"  ✓ Database exists: {database_name}")
            database_exists = True
        except Exception:
            print(f"  ✗ Database not found: {database_name}")
            database_exists = False
        
        # Test tables if database exists
        tables_status = {}
        if database_exists:
            tables_query = f"SHOW TABLES IN {database_name}"
            response = athena_client.start_query_execution(
                QueryString=tables_query,
                ResultConfiguration={
                    'OutputLocation': config['aws']['athena']['query_results_location']
                },
                WorkGroup=config['aws']['athena'].get('workgroup', 'primary')
            )
            
            query_id = response['QueryExecutionId']
            print(f"  ✓ Tables query submitted (Query ID: {query_id[:8]}...)")
            
            # Wait a bit and check status
            time.sleep(3)
            execution = athena_client.get_query_execution(QueryExecutionId=query_id)
            status = execution['QueryExecution']['Status']['State']
            
            if status == 'SUCCEEDED':
                # Get results
                results = athena_client.get_query_results(QueryExecutionId=query_id)
                table_names = []
                for row in results['ResultSet']['Rows'][1:]:  # Skip header
                    if row['Data']:
                        table_names.append(row['Data'][0]['VarCharValue'])
                
                print(f"  ✓ Found {len(table_names)} tables: {', '.join(table_names)}")
                tables_status = {table: True for table in table_names}
            else:
                print(f"    Tables query status: {status}")
        
        return {
            'database_exists': database_exists,
            'tables_status': tables_status,
            'query_id': query_id if database_exists else None
        }
    
    except Exception as e:
        print(f"  ✗ Database/tables test failed: {e}")
        return {'database_exists': False, 'error': str(e)}


def run_sample_data_query(config: Dict) -> Dict:
    """Run a sample data query to test table functionality"""
    print(" Running sample data query...")
    
    try:
        athena_client = boto3.client('athena', region_name=config['aws']['region'])
        database_name = config['aws']['athena']['database_name']
        
        # Simple count query
        query = f"""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT `Item Code`) as unique_items,
            MIN(`Date`) as earliest_date,
            MAX(`Date`) as latest_date
        FROM {database_name}.features_complete
        LIMIT 1
        """
        
        response = athena_client.start_query_execution(
            QueryString=query,
            ResultConfiguration={
                'OutputLocation': config['aws']['athena']['query_results_location']
            },
            WorkGroup=config['aws']['athena'].get('workgroup', 'primary')
        )
        
        query_id = response['QueryExecutionId']
        print(f"  ✓ Sample query submitted (Query ID: {query_id[:8]}...)")
        
        # Wait for completion
        max_wait = 30
        wait_time = 0
        while wait_time < max_wait:
            execution = athena_client.get_query_execution(QueryExecutionId=query_id)
            status = execution['QueryExecution']['Status']['State']
            
            if status == 'SUCCEEDED':
                # Get results
                results = athena_client.get_query_results(QueryExecutionId=query_id)
                if len(results['ResultSet']['Rows']) > 1:
                    data_row = results['ResultSet']['Rows'][1]['Data']
                    record_count = data_row[0]['VarCharValue']
                    unique_items = data_row[1]['VarCharValue']
                    earliest_date = data_row[2]['VarCharValue']
                    latest_date = data_row[3]['VarCharValue']
                    
                    print(f"  ✓ Query successful!")
                    print(f"    • Total records: {record_count}")
                    print(f"    • Unique items: {unique_items}")
                    print(f"    • Date range: {earliest_date} to {latest_date}")
                    
                    return {
                        'success': True,
                        'results': {
                            'total_records': record_count,
                            'unique_items': unique_items,
                            'date_range': f"{earliest_date} to {latest_date}"
                        }
                    }
                else:
                    print("    Query returned no data")
                    return {'success': False, 'error': 'No data returned'}
                    
            elif status in ['FAILED', 'CANCELLED']:
                error_reason = execution['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                print(f"  ✗ Query failed: {error_reason}")
                return {'success': False, 'error': error_reason}
            
            time.sleep(2)
            wait_time += 2
        
        print(f"    Query timeout after {max_wait} seconds")
        return {'success': False, 'error': 'Query timeout'}
    
    except Exception as e:
        print(f"  ✗ Sample query failed: {e}")
        return {'success': False, 'error': str(e)}


def main():
    """Main test function"""
    print(" Athena Integration Test Suite")
    print("=" * 50)
    
    # Load configuration
    try:
        config = load_config()
        print(f"✓ Configuration loaded for project: {config['project']['name']}")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return
    
    print()
    
    # Run tests
    tests = [
        ("S3 Data Availability", lambda: test_s3_data_availability(config)),
        ("Feature Metadata Validity", lambda: test_feature_metadata_validity(config)),
        ("Athena Connectivity", lambda: test_athena_connectivity(config)),
        ("Database and Tables", lambda: test_database_and_tables(config)),
        ("Sample Data Query", lambda: run_sample_data_query(config))
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            results[test_name] = {'error': str(e)}
            print(f"  ✗ Test failed with exception: {e}")
        print()
    
    # Summary
    print(" Test Summary")
    print("=" * 50)
    
    total_tests = len(tests)
    passed_tests = 0
    
    for test_name, result in results.items():
        if isinstance(result, dict):
            if result.get('success', result.get('connected', result.get('valid', result.get('database_exists', False)))):
                print(f"✓ {test_name}: PASSED")
                passed_tests += 1
            else:
                print(f"✗ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
        else:
            print(f"  {test_name}: UNCLEAR")
    
    print()
    print(f" Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print(" All tests passed! Your Athena integration is ready!")
        print()
        print(" Next steps:")
        print("  1. Open AWS Athena Console")
        print("  2. Select your database: demand_stock_forecasting_mlops_feature_store")
        print("  3. Try queries like: SELECT * FROM features_complete LIMIT 10")
    else:
        print("  Some tests failed. Please check the errors above.")
        print()
        print(" Common fixes:")
        print("  • Run: make pipeline-data-enhanced")
        print("  • Check AWS permissions")
        print("  • Verify S3 bucket access")


if __name__ == "__main__":
    main()