#!/usr/bin/env python3
"""
Feature Store Integration for Demand Stock Forecasting MLOps
Fixes - Handles S3 reorganization and proper Athena table creation

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import argparse
import logging
import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import yaml
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature_store_integration')


class FeatureStoreManager:
    """Fixed Feature Store manager with proper Athena integration"""
    
    def __init__(self, config_path: str):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # AWS clients
        session = boto3.Session()
        self.sagemaker_client = session.client('sagemaker', region_name=self.config['aws']['region'])
        self.featurestore_runtime_client = session.client('sagemaker-featurestore-runtime', 
                                                         region_name=self.config['aws']['region'])
        self.s3_client = session.client('s3', region_name=self.config['aws']['region'])
        self.glue_client = session.client('glue', region_name=self.config['aws']['region'])
        self.athena_client = session.client('athena', region_name=self.config['aws']['region'])
        
        # Configuration
        self.project_name = self.config['project']['name']
        self.bucket_name = self.config['aws']['s3']['bucket_name']
        self.feature_group_prefix = f"{self.project_name}-features"
        self.database_name = f"{self.project_name.replace('-', '_')}_feature_store"

    def clean_column_name(self, name):
        """Clean column name for Feature Store"""
        import re
        
        clean_name = str(name)
        clean_name = clean_name.replace(' ', '_')
        clean_name = clean_name.replace('(', '')
        clean_name = clean_name.replace(')', '')
        clean_name = clean_name.replace('/', '_')
        clean_name = clean_name.replace('%', '_pct')
        clean_name = clean_name.replace('$', '_dollar')
        clean_name = clean_name.replace('#', '_num')
        clean_name = clean_name.replace('@', '_at')
        clean_name = clean_name.replace('&', '_and')
        clean_name = clean_name.replace('+', '_plus')
        clean_name = clean_name.replace('=', '_eq')
        clean_name = clean_name.replace('.', '_')
        clean_name = clean_name.replace(',', '_')
        clean_name = clean_name.replace(':', '_')
        clean_name = clean_name.replace(';', '_')
        
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', clean_name)
        clean_name = re.sub(r'_+', '_', clean_name)
        clean_name = clean_name.strip('_')
        
        if not clean_name:
            clean_name = 'unnamed_feature'
        
        if clean_name and not clean_name[0].isalnum():
            clean_name = 'feature_' + clean_name
        
        if len(clean_name) > 64:
            clean_name = clean_name[:64].rstrip('_')
        
        return clean_name

    def map_pandas_to_athena_type(self, column_name: str, sample_value) -> str:
        """Map pandas dtype to Athena data type using sample value"""
        
        if column_name.lower() == 'date':
            return 'timestamp'
        elif 'time' in column_name.lower():
            return 'timestamp'
        elif column_name in ['Item Code', 'Category Code']:
            return 'bigint'
        elif 'code' in column_name.lower():
            return 'bigint'
        
        if pd.isna(sample_value):
            return 'string'
        elif isinstance(sample_value, bool):
            return 'boolean'
        elif isinstance(sample_value, int):
            return 'bigint'
        elif isinstance(sample_value, float):
            return 'double'
        elif isinstance(sample_value, str):
            return 'string'
        else:
            return 'string'

    def copy_file_to_table_directory(self, source_key: str, dest_dir: str) -> str:
        """Copy a file to its own table directory"""
        
        filename = source_key.split('/')[-1]
        dest_key = f"{dest_dir}{filename}"
        
        copy_source = {'Bucket': self.bucket_name, 'Key': source_key}
        
        try:
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket_name,
                Key=dest_key
            )
            return f"s3://{self.bucket_name}/{dest_dir}"
        except Exception as e:
            logger.error(f"Error copying {source_key} to {dest_key}: {e}")
            return None

    def reorganize_s3_for_athena(self, s3_upload_results: Dict) -> Dict:
        """Reorganize S3 files into table-specific directories for Athena"""
        logger.info("Reorganizing S3 structure for Athena compatibility...")
        
        data_prefix = self.config['aws']['s3']['data_prefix']
        
        # Define file mappings to individual directories
        file_mappings = [
            {
                'source_key': f"{data_prefix}processed/features_complete.parquet",
                'dest_dir': f"{data_prefix}athena-tables/features_complete/",
                'table_name': 'features_complete'
            },
            {
                'source_key': f"{data_prefix}processed/train.parquet", 
                'dest_dir': f"{data_prefix}athena-tables/train_data/",
                'table_name': 'train_data'
            },
            {
                'source_key': f"{data_prefix}processed/validation.parquet",
                'dest_dir': f"{data_prefix}athena-tables/validation_data/", 
                'table_name': 'validation_data'
            },
            {
                'source_key': f"{data_prefix}processed/test.parquet",
                'dest_dir': f"{data_prefix}athena-tables/test_data/",
                'table_name': 'test_data'
            }
        ]
        
        table_locations = {}
        
        for mapping in file_mappings:
            logger.info(f"  Copying {mapping['source_key']} to {mapping['dest_dir']}...")
            
            s3_location = self.copy_file_to_table_directory(
                mapping['source_key'], 
                mapping['dest_dir']
            )
            
            if s3_location:
                table_locations[mapping['table_name']] = s3_location
                logger.info(f"  ✓ {mapping['table_name']} → {s3_location}")
            else:
                logger.error(f"  ✗ Failed to copy {mapping['table_name']}")
        
        return table_locations

    def create_athena_table_from_metadata(self, metadata_path: str, table_name: str, 
                                        s3_location: str) -> str:
        """Create Athena table using feature metadata - FIXED VERSION"""
        logger.info(f"Creating Athena table: {table_name}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load sample data to infer types - FIXED: Remove nrows parameter
        features_file = os.path.join(os.path.dirname(metadata_path), 'features_complete.parquet')
        if os.path.exists(features_file):
            full_df = pd.read_parquet(features_file)
            sample_df = full_df.head(100)  # Get first 100 rows manually
        else:
            raise FileNotFoundError(f"Cannot find features_complete.parquet at {features_file}")
        
        # Build column definitions
        column_definitions = []
        for col_name in metadata['feature_columns']:
            if col_name in sample_df.columns:
                # Get sample value for type inference
                non_null_values = sample_df[col_name].dropna()
                sample_value = non_null_values.iloc[0] if len(non_null_values) > 0 else None
                
                athena_type = self.map_pandas_to_athena_type(col_name, sample_value)
                
                # Escape column names with special characters
                escaped_name = f"`{col_name}`" if any(c in col_name for c in [' ', '(', ')', '%', '/', '-']) else col_name
                column_definitions.append(f"  {escaped_name} {athena_type}")
        
        columns_ddl = ",\n".join(column_definitions)
        
        # Ensure s3_location ends with /
        if not s3_location.endswith('/'):
            s3_location += '/'
        
        # Create table DDL
        create_table_sql = f"""
CREATE EXTERNAL TABLE {self.database_name}.{table_name} (
{columns_ddl}
)
STORED AS PARQUET
LOCATION '{s3_location}'
TBLPROPERTIES (
  'has_encrypted_data'='false'
)"""
        
        # Execute DDL
        try:
            response = self.athena_client.start_query_execution(
                QueryString=create_table_sql,
                ResultConfiguration={
                    'OutputLocation': f"s3://{self.bucket_name}/athena-results/"
                },
                WorkGroup='primary'
            )
            
            query_execution_id = response['QueryExecutionId']
            logger.info(f"Table creation query submitted: {query_execution_id}")
            
            # Wait for completion
            self.wait_for_query_completion(query_execution_id)
            logger.info(f"Successfully created table: {table_name}")
            
            return create_table_sql
            
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {e}")
            raise

    def wait_for_query_completion(self, query_execution_id: str, max_wait_time: int = 300):
        """Wait for Athena query to complete"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            response = self.athena_client.get_query_execution(
                QueryExecutionId=query_execution_id
            )
            
            status = response['QueryExecution']['Status']['State']
            
            if status == 'SUCCEEDED':
                logger.info(f"Query {query_execution_id} completed successfully")
                return
            elif status in ['FAILED', 'CANCELLED']:
                error_msg = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                raise Exception(f"Query {query_execution_id} failed: {error_msg}")
            
            time.sleep(5)
        
        raise Exception(f"Query {query_execution_id} timed out after {max_wait_time} seconds")

    def drop_existing_tables(self):
        """Drop existing tables to recreate them properly"""
        tables = ['features_complete', 'train_data', 'validation_data', 'test_data']
        
        for table in tables:
            try:
                drop_sql = f"DROP TABLE IF EXISTS {self.database_name}.{table}"
                response = self.athena_client.start_query_execution(
                    QueryString=drop_sql,
                    ResultConfiguration={
                        'OutputLocation': f"s3://{self.bucket_name}/athena-results/"
                    },
                    WorkGroup='primary'
                )
                self.wait_for_query_completion(response['QueryExecutionId'])
                logger.info(f"✓ Dropped table: {table}")
            except Exception as e:
                logger.warning(f"Could not drop {table}: {e}")

    def create_feature_definitions(self, df):
        """Create feature definitions"""
        logger.info("Creating feature definitions...")
        
        feature_definitions = []
        
        for column in df.columns:
            if column in ['Date', 'Item Code']:
                continue
                
            dtype = str(df[column].dtype)
            clean_name = self.clean_column_name(column)
            
            if dtype.startswith('int') or dtype.startswith('uint'):
                feature_type = 'Integral'
            elif dtype.startswith('float'):
                feature_type = 'Fractional'
            elif dtype == 'bool':
                feature_type = 'Integral'
            else:
                feature_type = 'String'
            
            feature_definitions.append({
                'FeatureName': clean_name,
                'FeatureType': feature_type
            })
        
        logger.info(f"Created {len(feature_definitions)} feature definitions")
        return feature_definitions

    def upload_data_to_s3(self, local_data_path):
        """Upload data to S3"""
        logger.info("Uploading processed data to S3...")
        
        upload_results = {}
        files_to_upload = [
            'train.parquet',
            'validation.parquet', 
            'test.parquet',
            'features_complete.parquet',
            'feature_metadata.json'
        ]
        
        for filename in files_to_upload:
            local_file_path = os.path.join(local_data_path, filename)
            
            if os.path.exists(local_file_path):
                s3_key = f"{self.config['aws']['s3']['data_prefix']}processed/{filename}"
                
                try:
                    self.s3_client.upload_file(local_file_path, self.bucket_name, s3_key)
                    s3_uri = f"s3://{self.bucket_name}/{s3_key}"
                    upload_results[filename] = s3_uri
                    logger.info(f"Uploaded {filename} to {s3_uri}")
                    
                except Exception as e:
                    logger.error(f"Error uploading {filename}: {e}")
            else:
                logger.warning(f"File not found: {local_file_path}")
        
        return upload_results

    def create_glue_database(self) -> bool:
        """Create Glue database for Athena queries"""
        logger.info(f"Creating Glue database: {self.database_name}")
        
        try:
            self.glue_client.create_database(
                DatabaseInput={
                    'Name': self.database_name,
                    'Description': f'Feature Store database for {self.project_name}',
                    'Parameters': {
                        'project': self.project_name,
                        'created_by': 'mlops-pipeline'
                    }
                }
            )
            logger.info(f"Database {self.database_name} created successfully")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'AlreadyExistsException':
                logger.info(f"Database {self.database_name} already exists")
                return True
            else:
                logger.error(f"Error creating database: {e}")
                return False

    def create_feature_group(self, feature_group_name, feature_definitions, description):
        """Create Feature Group"""
        logger.info(f"Creating feature group: {feature_group_name}")
        
        try:
            # Create Glue database first
            self.create_glue_database()
            
            # Check if exists
            try:
                response = self.sagemaker_client.describe_feature_group(
                    FeatureGroupName=feature_group_name
                )
                logger.info(f"Feature group {feature_group_name} already exists")
                return response['FeatureGroupArn']
            except ClientError as e:
                if e.response['Error']['Code'] != 'ResourceNotFound':
                    raise
            
            # Get the execution role from config
            execution_role = self.config['aws']['sagemaker']['execution_role']
            logger.info(f"Using execution role: {execution_role}")
            
            # Create new feature group with custom database
            create_response = self.sagemaker_client.create_feature_group(
                FeatureGroupName=feature_group_name,
                RecordIdentifierFeatureName='Item_Code',
                EventTimeFeatureName='EventTime',
                FeatureDefinitions=feature_definitions + [
                    {'FeatureName': 'Item_Code', 'FeatureType': 'Integral'},
                    {'FeatureName': 'EventTime', 'FeatureType': 'Fractional'}
                ],
                OnlineStoreConfig={'EnableOnlineStore': True},
                OfflineStoreConfig={
                    'S3StorageConfig': {
                        'S3Uri': f"s3://{self.bucket_name}/feature-store/{feature_group_name}/"
                    },
                    'DataCatalogConfig': {
                        'TableName': feature_group_name.replace('-', '_'),
                        'Catalog': 'AwsDataCatalog',
                        'Database': self.database_name
                    }
                },
                RoleArn=execution_role,
                Description=description,
                Tags=[
                    {'Key': 'Project', 'Value': self.project_name},
                    {'Key': 'Environment', 'Value': 'dev'}
                ]
            )
            
            logger.info(f"Feature group created: {create_response['FeatureGroupArn']}")
            return create_response['FeatureGroupArn']
            
        except Exception as e:
            logger.error(f"Error creating feature group: {e}")
            raise

    def setup_athena_tables(self, processed_data_path: str, s3_upload_results: Dict) -> Dict:
        """Set up all Athena tables - FINAL FIXED VERSION"""
        logger.info("Setting up Athena tables...")
        
        athena_results = {
            'database_created': False,
            'tables_created': {},
            'table_ddls': {},
            's3_reorganized': False
        }
        
        try:
            # Create Glue database
            athena_results['database_created'] = self.create_glue_database()
            
            # Drop existing tables first
            logger.info("Dropping existing tables...")
            self.drop_existing_tables()
            
            # Reorganize S3 structure
            logger.info("Reorganizing S3 structure...")
            table_locations = self.reorganize_s3_for_athena(s3_upload_results)
            
            if not table_locations:
                logger.error("Failed to reorganize S3 structure")
                return athena_results
            
            athena_results['s3_reorganized'] = True
            athena_results['table_locations'] = table_locations
            
            # Load metadata
            metadata_path = os.path.join(processed_data_path, 'feature_metadata.json')
            if not os.path.exists(metadata_path):
                logger.warning("Feature metadata not found, skipping Athena table creation")
                return athena_results
            
            # Create tables with reorganized locations
            for table_name, s3_location in table_locations.items():
                try:
                    ddl = self.create_athena_table_from_metadata(
                        metadata_path, 
                        table_name,
                        s3_location
                    )
                    
                    athena_results['tables_created'][table_name] = True
                    athena_results['table_ddls'][table_name] = ddl
                    logger.info(f"✓ Created table: {table_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to create table {table_name}: {e}")
                    athena_results['tables_created'][table_name] = False
            
            # Test the first table
            if 'features_complete' in table_locations:
                logger.info("Testing features_complete table...")
                test_query = f"SELECT COUNT(*) as total_records FROM {self.database_name}.features_complete"
                try:
                    response = self.athena_client.start_query_execution(
                        QueryString=test_query,
                        ResultConfiguration={
                            'OutputLocation': f"s3://{self.bucket_name}/athena-results/"
                        },
                        WorkGroup='primary'
                    )
                    
                    success, result = self.wait_for_query_with_result(response['QueryExecutionId'])
                    if success and result:
                        count = result['ResultSet']['Rows'][1]['Data'][0]['VarCharValue']
                        logger.info(f" SUCCESS! Found {count} records in features_complete table")
                        athena_results['test_query_success'] = True
                        athena_results['record_count'] = count
                    else:
                        logger.warning("Test query failed or returned no data")
                        athena_results['test_query_success'] = False
                        
                except Exception as e:
                    logger.warning(f"Test query error: {e}")
                    athena_results['test_query_success'] = False
            
            logger.info("Athena tables setup completed!")
            
        except Exception as e:
            logger.error(f"Error setting up Athena tables: {e}")
            athena_results['error'] = str(e)
        
        return athena_results

    def wait_for_query_with_result(self, query_execution_id: str, timeout: int = 60):
        """Wait for query and return results"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self.athena_client.get_query_execution(QueryExecutionId=query_execution_id)
            status = response['QueryExecution']['Status']['State']
            
            if status == 'SUCCEEDED':
                # Get results
                results = self.athena_client.get_query_results(QueryExecutionId=query_execution_id)
                return True, results
            elif status in ['FAILED', 'CANCELLED']:
                error_reason = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                return False, error_reason
            
            time.sleep(2)
        
        return False, "Query timeout"

    def run_integration_pipeline(self, processed_data_path, enable_feature_store=True, enable_athena=False):
        """Run integration pipeline - FINAL VERSION"""
        logger.info("Starting FINAL FIXED Feature Store integration...")
        
        integration_results = {
            'timestamp': datetime.now().isoformat(),
            'feature_store_results': {},
            'athena_results': {}
        }
        
        try:
            # Upload to S3
            logger.info("Step 1: Uploading data to S3...")
            s3_upload_results = self.upload_data_to_s3(processed_data_path)
            integration_results['s3_uploads'] = s3_upload_results
            
            if enable_feature_store:
                logger.info("Step 2: Setting up Feature Store...")
                
                # Load features
                features_file = os.path.join(processed_data_path, 'features_complete.parquet')
                if os.path.exists(features_file):
                    logger.info(f"Loading feature data from {features_file}...")
                    
                    df_full = pd.read_parquet(features_file)
                    logger.info(f"Loaded {len(df_full)} records for feature store")
                    
                    # Take small sample
                    df_sample = df_full.head(10)
                    logger.info(f"Using {len(df_sample)} records for feature store demo")
                    
                    # Create feature definitions
                    feature_definitions = self.create_feature_definitions(df_sample)
                    
                    # Create feature group
                    feature_group_name = f"{self.feature_group_prefix}-complete"
                    feature_group_arn = self.create_feature_group(
                        feature_group_name,
                        feature_definitions,
                        "Complete feature set for demand stock forecasting"
                    )
                    
                    integration_results['feature_store_results'] = {
                        'feature_group_name': feature_group_name,
                        'feature_group_arn': feature_group_arn,
                        'sample_records': len(df_sample),
                        'total_features': len(feature_definitions),
                        'database_name': self.database_name
                    }
                    
                    logger.info("Feature Store setup completed!")
                    logger.info(f"Custom database created: {self.database_name}")
                else:
                    logger.warning("features_complete.parquet not found")
            
            if enable_athena:
                logger.info("Step 3: Setting up Athena tables with S3 reorganization...")
                athena_results = self.setup_athena_tables(processed_data_path, s3_upload_results)
                integration_results['athena_results'] = athena_results
            
            # Save metadata
            metadata_file = os.path.join(processed_data_path, 'feature_store_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(integration_results, f, indent=2)
            
            logger.info("FINAL FIXED integration completed successfully!")
            
        except Exception as e:
            logger.error(f"Integration failed: {e}")
            integration_results['error'] = str(e)
            raise
        
        return integration_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Feature Store Integration - FINAL FIXED VERSION')
    parser.add_argument('--config', required=True, help='Path to config.yaml')
    parser.add_argument('--data-path', required=True, help='Path to processed data')
    parser.add_argument('--feature-store', action='store_true', help='Enable Feature Store')
    parser.add_argument('--athena', action='store_true', help='Enable Athena')
    parser.add_argument('--athena-only', action='store_true', help='Enable Athena only')
    parser.add_argument('--all', action='store_true', help='Enable both Feature Store and Athena')
    
    args = parser.parse_args()
    
    # Initialize manager
    feature_store_manager = FeatureStoreManager(args.config)
    
    # Determine what to enable
    if args.all:
        enable_feature_store = True
        enable_athena = True
    elif args.athena_only:
        enable_feature_store = False
        enable_athena = True
    elif args.feature_store:
        enable_feature_store = True
        enable_athena = False
    elif args.athena:
        enable_feature_store = False
        enable_athena = True
    else:
        # Default to Feature Store only for backward compatibility
        enable_feature_store = True
        enable_athena = False
    
    # Run integration
    results = feature_store_manager.run_integration_pipeline(
        args.data_path,
        enable_feature_store=enable_feature_store,
        enable_athena=enable_athena
    )
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL FIXED FEATURE STORE INTEGRATION SUMMARY")
    print("="*60)
    
    if 'feature_store_results' in results and results['feature_store_results']:
        fs_results = results['feature_store_results']
        print(f"Feature Store:")
        print(f"  Feature Group: {fs_results.get('feature_group_name', 'N/A')}")
        print(f"  Feature Group ARN: {fs_results.get('feature_group_arn', 'N/A')}")
        print(f"  Total Features: {fs_results.get('total_features', 0)}")
        print(f"  Sample Records: {fs_results.get('sample_records', 0)}")
        print(f"  Database: {fs_results.get('database_name', 'N/A')}")
    
    if 'athena_results' in results and results['athena_results']:
        athena_results = results['athena_results']
        print(f"\nAthena Integration:")
        print(f"  Database Created: {athena_results.get('database_created', False)}")
        print(f"  S3 Reorganized: {athena_results.get('s3_reorganized', False)}")
        
        if 'tables_created' in athena_results:
            print(f"  Tables Created:")
            for table_name, status in athena_results['tables_created'].items():
                status_icon = "✓" if status else "✗"
                print(f"    {status_icon} {table_name}")
        
        if athena_results.get('test_query_success'):
            print(f"   Test Query: SUCCESS ({athena_results.get('record_count', 'N/A')} records)")
        else:
            print(f"    Test Query: Failed or incomplete")
    
    print("\nIntegration completed!")
    
    # Success message
    if 'athena_results' in results and results['athena_results'].get('test_query_success'):
        print("\n Your Athena integration is working!")
        print(" Run: python3 scripts/run_sample_queries.py")
    else:
        print("\n If Athena tables were created, test with:")
        print("   python3 scripts/run_sample_queries.py")


if __name__ == "__main__":
    main()