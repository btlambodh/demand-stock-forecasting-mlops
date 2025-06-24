#!/usr/bin/env python3
"""
Debug Athena Schema Structure
This script will examine the actual structure of Athena DESCRIBE results to understand 
how column names are stored.
"""

import boto3
import yaml
import json
import time
from datetime import datetime


def debug_athena_schema(config_path='config.yaml'):
    """Debug the actual structure of Athena schema results"""
    
    print("="*70)
    print("ATHENA SCHEMA STRUCTURE DEBUGGER")
    print("="*70)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize Athena client
    athena_client = boto3.client('athena', region_name=config['aws']['region'])
    
    database_name = config['aws']['athena']['database_name']
    query_results_location = config['aws']['athena']['query_results_location']
    workgroup = config['aws']['athena'].get('workgroup', 'primary')
    
    print(f"Database: {database_name}")
    print(f"Results Location: {query_results_location}")
    print(f"Workgroup: {workgroup}")
    
    # Test basic connectivity
    print(f"\n1. Testing basic connectivity...")
    test_query = f"SELECT COUNT(*) as record_count FROM {database_name}.features_complete LIMIT 1"
    
    try:
        response = athena_client.start_query_execution(
            QueryString=test_query,
            QueryExecutionContext={'Database': database_name},
            ResultConfiguration={'OutputLocation': query_results_location},
            WorkGroup=workgroup
        )
        
        query_id = response['QueryExecutionId']
        print(f"   Query ID: {query_id}")
        
        # Wait for completion
        while True:
            result = athena_client.get_query_execution(QueryExecutionId=query_id)
            status = result['QueryExecution']['Status']['State']
            
            if status == 'SUCCEEDED':
                print("   Connection test: SUCCESS")
                break
            elif status in ['FAILED', 'CANCELLED']:
                error = result['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                print(f"   Connection test: FAILED - {error}")
                return False
            
            time.sleep(1)
    except Exception as e:
        print(f"   Connection test: ERROR - {e}")
        return False
    
    # Get schema structure
    print(f"\n2. Examining schema structure...")
    schema_query = f"DESCRIBE {database_name}.features_complete"
    
    try:
        response = athena_client.start_query_execution(
            QueryString=schema_query,
            QueryExecutionContext={'Database': database_name},
            ResultConfiguration={'OutputLocation': query_results_location},
            WorkGroup=workgroup
        )
        
        query_id = response['QueryExecutionId']
        print(f"   Schema Query ID: {query_id}")
        
        # Wait for completion
        while True:
            result = athena_client.get_query_execution(QueryExecutionId=query_id)
            status = result['QueryExecution']['Status']['State']
            
            if status == 'SUCCEEDED':
                print("   Schema query: SUCCESS")
                break
            elif status in ['FAILED', 'CANCELLED']:
                error = result['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                print(f"   Schema query: FAILED - {error}")
                return False
            
            time.sleep(1)
        
        # Get the actual results
        results = athena_client.get_query_results(QueryExecutionId=query_id)
        
        print(f"\n3. Analyzing schema results...")
        rows = results['ResultSet']['Rows']
        print(f"   Total rows returned: {len(rows)}")
        
        if len(rows) > 0:
            print(f"\n4. Raw schema structure analysis:")
            
            # Examine first few rows in detail
            for i, row in enumerate(rows[:5]):
                print(f"\n   Row {i}:")
                print(f"      Raw data: {row}")
                
                if 'Data' in row:
                    data_cells = row['Data']
                    print(f"      Data cells: {len(data_cells)}")
                    
                    for j, cell in enumerate(data_cells):
                        print(f"         Cell {j}: {cell}")
                        
                        # Try to extract value from cell
                        value = None
                        for key in ['VarCharValue', 'LongValue', 'DoubleValue', 'BooleanValue']:
                            if key in cell and cell[key] is not None:
                                value = cell[key]
                                break
                        
                        if value:
                            print(f"            Extracted value: '{value}'")
                        else:
                            print(f"            No value found in cell")
            
            # Try to extract headers from first row
            print(f"\n5. Header extraction attempt:")
            if len(rows) > 0:
                header_row = rows[0]['Data']
                headers = []
                
                for i, cell in enumerate(header_row):
                    value = None
                    for key in ['VarCharValue', 'LongValue', 'DoubleValue', 'BooleanValue']:
                        if key in cell and cell[key] is not None:
                            value = str(cell[key])
                            break
                    
                    if value:
                        headers.append(value)
                        print(f"      Header {i}: '{value}'")
                    else:
                        headers.append('')
                        print(f"      Header {i}: (empty)")
                
                print(f"   Extracted headers: {headers}")
            
            # Try to extract actual column names from data rows
            print(f"\n6. Column name extraction from data rows:")
            potential_columns = []
            
            for i, row in enumerate(rows[1:11], 1):  # Skip header, check first 10 data rows
                if 'Data' in row and len(row['Data']) > 0:
                    # Get first cell value (likely the column name)
                    first_cell = row['Data'][0]
                    value = None
                    
                    for key in ['VarCharValue', 'LongValue', 'DoubleValue', 'BooleanValue']:
                        if key in first_cell and first_cell[key] is not None:
                            value = str(first_cell[key])
                            break
                    
                    if value and not value.startswith('#'):
                        potential_columns.append(value)
                        print(f"      Row {i}, Column: '{value}'")
                        
                        # Show other cells in this row (data type, comments, etc.)
                        if len(row['Data']) > 1:
                            other_values = []
                            for cell in row['Data'][1:]:
                                cell_value = None
                                for key in ['VarCharValue', 'LongValue', 'DoubleValue', 'BooleanValue']:
                                    if key in cell and cell[key] is not None:
                                        cell_value = str(cell[key])
                                        break
                                if cell_value:
                                    other_values.append(cell_value)
                            print(f"         Other cells: {other_values}")
            
            print(f"\n7. Summary:")
            print(f"   Potential column names found: {len(potential_columns)}")
            print(f"   First 10 columns: {potential_columns[:10]}")
            
            if len(potential_columns) > 10:
                print(f"   Last 5 columns: {potential_columns[-5:]}")
            
            # Check if any of our expected columns are in the list
            expected_columns = [
                'Total_Quantity', 'Avg_Price', 'Revenue', 'Month', 'Year',
                'Category Code', 'Loss Rate (%)', 'Wholesale Price (RMB/kg)'
            ]
            
            print(f"\n8. Checking for expected columns:")
            found_expected = []
            for expected in expected_columns:
                matches = [col for col in potential_columns if expected.lower() in col.lower() or col.lower() in expected.lower()]
                if matches:
                    found_expected.extend(matches)
                    print(f"   '{expected}' -> Found matches: {matches}")
                else:
                    print(f"   '{expected}' -> No matches found")
            
            print(f"\n9. Recommendations:")
            if potential_columns:
                print(f"   Column extraction IS working - found {len(potential_columns)} columns")
                print(f"   Update bi_dashboard_generator.py column mapping with these actual names:")
                print(f"   ")
                print(f"   # Sample mapping based on found columns:")
                for i, col in enumerate(potential_columns[:20]):
                    # Try to guess the mapping
                    if 'quantity' in col.lower():
                        print(f"   'total_quantity': '{col}',")
                    elif 'price' in col.lower() and 'avg' in col.lower():
                        print(f"   'avg_price': '{col}',")
                    elif 'revenue' in col.lower():
                        print(f"   'revenue': '{col}',")
                    elif 'month' in col.lower():
                        print(f"   'month': '{col}',")
                    elif 'year' in col.lower():
                        print(f"   'year': '{col}',")
                    else:
                        print(f"   'feature_{i}': '{col}',")
            else:
                print(f"   Column extraction FAILED - no columns found")
                print(f"   This suggests the DESCRIBE result structure is different than expected")
                print(f"   Check the raw data structure above to understand the format")
        
        return True
        
    except Exception as e:
        print(f"   Schema query: ERROR - {e}")
        return False


if __name__ == "__main__":
    debug_athena_schema()
