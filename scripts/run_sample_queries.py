#!/usr/bin/env python3
"""
Fixed Sample BI Queries Script
Properly handles database context and column escaping

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import boto3
import yaml
import time
import sys


def load_config():
    """Load configuration"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)


def wait_for_query(athena_client, query_execution_id, timeout=60):
    """Wait for query to complete and return results"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        status = response['QueryExecution']['Status']['State']
        
        if status == 'SUCCEEDED':
            # Get results
            results = athena_client.get_query_results(QueryExecutionId=query_execution_id)
            return True, results
        elif status in ['FAILED', 'CANCELLED']:
            error_reason = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
            return False, error_reason
        
        time.sleep(2)
    
    return False, "Query timeout"


def run_sample_queries():
    """Run sample BI queries with proper database context"""
    print("🔍 Running Fixed Sample Business Intelligence Queries")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Initialize Athena client
    athena_client = boto3.client('athena', region_name=config['aws']['region'])
    database_name = config['aws']['athena']['database_name']
    
    # FIXED: Sample queries with proper database context and simplified column names
    queries = [
        {
            'name': 'Total Records Count',
            'sql': f'SELECT COUNT(*) as total_records FROM {database_name}.features_complete'
        },
        {
            'name': 'Unique Items Count', 
            'sql': f'SELECT COUNT(DISTINCT "Item Code") as unique_items FROM {database_name}.features_complete'
        },
        {
            'name': 'Date Range',
            'sql': f'SELECT MIN("Date") as earliest_date, MAX("Date") as latest_date FROM {database_name}.features_complete'
        },
        {
            'name': 'Revenue Summary',
            'sql': f'SELECT SUM("Revenue") as total_revenue, AVG("Revenue") as avg_revenue FROM {database_name}.features_complete WHERE "Revenue" IS NOT NULL'
        },
        {
            'name': 'Top Categories by Records',
            'sql': f'SELECT "Category Name", COUNT(*) as records FROM {database_name}.features_complete WHERE "Category Name" IS NOT NULL GROUP BY "Category Name" ORDER BY records DESC LIMIT 5'
        }
    ]
    
    results_summary = []
    
    for i, query_info in enumerate(queries, 1):
        print(f"\n{i}. {query_info['name']}")
        print(f"   Query: {query_info['sql'][:80]}{'...' if len(query_info['sql']) > 80 else ''}")
        
        try:
            # FIXED: Submit query with proper QueryExecutionContext
            response = athena_client.start_query_execution(
                QueryString=query_info['sql'],
                QueryExecutionContext={
                    'Database': database_name
                },
                ResultConfiguration={
                    'OutputLocation': config['aws']['athena']['query_results_location']
                },
                WorkGroup=config['aws']['athena'].get('workgroup', 'primary')
            )
            
            query_execution_id = response['QueryExecutionId']
            print(f"   Query ID: {query_execution_id}")
            
            # Wait for results
            success, result = wait_for_query(athena_client, query_execution_id)
            
            if success:
                # Display results
                rows = result['ResultSet']['Rows']
                if len(rows) > 1:  # Skip header row
                    print("   Results:")
                    # Header
                    headers = [col['VarCharValue'] for col in rows[0]['Data']]
                    print(f"   {'  |  '.join(headers)}")
                    print(f"   {'-' * (len('  |  '.join(headers)))}")
                    
                    # Data rows (limit to first 5)
                    for row in rows[1:6]:
                        values = []
                        for cell in row['Data']:
                            if 'VarCharValue' in cell:
                                values.append(cell['VarCharValue'])
                            else:
                                values.append('NULL')
                        print(f"   {'  |  '.join(values)}")
                    
                    if len(rows) > 6:
                        print(f"   ... ({len(rows)-1} total rows)")
                    
                    results_summary.append(f"✓ {query_info['name']}: SUCCESS")
                else:
                    print("   No data returned")
                    results_summary.append(f"⚠️  {query_info['name']}: NO DATA")
            else:
                print(f"   ❌ Failed: {result}")
                results_summary.append(f"❌ {query_info['name']}: FAILED")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results_summary.append(f"❌ {query_info['name']}: ERROR")
    
    # Summary
    print("\n" + "=" * 60)
    print(" Query Results Summary")
    print("=" * 60)
    
    for summary in results_summary:
        print(summary)
    
    successful_queries = len([s for s in results_summary if s.startswith('✓')])
    total_queries = len(results_summary)
    
    print(f"\n {successful_queries}/{total_queries} queries completed successfully")
    
    if successful_queries == total_queries:
        print("\n All sample queries completed successfully!")
        print(" Your Athena integration is working perfectly!")
        print("\n Next Steps:")
        print("   • Open AWS Athena Console")
        print("   • Use database: demand_stock_forecasting_mlops_feature_store")
        print("   • Try your own custom queries!")
    elif successful_queries > 0:
        print(f"\n Basic functionality working! {successful_queries} queries succeeded")
        print("💡 For remaining issues, try these manual queries in Athena Console:")
        
        for query_info in queries:
            print(f"   • {query_info['sql']}")
    else:
        print(f"\n  All queries had issues")
        print("💡 Try this simple test in Athena Console:")
        print(f"   SELECT * FROM {database_name}.features_complete LIMIT 10;")


def test_simple_query():
    """Test with the simplest possible query"""
    print("\n Testing Simple Query")
    print("=" * 30)
    
    config = load_config()
    athena_client = boto3.client('athena', region_name=config['aws']['region'])
    database_name = config['aws']['athena']['database_name']
    
    simple_query = f"SELECT * FROM {database_name}.features_complete LIMIT 5"
    
    try:
        response = athena_client.start_query_execution(
            QueryString=simple_query,
            QueryExecutionContext={'Database': database_name},
            ResultConfiguration={
                'OutputLocation': config['aws']['athena']['query_results_location']
            },
            WorkGroup=config['aws']['athena'].get('workgroup', 'primary')
        )
        
        query_id = response['QueryExecutionId']
        print(f"✓ Simple query submitted: {query_id}")
        print("✓ Check Athena console for results!")
        
        # Wait a bit and check status
        time.sleep(5)
        status_response = athena_client.get_query_execution(QueryExecutionId=query_id)
        status = status_response['QueryExecution']['Status']['State']
        print(f"✓ Query status: {status}")
        
    except Exception as e:
        print(f" Simple query failed: {e}")


if __name__ == "__main__":
    # Run main queries
    run_sample_queries()
    
    # Also test simple query
    test_simple_query()