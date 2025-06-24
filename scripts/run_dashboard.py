#!/usr/bin/env python3
"""
Dashboard Management Script - Updated Version
Handles all dashboard operations with improved error handling and column verification

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import os
import sys
import json
import time
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dashboard_commands')


def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)


def print_success(message):
    """Print success message"""
    print(f"SUCCESS: {message}")


def print_error(message):
    """Print error message"""
    print(f"ERROR: {message}")


def print_warning(message):
    """Print warning message"""
    print(f"WARNING: {message}")


def print_info(message):
    """Print info message"""
    print(f"INFO: {message}")


def setup_dashboard():
    """Set up dashboard infrastructure"""
    print_header("Setting up dashboard infrastructure")
    
    try:
        # Create necessary directories
        directories = [
            'dashboard_data',
            'reports', 
            'src/dashboard',
            'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"  Created directory: {directory}")
        
        # Check essential files
        essential_files = {
            'config.yaml': 'Configuration file',
            'src/dashboard/bi_dashboard_generator.py': 'BI Dashboard Generator',
            'src/dashboard/dashboard_viewer.py': 'Dashboard Viewer'
        }
        
        missing_files = []
        for file_path, description in essential_files.items():
            if os.path.exists(file_path):
                print(f"  Found: {description}")
            else:
                print(f"  Missing: {description} ({file_path})")
                missing_files.append(file_path)
        
        if missing_files:
            print_warning(f"Missing {len(missing_files)} essential files")
            print("  Run the following to check what's available:")
            print("  python3 scripts/run_dashboard.py debug")
            return False
        
        print_success("Dashboard infrastructure setup completed!")
        return True
        
    except Exception as e:
        print_error(f"Setup failed: {e}")
        return False


def verify_column_names():
    """Verify that the column names in the generator match the actual database"""
    print_header("Verifying database column names")
    
    try:
        # Import the dashboard generator to check table schema
        sys.path.append('src/dashboard')
        from bi_dashboard_generator import BIDashboardGenerator
        
        dashboard_generator = BIDashboardGenerator('config.yaml')
        
        # Test basic connectivity first
        print_info("Testing database connectivity...")
        test_query = f"SELECT COUNT(*) as total_count FROM {dashboard_generator.database_name}.features_complete LIMIT 1"
        result = dashboard_generator.execute_query(test_query, "connectivity_test")
        
        if not result:
            print_error("Cannot connect to Athena database")
            print("  Check your AWS credentials and config.yaml settings")
            return False
        
        print_success(f"Database connection successful ({result[0]['total_count']:,} records)")
        
        # Get actual table schema
        print_info("Retrieving table schema...")
        schema_query = f"DESCRIBE {dashboard_generator.database_name}.features_complete"
        schema_result = dashboard_generator.execute_query(schema_query, "schema_discovery")
        
        if not schema_result:
            print_error("Cannot retrieve table schema")
            return False
        
        # Extract available columns
        available_columns = []
        for row in schema_result:
            col_name = row.get('col_name', '')
            if col_name and not col_name.startswith('#'):
                available_columns.append(col_name)
        
        print_success(f"Found {len(available_columns)} columns in database")
        
        # Check which expected columns exist
        expected_columns = dashboard_generator.columns
        missing_columns = []
        existing_columns = []
        
        print_info("Checking column availability:")
        for key, col_name in expected_columns.items():
            if col_name in available_columns:
                existing_columns.append(col_name)
                print(f"  FOUND {key}: '{col_name}'")
            else:
                missing_columns.append((key, col_name))
                print(f"  MISSING {key}: '{col_name}'")
        
        if missing_columns:
            print_warning(f"Missing {len(missing_columns)} expected columns")
            print("\nAvailable columns in database:")
            for col in sorted(available_columns):
                print(f"    {col}")
            
            print("\nSuggested improvements:")
            print("1. Update the column mapping in bi_dashboard_generator.py")
            print("2. Check if column names have changed in your data pipeline")
            print("3. Verify the table structure matches your expectations")
            return False
        else:
            print_success("All expected columns found in database!")
            return True
            
    except Exception as e:
        print_error(f"Column verification failed: {e}")
        logger.exception("Column verification error")
        return False


def test_individual_queries():
    """Test individual queries to identify which ones are failing"""
    print_header("Testing individual dashboard queries")
    
    try:
        sys.path.append('src/dashboard')
        from bi_dashboard_generator import BIDashboardGenerator
        
        dashboard_generator = BIDashboardGenerator('config.yaml')
        
        # Define test queries for each component
        test_queries = {
            'basic_count': f"SELECT COUNT(*) as count FROM {dashboard_generator.database_name}.features_complete",
            'revenue_sum': f"SELECT SUM(\"{dashboard_generator.columns['revenue']}\") as total_revenue FROM {dashboard_generator.database_name}.features_complete WHERE \"{dashboard_generator.columns['revenue']}\" IS NOT NULL",
            'monthly_revenue': f"SELECT \"{dashboard_generator.columns['year']}\" as year, \"{dashboard_generator.columns['month']}\" as month, SUM(\"{dashboard_generator.columns['revenue']}\") as monthly_revenue FROM {dashboard_generator.database_name}.features_complete WHERE \"{dashboard_generator.columns['revenue']}\" IS NOT NULL GROUP BY \"{dashboard_generator.columns['year']}\", \"{dashboard_generator.columns['month']}\" ORDER BY \"{dashboard_generator.columns['year']}\", \"{dashboard_generator.columns['month']}\" LIMIT 5"
        }
        
        results = {}
        for query_name, query in test_queries.items():
            print_info(f"Testing query: {query_name}")
            result = dashboard_generator.execute_query(query, f"test_{query_name}")
            
            if result is not None:
                results[query_name] = len(result)
                print_success(f"{query_name}: returned {len(result)} rows")
                if len(result) > 0:
                    print(f"  Sample result: {result[0]}")
            else:
                results[query_name] = 0
                print_error(f"{query_name}: failed")
        
        # Analysis
        print_info("Query test analysis:")
        all_passed = True
        for query_name, row_count in results.items():
            if row_count > 0:
                print(f"  PASSED {query_name}: {row_count} rows")
            else:
                print(f"  FAILED {query_name}: query failed")
                all_passed = False
        
        if all_passed:
            print_success("All test queries passed!")
        else:
            print_warning("Some queries failed - check column names and data availability")
        
        return all_passed
        
    except Exception as e:
        print_error(f"Query testing failed: {e}")
        logger.exception("Query testing error")
        return False


def generate_dashboard_data(component=None):
    """Generate dashboard data using the BI generator with enhanced error handling"""
    if component:
        print_header(f"Generating {component} component")
    else:
        print_header("Generating complete dashboard data")
    
    try:
        # Check if generator exists
        generator_path = 'src/dashboard/bi_dashboard_generator.py'
        if not os.path.exists(generator_path):
            print_error(f"Generator not found: {generator_path}")
            return False
        
        # Check if config exists
        if not os.path.exists('config.yaml'):
            print_error("config.yaml not found")
            return False
        
        # Verify columns before generating (for complete dashboard only)
        if not component:
            print_info("Verifying database setup before generation...")
            if not verify_column_names():
                print_error("Column verification failed - stopping generation")
                return False
            
            if not test_individual_queries():
                print_warning("Some test queries failed - proceeding with caution")
        
        # Build command
        cmd = [
            sys.executable, generator_path,
            '--config', 'config.yaml',
            '--output', 'dashboard_data'
        ]
        
        if component:
            cmd.extend(['--component', component])
        
        print_info(f"Executing: {' '.join(cmd)}")
        
        # Run the generator with extended timeout
        print_info("Starting dashboard generation (this may take several minutes)...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minutes
        
        if result.returncode == 0:
            print_success("Dashboard data generated successfully!")
            if result.stdout:
                print("Generator output:")
                print(result.stdout)
            
            # Verify output files
            verify_dashboard_data()
            return True
        else:
            print_error("Dashboard generation failed!")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            if result.stdout:
                print("Standard output:")
                print(result.stdout)
            
            # Try to provide helpful error analysis
            analyze_generation_error(result.stderr, result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print_error("Dashboard generation timed out (10 minutes)")
        print_info("Try generating individual components:")
        print("  python3 scripts/run_dashboard.py overview")
        print("  python3 scripts/run_dashboard.py revenue")
        return False
    except Exception as e:
        print_error(f"Dashboard generation failed: {e}")
        logger.exception("Dashboard generation error")
        return False


def analyze_generation_error(stderr, stdout):
    """Analyze generation errors and provide helpful suggestions"""
    print_info("Analyzing error...")
    
    error_suggestions = {
        'column': "Column name mismatch - run 'python3 scripts/run_dashboard.py debug'",
        'timeout': "Query timeout - try generating individual components",
        'permission': "AWS permission error - check your credentials",
        'syntax': "SQL syntax error - check the query formatting",
        'connection': "Database connection issue - verify your config.yaml"
    }
    
    combined_output = (stderr or '') + (stdout or '')
    combined_output_lower = combined_output.lower()
    
    suggestions_found = []
    for error_type, suggestion in error_suggestions.items():
        if error_type in combined_output_lower:
            suggestions_found.append(suggestion)
    
    if suggestions_found:
        print_info("Suggested improvements:")
        for suggestion in suggestions_found:
            print(f"  • {suggestion}")
    else:
        print_info("General troubleshooting steps:")
        print("  • Run: python3 scripts/run_dashboard.py debug")
        print("  • Check AWS credentials: aws sts get-caller-identity")
        print("  • Verify config.yaml settings")
        print("  • Try individual components first")


def verify_dashboard_data():
    """Verify dashboard data files exist and contain data"""
    print_info("Verifying dashboard data files...")
    
    data_file = 'dashboard_data/dashboard_data.json'
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            print_success("Main data file exists")
            print(f"  Data components: {len(data)}")
            
            # Check individual components with expected counts
            expected_datasets = {
                'overview': 5,
                'revenue_trends': 3,
                'category_analysis': 3,
                'price_analysis': 3,
                'market_insights': 4,
                'forecasting_features': 3
            }
            
            print_info("Component verification:")
            total_expected = 0
            total_actual = 0
            
            for component, expected_count in expected_datasets.items():
                total_expected += expected_count
                if component in data:
                    if isinstance(data[component], dict):
                        actual_count = len(data[component])
                        total_actual += actual_count
                        status = "COMPLETE" if actual_count == expected_count else "PARTIAL"
                        print(f"  {status} {component}: {actual_count}/{expected_count} datasets")
                        
                        # Show which datasets are available
                        if actual_count > 0:
                            datasets = list(data[component].keys())
                            print(f"      Available: {', '.join(datasets)}")
                    else:
                        print(f"  COMPLETE {component}: Available (single dataset)")
                        total_actual += 1
                else:
                    print(f"  MISSING {component}: 0/{expected_count}")
            
            print_info(f"Overall completion: {total_actual}/{total_expected} datasets ({total_actual/total_expected*100:.1f}%)")
            
            if total_actual < total_expected:
                print_warning("Some datasets are missing. Try:")
                print("  python3 scripts/run_dashboard.py generate  # Regenerate all")
                print("  python3 scripts/run_dashboard.py debug    # Check for issues")
            
            return True
            
        except Exception as e:
            print_error(f"Error reading dashboard data: {e}")
            return False
    else:
        print_error("Dashboard data file not found")
        print_info("Generate data first: python3 scripts/run_dashboard.py generate")
        return False


def view_dashboard_data():
    """View dashboard data in terminal with enhanced formatting"""
    print_header("Dashboard data summary")
    
    data_file = 'dashboard_data/dashboard_data.json'
    
    if not os.path.exists(data_file):
        print_error("Dashboard data not found. Generate data first:")
        print("  python3 scripts/run_dashboard.py generate")
        return False
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Overview metrics
        if 'overview' in data:
            overview = data['overview']
            print("\nOVERVIEW METRICS:")
            
            if 'date_range' in overview:
                date_range = overview['date_range']
                print(f"  Data Period: {date_range.get('earliest_date', 'N/A')} to {date_range.get('latest_date', 'N/A')}")
            
            if 'total_records' in overview:
                records = overview['total_records'].get('total_records', 0)
                print(f"  Total Records: {records:,}")
            
            if 'total_revenue' in overview:
                revenue_data = overview['total_revenue']
                total_revenue = revenue_data.get('total_revenue', 0)
                avg_revenue = revenue_data.get('avg_revenue', 0)
                transaction_count = revenue_data.get('transaction_count', 0)
                
                print(f"  Total Revenue: {total_revenue:,.2f} RMB")
                print(f"  Total Transactions: {transaction_count:,}")
                print(f"  Average Revenue: {avg_revenue:.2f} RMB")
            
            if 'unique_items' in overview:
                unique = overview['unique_items'].get('unique_items', 0)
                print(f"  Unique Categories: {unique:,}")
            elif 'unique_categories' in overview:
                unique = overview['unique_categories'].get('unique_categories', 0)
                print(f"  Unique Categories: {unique:,}")
        
        # Revenue trends
        if 'revenue_trends' in data and 'monthly_revenue' in data['revenue_trends']:
            trends = data['revenue_trends']['monthly_revenue']
            print(f"\nREVENUE TRENDS:")
            print(f"  Available months: {len(trends)}")
            
            if len(trends) >= 3:
                recent = trends[-3:]
                print("  Recent months:")
                for month in recent:
                    year = month.get('year', 'N/A')
                    month_num = month.get('month', 'N/A')
                    revenue = month.get('monthly_revenue', 0)
                    print(f"    {year}-{month_num:02d}: {revenue:,.0f} RMB")
        elif 'revenue_trends' in data:
            print(f"\nREVENUE TRENDS: No monthly data available")
            available_trends = list(data['revenue_trends'].keys()) if data['revenue_trends'] else []
            if available_trends:
                print(f"    Available: {', '.join(available_trends)}")
        
        # Category analysis
        if 'category_analysis' in data and 'top_categories_by_revenue' in data['category_analysis']:
            categories = data['category_analysis']['top_categories_by_revenue'][:5]
            print(f"\nTOP 5 CATEGORIES:")
            for i, cat in enumerate(categories, 1):
                name = cat.get('Category_Name', 'Unknown')
                revenue = cat.get('total_revenue', 0)
                transactions = cat.get('transaction_count', 0)
                print(f"  {i}. {name}: {revenue:,.0f} RMB ({transactions:,} transactions)")
        
        # Market insights
        if 'market_insights' in data:
            insights = data['market_insights']
            print(f"\nMARKET INSIGHTS:")
            
            if 'weekend_vs_weekday' in insights:
                weekend_data = insights['weekend_vs_weekday']
                print("  Weekend vs Weekday:")
                for day_data in weekend_data:
                    day_type = day_data.get('day_type', 'Unknown')
                    revenue = day_data.get('total_revenue', 0)
                    transactions = day_data.get('transaction_count', 0)
                    print(f"    {day_type}: {revenue:,.0f} RMB ({transactions:,} transactions)")
        
        # Data components summary
        print(f"\nDATA COMPONENTS:")
        for component_name, component_data in data.items():
            if isinstance(component_data, dict):
                sub_components = len(component_data)
                status = "AVAILABLE" if sub_components > 0 else "EMPTY"
                print(f"  {status} {component_name.replace('_', ' ').title()}: {sub_components} datasets")
            else:
                print(f"  AVAILABLE {component_name.replace('_', ' ').title()}: Available")
        
        print_success("Dashboard data loaded and displayed successfully!")
        return True
        
    except Exception as e:
        print_error(f"Error reading dashboard data: {e}")
        return False


def generate_html_dashboard():
    """Generate HTML dashboard"""
    print_header("Generating HTML dashboard")
    
    try:
        # Check if dashboard data exists
        if not os.path.exists('dashboard_data/dashboard_data.json'):
            print_error("Dashboard data not found. Generate data first:")
            print("  python3 scripts/run_dashboard.py generate")
            return False
        
        # Check if viewer exists
        viewer_path = 'src/dashboard/dashboard_viewer.py'
        if not os.path.exists(viewer_path):
            print_error(f"Dashboard viewer not found: {viewer_path}")
            return False
        
        # Generate HTML dashboard
        cmd = [sys.executable, viewer_path, '--html']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print_success("HTML dashboard generated successfully!")
            print(result.stdout)
            
            # Check if HTML file was created
            html_file = 'dashboard_data/dashboard.html'
            if os.path.exists(html_file):
                print(f"\nDashboard ready!")
                print(f"  HTML file: {html_file}")
                print(f"  To serve the dashboard:")
                print(f"    cd dashboard_data && python3 -m http.server 8080")
                print(f"  Then open: http://localhost:8080/dashboard.html")
                return True
            else:
                print_warning("HTML file not found after generation")
                return False
        else:
            print_error("HTML generation failed!")
            print(result.stderr)
            return False
            
    except Exception as e:
        print_error(f"HTML generation failed: {e}")
        return False


def run_streamlit_dashboard():
    """Run Streamlit dashboard"""
    print_header("Starting Streamlit dashboard")
    
    try:
        # Check if dashboard data exists
        if not os.path.exists('dashboard_data/dashboard_data.json'):
            print_warning("Dashboard data not found. Generating now...")
            if not generate_dashboard_data():
                return False
        
        # Check if viewer exists
        viewer_path = 'src/dashboard/dashboard_viewer.py'
        if not os.path.exists(viewer_path):
            print_error(f"Dashboard viewer not found: {viewer_path}")
            return False
        
        # Check if streamlit is available
        try:
            import streamlit
            print("Streamlit is available")
        except ImportError:
            print_error("Streamlit not installed. Install with: pip install streamlit")
            return False
        
        print("Starting Streamlit dashboard...")
        print("   Dashboard will open in your browser automatically")
        print("   Press Ctrl+C to stop the dashboard")
        
        # Run streamlit
        cmd = ['streamlit', 'run', viewer_path]
        subprocess.run(cmd)
        
        return True
        
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
        return True
    except Exception as e:
        print_error(f"Streamlit dashboard failed: {e}")
        return False


def generate_daily_report():
    """Generate daily business report"""
    print_header("Generating daily business report")
    
    try:
        # Check if dashboard data exists
        data_file = 'dashboard_data/dashboard_data.json'
        if not os.path.exists(data_file):
            print_error("Dashboard data not found. Generate data first:")
            print("  python3 scripts/run_dashboard.py generate")
            return False
        
        # Load dashboard data
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Create reports directory
        os.makedirs('reports', exist_ok=True)
        
        # Generate report
        report_date = datetime.now().strftime('%Y-%m-%d')
        report_file = f'reports/daily_report_{report_date}.txt'
        
        with open(report_file, 'w') as f:
            f.write(f"DAILY BUSINESS REPORT - {report_date}\n")
            f.write("=" * 50 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY:\n")
            f.write("-" * 20 + "\n")
            
            if 'overview' in data:
                overview = data['overview']
                
                if 'total_revenue' in overview:
                    revenue_data = overview['total_revenue']
                    total_revenue = revenue_data.get('total_revenue', 0)
                    avg_revenue = revenue_data.get('avg_revenue', 0)
                    transaction_count = revenue_data.get('transaction_count', 0)
                    
                    f.write(f"Total Revenue: {total_revenue:,.2f} RMB\n")
                    f.write(f"Total Transactions: {transaction_count:,}\n")
                    f.write(f"Average Revenue per Transaction: {avg_revenue:.2f} RMB\n")
                
                if 'total_records' in overview:
                    records = overview['total_records'].get('total_records', 0)
                    f.write(f"Total Records Processed: {records:,}\n")
                
                if 'date_range' in overview:
                    date_range = overview['date_range']
                    f.write(f"Data Coverage: {date_range.get('earliest_date', 'N/A')} to {date_range.get('latest_date', 'N/A')}\n")
            
            # Top performing categories
            if 'category_analysis' in data and 'top_categories_by_revenue' in data['category_analysis']:
                f.write(f"\nTOP PERFORMING CATEGORIES:\n")
                f.write("-" * 30 + "\n")
                
                categories = data['category_analysis']['top_categories_by_revenue'][:5]
                for i, cat in enumerate(categories, 1):
                    name = cat.get('Category_Name', 'Unknown')
                    revenue = cat.get('total_revenue', 0)
                    transactions = cat.get('transaction_count', 0)
                    f.write(f"{i}. {name}\n")
                    f.write(f"   Revenue: {revenue:,.0f} RMB\n")
                    f.write(f"   Transactions: {transactions:,}\n\n")
            
            # Recent trends
            if 'revenue_trends' in data and 'monthly_revenue' in data['revenue_trends']:
                trends = data['revenue_trends']['monthly_revenue']
                if len(trends) >= 3:
                    f.write(f"RECENT REVENUE TRENDS:\n")
                    f.write("-" * 25 + "\n")
                    
                    recent = trends[-3:]
                    for month in recent:
                        year = month.get('year', 'N/A')
                        month_num = month.get('month', 'N/A')
                        revenue = month.get('monthly_revenue', 0)
                        transactions = month.get('transaction_count', 0)
                        f.write(f"{year}-{month_num:02d}: {revenue:,.0f} RMB ({transactions:,} transactions)\n")
            
            # Market insights
            if 'market_insights' in data and 'weekend_vs_weekday' in data['market_insights']:
                f.write(f"\nMARKET INSIGHTS:\n")
                f.write("-" * 20 + "\n")
                
                weekend_data = data['market_insights']['weekend_vs_weekday']
                for day_data in weekend_data:
                    day_type = day_data.get('day_type', 'Unknown')
                    revenue = day_data.get('total_revenue', 0)
                    transactions = day_data.get('transaction_count', 0)
                    avg_price = day_data.get('avg_price', 0)
                    f.write(f"{day_type}:\n")
                    f.write(f"  Revenue: {revenue:,.0f} RMB\n")
                    f.write(f"  Transactions: {transactions:,}\n")
                    f.write(f"  Average Price: {avg_price:.2f} RMB\n\n")
            
            # Recommendations
            f.write(f"RECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            f.write("1. Monitor seasonal patterns in top-performing categories\n")
            f.write("2. Analyze price volatility trends for optimization opportunities\n")
            f.write("3. Focus marketing efforts on high-revenue categories\n")
            f.write("4. Review weekend vs weekday performance for scheduling\n")
            f.write("5. Track loss rate impact on revenue performance\n")
            
            f.write(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Generated by Chinese Produce Market Analytics Dashboard\n")
        
        print_success(f"Daily report generated: {report_file}")
        
        # Display report summary
        print("\nREPORT SUMMARY:")
        with open(report_file, 'r') as f:
            lines = f.readlines()[:20]  # Show first 20 lines
            for line in lines:
                print(f"  {line.rstrip()}")
        
        if len(lines) >= 20:
            print("  ... (see full report in file)")
        
        return True
        
    except Exception as e:
        print_error(f"Report generation failed: {e}")
        return False


def check_system_status():
    """Check dashboard system status"""
    print_header("Dashboard system status")
    
    # Check files
    files_to_check = [
        ('Config File', 'config.yaml'),
        ('BI Generator', 'src/dashboard/bi_dashboard_generator.py'),
        ('Dashboard Viewer', 'src/dashboard/dashboard_viewer.py'),
        ('Dashboard Data', 'dashboard_data/dashboard_data.json'),
        ('HTML Dashboard', 'dashboard_data/dashboard.html')
    ]
    
    print("FILE STATUS:")
    all_files_ok = True
    for name, path in files_to_check:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  EXISTS {name}: ({size:,} bytes)")
        else:
            print(f"  MISSING {name}: ({path})")
            all_files_ok = False
    
    # Check directories
    print("\nDIRECTORY STATUS:")
    directories = ['dashboard_data', 'reports', 'src/dashboard']
    for directory in directories:
        if os.path.exists(directory):
            file_count = len(os.listdir(directory))
            print(f"  EXISTS {directory}: ({file_count} files)")
        else:
            print(f"  MISSING {directory}")
    
    # Check Python dependencies
    print("\nPYTHON DEPENDENCIES:")
    dependencies = [
        ('yaml', 'PyYAML'),
        ('boto3', 'Boto3'),
        ('pandas', 'Pandas'),
        ('json', 'JSON (built-in)'),
        ('streamlit', 'Streamlit (optional)')
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  AVAILABLE {name}")
        except ImportError:
            print(f"  MISSING {name}")
            if module != 'streamlit':
                all_files_ok = False
    
    # Check dashboard data content
    if os.path.exists('dashboard_data/dashboard_data.json'):
        print("\nDASHBOARD DATA STATUS:")
        try:
            with open('dashboard_data/dashboard_data.json', 'r') as f:
                data = json.load(f)
            
            components = ['overview', 'revenue_trends', 'category_analysis', 'price_analysis', 'market_insights']
            for component in components:
                if component in data:
                    if isinstance(data[component], dict):
                        count = len(data[component])
                        status = "AVAILABLE" if count > 0 else "EMPTY"
                        print(f"  {status} {component}: {count} datasets")
                    else:
                        print(f"  AVAILABLE {component}")
                else:
                    print(f"  MISSING {component}")
            
        except Exception as e:
            print(f"  CORRUPTED Data file: {e}")
            all_files_ok = False
    
    # Overall status
    print(f"\nOVERALL STATUS:")
    if all_files_ok:
        print("  Dashboard system is ready!")
        print("\nNext steps:")
        print("  • python3 scripts/run_dashboard.py view      # View current data")
        print("  • python3 scripts/run_dashboard.py streamlit # Launch interactive dashboard")
        print("  • python3 scripts/run_dashboard.py html     # Generate static HTML")
    else:
        print("  Dashboard system has issues")
        print("\nRecommended actions:")
        print("  • python3 scripts/run_dashboard.py setup    # Setup infrastructure")
        print("  • python3 scripts/run_dashboard.py debug    # Check database connectivity")
        print("  • python3 scripts/run_dashboard.py generate # Generate missing data")
    
    return all_files_ok


def debug_table_schema():
    """Debug the actual table schema to identify available columns"""
    print_header("Debugging Table Schema and Connectivity")
    
    try:
        # Import the dashboard generator to check table schema
        sys.path.append('src/dashboard')
        from bi_dashboard_generator import BIDashboardGenerator
        
        print_info("Initializing dashboard generator...")
        dashboard_generator = BIDashboardGenerator('config.yaml')
        
        # Test basic connectivity
        print_info("Testing database connectivity...")
        test_query = f"SELECT COUNT(*) as total_count FROM {dashboard_generator.database_name}.features_complete LIMIT 1"
        result = dashboard_generator.execute_query(test_query, "connectivity_test")
        
        if result:
            print_success(f"Database connection successful ({result[0]['total_count']:,} records)")
        else:
            print_error("Cannot connect to Athena table")
            print_info("Check:")
            print("  • AWS credentials: aws sts get-caller-identity")
            print("  • config.yaml database settings")
            print("  • VPN/network connection")
            return False
        
        # Get table schema
        print_info("Retrieving table schema...")
        schema_query = f"DESCRIBE {dashboard_generator.database_name}.features_complete"
        schema_result = dashboard_generator.execute_query(schema_query, "schema_discovery")
        
        if schema_result:
            print_success(f"Schema retrieved ({len(schema_result)} rows)")
            
            # Debug the schema result structure
            print_info("Analyzing schema result structure...")
            if len(schema_result) > 0:
                first_row = schema_result[0]
                print(f"First row keys: {list(first_row.keys())}")
                print(f"First row values: {first_row}")
            
            print(f"\nExtracting column names...")
            available_columns = []
            
            for i, row in enumerate(schema_result):
                # Try different possible column name keys
                col_name = None
                for possible_key in ['col_name', 'column_name', 'field', 'Column', 'column']:
                    if possible_key in row:
                        col_name = row[possible_key]
                        break
                
                if col_name:
                    if not col_name.startswith('#') and col_name.strip():
                        available_columns.append(col_name.strip())
                        if i < 10:  # Show first 10 for debugging
                            print(f"  Row {i}: {col_name}")
                else:
                    if i < 5:  # Show first 5 problematic rows
                        print(f"  Row {i} (no column name found): {row}")
            
            print_success(f"Extracted {len(available_columns)} valid columns")
            
            if available_columns:
                print(f"\nFirst 20 Available Columns:")
                for i, col in enumerate(available_columns[:20]):
                    print(f"  {i+1:2d}. {col}")
                
                if len(available_columns) > 20:
                    print(f"  ... and {len(available_columns) - 20} more columns")
            else:
                print_error("No columns extracted from schema!")
                print("Raw schema data (first 3 rows):")
                for i, row in enumerate(schema_result[:3]):
                    print(f"  Row {i}: {row}")
                return False
            
            # Check which of our expected columns exist
            expected_columns = dashboard_generator.columns
            print(f"\nColumn Mapping Verification:")
            missing_columns = []
            found_columns = []
            
            for key, expected_col in expected_columns.items():
                # Check exact match
                if expected_col in available_columns:
                    found_columns.append((key, expected_col))
                    print(f"  FOUND {key} -> '{expected_col}'")
                else:
                    # Check case-insensitive match
                    case_match = None
                    for col in available_columns:
                        if col.lower() == expected_col.lower():
                            case_match = col
                            break
                    
                    if case_match:
                        found_columns.append((key, case_match))
                        print(f"  FOUND (case diff) {key} -> '{case_match}' (expected '{expected_col}')")
                    else:
                        missing_columns.append((key, expected_col))
                        print(f"  MISSING {key} -> '{expected_col}'")
            
            print(f"\nSummary:")
            print(f"  • Total columns in database: {len(available_columns)}")
            print(f"  • Expected columns found: {len(found_columns)}")
            print(f"  • Missing columns: {len(missing_columns)}")
            
            if missing_columns:
                print(f"\nMissing columns may cause query failures:")
                for key, col_name in missing_columns:
                    print(f"    {key}: '{col_name}'")
                
                print(f"\nColumn name suggestions:")
                print("Update the column mapping in bi_dashboard_generator.py with these corrections:")
                print("self.columns = {")
                for key, expected_col in expected_columns.items():
                    # Find best match
                    best_match = expected_col
                    for col in available_columns:
                        if col.lower() == expected_col.lower():
                            best_match = col
                            break
                        elif expected_col.lower().replace(' ', '_') == col.lower().replace(' ', '_'):
                            best_match = col
                            break
                    
                    if best_match != expected_col:
                        print(f"    '{key}': '{best_match}',  # Was '{expected_col}'")
                    else:
                        # Find partial matches
                        partial_matches = [col for col in available_columns if 
                                         expected_col.lower() in col.lower() or 
                                         col.lower() in expected_col.lower()]
                        if partial_matches:
                            print(f"    '{key}': '{partial_matches[0]}',  # Was '{expected_col}', similar: {partial_matches[:3]}")
                        else:
                            print(f"    '{key}': '{expected_col}',  # NOT FOUND")
                print("}")
            else:
                print(f"\nAll expected columns found!")
            
            # Test a few sample queries with found columns
            if found_columns:
                print(f"\nTesting sample queries with found columns...")
                test_individual_queries()
            
            return True
        else:
            print_error("Cannot describe table schema")
            return False
            
    except Exception as e:
        print_error(f"Debug failed: {e}")
        logger.exception("Debug error")
        return False


def clean_dashboard():
    """Clean dashboard files and data"""
    print_header("Cleaning dashboard files")
    
    try:
        items_to_clean = [
            'dashboard_data',
            'reports',
            'logs'
        ]
        
        cleaned_count = 0
        for item in items_to_clean:
            if os.path.exists(item):
                if os.path.isdir(item):
                    shutil.rmtree(item)
                    print(f"  Removed directory: {item}")
                else:
                    os.remove(item)
                    print(f"  Removed file: {item}")
                cleaned_count += 1
            else:
                print(f"  Not found: {item}")
        
        if cleaned_count > 0:
            print_success(f"Cleaned {cleaned_count} items")
        else:
            print_info("Nothing to clean")
        
        return True
        
    except Exception as e:
        print_error(f"Cleaning failed: {e}")
        return False


def show_help():
    """Show help information"""
    print_header("Dashboard commands help")
    
    commands = [
        ('setup', 'Set up dashboard infrastructure'),
        ('generate', 'Generate complete dashboard data'),
        ('overview', 'Generate overview metrics only'),
        ('revenue', 'Generate revenue trends only'),
        ('categories', 'Generate category analysis only'),
        ('market', 'Generate market insights only'),
        ('view', 'View dashboard data in terminal'),
        ('html', 'Generate HTML dashboard'),
        ('streamlit', 'Run interactive Streamlit dashboard'),
        ('report', 'Generate daily business report'),
        ('status', 'Check system status'),
        ('debug', 'Debug table schema and column availability'),
        ('clean', 'Clean dashboard files'),
        ('help', 'Show this help message')
    ]
    
    print("AVAILABLE COMMANDS:")
    for cmd, desc in commands:
        print(f"  {cmd:<12} - {desc}")
    
    print(f"\nUSAGE:")
    print(f"  python3 scripts/run_dashboard.py <command>")
    
    print(f"\nEXAMPLE WORKFLOW:")
    print(f"  1. python3 scripts/run_dashboard.py setup")
    print(f"  2. python3 scripts/run_dashboard.py debug    # Verify database connectivity")
    print(f"  3. python3 scripts/run_dashboard.py generate")
    print(f"  4. python3 scripts/run_dashboard.py view")
    print(f"  5. python3 scripts/run_dashboard.py streamlit")
    
    print(f"\nTROUBLESHOOTING:")
    print(f"  python3 scripts/run_dashboard.py debug    # Check database & columns")
    print(f"  python3 scripts/run_dashboard.py status   # Check system health")
    print(f"  python3 scripts/run_dashboard.py clean    # Clean and restart")
    
    print(f"\nCOMPONENT COMMANDS:")
    print(f"  python3 scripts/run_dashboard.py overview   # Generate overview only")
    print(f"  python3 scripts/run_dashboard.py revenue    # Generate revenue trends only") 
    print(f"  python3 scripts/run_dashboard.py categories # Generate category analysis only")
    print(f"  python3 scripts/run_dashboard.py market     # Generate market insights only")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        show_help()
        return 1
    
    command = sys.argv[1].lower()
    
    # Command mapping
    commands = {
        'setup': setup_dashboard,
        'generate': lambda: generate_dashboard_data(),
        'overview': lambda: generate_dashboard_data('overview'),
        'revenue': lambda: generate_dashboard_data('revenue_trends'),
        'categories': lambda: generate_dashboard_data('category_analysis'),
        'market': lambda: generate_dashboard_data('market_insights'),
        'view': view_dashboard_data,
        'html': generate_html_dashboard,
        'streamlit': run_streamlit_dashboard,
        'report': generate_daily_report,
        'status': check_system_status,
        'debug': debug_table_schema,
        'clean': clean_dashboard,
        'help': show_help
    }
    
    if command in commands:
        try:
            start_time = time.time()
            success = commands[command]()
            elapsed = time.time() - start_time
            
            print(f"\nOperation completed in {elapsed:.2f} seconds")
            return 0 if success is not False else 1
            
        except KeyboardInterrupt:
            print("\n\nOperation interrupted by user")
            return 1
        except Exception as e:
            print_error(f"Command failed: {e}")
            logger.exception("Command execution failed")
            return 1
    else:
        print_error(f"Unknown command: {command}")
        print("\nRun 'python3 scripts/run_dashboard.py help' for available commands")
        return 1


if __name__ == "__main__":
    exit(main())