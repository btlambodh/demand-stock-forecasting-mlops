#!/usr/bin/env python3
"""
Dashboard Management Script - Updated Version
Handles all dashboard operations with correct column names

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


def setup_dashboard():
    """Set up dashboard infrastructure"""
    print_header("Setting up dashboard infrastructure")
    
    try:
        # Create necessary directories
        directories = [
            'dashboard_data',
            'reports', 
            'src/dashboard'
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
            return False
        
        print_success("Dashboard infrastructure setup completed!")
        return True
        
    except Exception as e:
        print_error(f"Setup failed: {e}")
        return False


def generate_dashboard_data(component=None):
    """Generate dashboard data using the fixed BI generator"""
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
        
        # Build command
        cmd = [
            sys.executable, generator_path,
            '--config', 'config.yaml',
            '--output', 'dashboard_data'
        ]
        
        if component:
            cmd.extend(['--component', component])
        
        print(f"Executing: {' '.join(cmd)}")
        
        # Run the generator
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print_success("Dashboard data generated successfully!")
            print("Generator output:")
            print(result.stdout)
            
            # Verify output files
            verify_dashboard_data()
            return True
        else:
            print_error("Dashboard generation failed!")
            print("Error output:")
            print(result.stderr)
            if result.stdout:
                print("Standard output:")
                print(result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print_error("Dashboard generation timed out (5 minutes)")
        return False
    except Exception as e:
        print_error(f"Dashboard generation failed: {e}")
        return False


def verify_dashboard_data():
    """Verify dashboard data files exist and contain data"""
    print("\nVerifying dashboard data files...")
    
    data_file = 'dashboard_data/dashboard_data.json'
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            print(f"  Main data file exists")
            print(f"  Data components: {len(data)}")
            
            # Check individual components
            components = ['overview', 'revenue_trends', 'category_analysis', 'price_analysis', 'market_insights']
            for component in components:
                if component in data:
                    if isinstance(data[component], dict):
                        sub_count = len(data[component])
                        print(f"    {component}: {sub_count} datasets")
                    else:
                        print(f"    {component}: Available")
                else:
                    print(f"    {component}: Missing")
            
            return True
            
        except Exception as e:
            print_error(f"Error reading dashboard data: {e}")
            return False
    else:
        print_error("Dashboard data file not found")
        return False


def view_dashboard_data():
    """View dashboard data in terminal"""
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
                print(f"  {component_name.replace('_', ' ').title()}: {sub_components} datasets")
            else:
                print(f"  {component_name.replace('_', ' ').title()}: Available")
        
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
        print("\n📋 REPORT SUMMARY:")
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
            print(f"  {name}: EXISTS ({size:,} bytes)")
        else:
            print(f"  {name}: NOT FOUND ({path})")
            all_files_ok = False
    
    # Check directories
    print("\nDIRECTORY STATUS:")
    directories = ['dashboard_data', 'reports', 'src/dashboard']
    for directory in directories:
        if os.path.exists(directory):
            file_count = len(os.listdir(directory))
            print(f"  {directory}: EXISTS ({file_count} files)")
        else:
            print(f"  {directory}: NOT FOUND")
    
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
            print(f"  {name}: Available")
        except ImportError:
            print(f"  {name}: Missing")
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
                        print(f"  {component}: {count} datasets")
                    else:
                        print(f"  {component}: Available")
                else:
                    print(f"  {component}: Missing")
            
        except Exception as e:
            print(f"  Data file corrupted: {e}")
            all_files_ok = False
    
    # Overall status
    print(f"\nOVERALL STATUS:")
    if all_files_ok:
        print("  Dashboard system is ready!")
    else:
        print("  Dashboard system has issues - run 'setup' and 'generate'")
    
    return all_files_ok


def clean_dashboard():
    """Clean dashboard files and data"""
    print_header("Cleaning dashboard files")
    
    try:
        items_to_clean = [
            'dashboard_data',
            'reports'
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
            print("Nothing to clean")
        
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
    print(f"  2. python3 scripts/run_dashboard.py generate")
    print(f"  3. python3 scripts/run_dashboard.py view")
    print(f"  4. python3 scripts/run_dashboard.py html")
    print(f"  5. python3 scripts/run_dashboard.py streamlit")
    
    print(f"\nCOMPONENT COMMANDS:")
    print(f"  python3 scripts/run_dashboard.py overview")
    print(f"  python3 scripts/run_dashboard.py revenue") 
    print(f"  python3 scripts/run_dashboard.py categories")
    print(f"  python3 scripts/run_dashboard.py market")


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