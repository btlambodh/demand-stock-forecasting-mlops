#!/usr/bin/env python3
"""
Business Intelligence Dashboard Data Generator - Schema Corrected Version
Updated with correct Athena tab-separated schema parsing and actual column names

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import boto3
import yaml
import json
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('bi_dashboard')


class BIDashboardGenerator:
    """BI Dashboard Generator with correct Athena schema parsing"""
    
    def __init__(self, config_path: str):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # AWS clients
        self.athena_client = boto3.client('athena', region_name=self.config['aws']['region'])
        self.s3_client = boto3.client('s3', region_name=self.config['aws']['region'])
        
        # Configuration
        self.database_name = self.config['aws']['athena']['database_name']
        self.query_results_location = self.config['aws']['athena']['query_results_location']
        self.workgroup = self.config['aws']['athena'].get('workgroup', 'primary')
        
        # Dashboard data storage
        self.dashboard_data = {}
        
        # CORRECTED column names based on actual Athena output (lowercase with underscores)
        self.columns = {
            # Basic transaction data
            'item_code': 'item code',
            'total_quantity': 'total_quantity',
            'avg_quantity': 'avg_quantity', 
            'transaction_count': 'transaction_count',
            'avg_price': 'avg_price',
            'price_volatility': 'price_volatility',
            'min_price': 'min_price',
            'max_price': 'max_price',
            'discount_count': 'discount_count',
            'revenue': 'revenue',
            'discount_rate': 'discount_rate',
            'price_range': 'price_range',
            
            # External data  
            'wholesale_price': 'wholesale price (rmb/kg)',
            'loss_rate': 'loss rate (%)',
            
            # Temporal features
            'year': 'year',
            'month': 'month',
            'quarter': 'quarter',
            'day_of_year': 'dayofyear',
            'day_of_week': 'dayofweek',
            'week_of_year': 'weekofyear',
            'is_weekend': 'isweekend',
            
            # Cyclical features
            'month_sin': 'month_sin',
            'month_cos': 'month_cos',
            'day_of_year_sin': 'dayofyear_sin',
            'day_of_year_cos': 'dayofyear_cos',
            'day_of_week_sin': 'dayofweek_sin',
            'day_of_week_cos': 'dayofweek_cos',
            
            # Holiday features
            'is_national_day': 'isnationalday',
            'is_labor_day': 'islaborday',
            'days_since_epoch': 'days_since_epoch',
            
            # Price features
            'retail_wholesale_ratio': 'retail_wholesale_ratio',
            'price_markup': 'price_markup',
            'price_markup_pct': 'price_markup_pct',
            'avg_price_change': 'avg_price_change',
            'wholesale_price_change': 'wholesale_price_change',
            
            # Category features
            'category_code': 'category code',
            'category_total_quantity': 'category_total_quantity',
            'category_avg_price': 'category_avg_price',
            'category_revenue': 'category_revenue',
            'item_quantity_share': 'item_quantity_share',
            'item_revenue_share': 'item_revenue_share',
            'price_relative_to_category': 'price_relative_to_category',
            'category_name_encoded': 'category name_encoded',
            
            # Loss features
            'effective_supply': 'effective_supply',
            'loss_adjusted_revenue': 'loss_adjusted_revenue',
            
            # Interaction features
            'price_quantity_interaction': 'price_quantity_interaction',
            'price_volatility_quantity': 'price_volatility_quantity',
            'spring_price': 'spring_price',
            'summer_price': 'summer_price',
            'autumn_price': 'autumn_price',
            'winter_price': 'winter_price',
            'holiday_demand': 'holiday_demand',
            
            # Season dummies
            'season_spring': 'season_spring',
            'season_summer': 'season_summer',
            'season_winter': 'season_winter',
            
            # Loss rate categories
            'loss_rate_category_medium': 'loss_rate_category_medium',
            'loss_rate_category_high': 'loss_rate_category_high',
            'loss_rate_category_very_high': 'loss_rate_category_very_high'
        }
        
        # Available columns cache
        self.available_columns = None
        self.validated_columns = {}

    def get_available_columns(self) -> List[str]:
        """Get list of available columns with correct Athena tab-separated parsing"""
        if self.available_columns is None:
            try:
                schema_query = f"DESCRIBE {self.database_name}.features_complete"
                schema_result = self.execute_query(schema_query, "get_schema", timeout=30)
                
                if schema_result:
                    self.available_columns = []
                    logger.info(f"Processing {len(schema_result)} schema rows...")
                    
                    for i, row in enumerate(schema_result):
                        # The schema format is: "column_name \t data_type \t comment"
                        # All in a single VarCharValue field
                        
                        raw_value = None
                        for key in ['VarCharValue', 'LongValue', 'DoubleValue', 'BooleanValue']:
                            if key in row and row[key] is not None:
                                raw_value = str(row[key])
                                break
                        
                        if raw_value:
                            # Split by tabs and get the first part (column name)
                            parts = raw_value.split('\t')
                            if len(parts) > 0:
                                column_name = parts[0].strip()
                                
                                # Filter out empty names and metadata
                                if column_name and not column_name.startswith('#'):
                                    self.available_columns.append(column_name)
                                    if len(self.available_columns) <= 10:  # Log first 10 for debugging
                                        logger.info(f"  Column {len(self.available_columns)}: '{column_name}'")
                    
                    logger.info(f"Extracted {len(self.available_columns)} valid columns from database")
                    
                    # Log summary
                    if self.available_columns:
                        logger.info(f"First 10 columns: {self.available_columns[:10]}")
                        if len(self.available_columns) > 10:
                            logger.info(f"Last 5 columns: {self.available_columns[-5:]}")
                    else:
                        logger.error("No columns extracted after tab-separated parsing")
                else:
                    logger.error("Failed to retrieve table schema")
                    self.available_columns = []
            except Exception as e:
                logger.error(f"Error getting available columns: {e}")
                self.available_columns = []
        
        return self.available_columns

    def validate_column(self, column_key: str) -> Optional[str]:
        """Validate column exists and return correct name"""
        if column_key in self.validated_columns:
            return self.validated_columns[column_key]
        
        available_cols = self.get_available_columns()
        expected_col = self.columns.get(column_key)
        
        if not expected_col:
            logger.warning(f"No expected column defined for key: {column_key}")
            self.validated_columns[column_key] = None
            return None
        
        # Check if column exists in available columns
        for col in available_cols:
            # Direct match
            if col == expected_col:
                self.validated_columns[column_key] = col
                return col
            
            # Case-insensitive match
            if col.lower() == expected_col.lower():
                logger.info(f"Case-insensitive match for {column_key}: '{col}' (expected '{expected_col}')")
                self.validated_columns[column_key] = col
                return col
        
        logger.warning(f"Column '{expected_col}' for key '{column_key}' not found in database")
        self.validated_columns[column_key] = None
        return None

    def execute_query(self, query: str, query_name: str, timeout: int = 120) -> Optional[List[Dict]]:
        """Execute Athena query with enhanced error handling"""
        try:
            logger.info(f"Executing query: {query_name}")
            logger.debug(f"Query SQL: {query}")
            
            # Submit query
            response = self.athena_client.start_query_execution(
                QueryString=query,
                QueryExecutionContext={'Database': self.database_name},
                ResultConfiguration={'OutputLocation': self.query_results_location},
                WorkGroup=self.workgroup
            )
            
            query_execution_id = response['QueryExecutionId']
            
            # Wait for completion
            success, result = self.wait_for_query_completion(query_execution_id, timeout)
            
            if success and result:
                # Parse results
                rows = result['ResultSet']['Rows']
                if len(rows) > 1:  # Skip header row
                    headers = [col['VarCharValue'] for col in rows[0]['Data']]
                    data = []
                    
                    for row in rows[1:]:
                        record = {}
                        for i, cell in enumerate(row['Data']):
                            if i < len(headers):
                                value = cell.get('VarCharValue', None)
                                # Try to convert numeric values
                                if value and value.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit():
                                    try:
                                        if '.' in value or 'e' in value.lower():
                                            record[headers[i]] = float(value)
                                        else:
                                            record[headers[i]] = int(value)
                                    except ValueError:
                                        record[headers[i]] = value
                                else:
                                    record[headers[i]] = value
                        data.append(record)
                    
                    logger.info(f"Query {query_name}: Retrieved {len(data)} records")
                    return data
                else:
                    logger.warning(f"Query {query_name}: No data returned")
                    return []
            else:
                logger.error(f"Query {query_name}: Failed - {result}")
                return None
                
        except Exception as e:
            logger.error(f"Query {query_name}: Exception - {e}")
            return None

    def wait_for_query_completion(self, query_execution_id: str, timeout: int = 120):
        """Wait for Athena query completion"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.athena_client.get_query_execution(QueryExecutionId=query_execution_id)
                status = response['QueryExecution']['Status']['State']
                
                if status == 'SUCCEEDED':
                    results = self.athena_client.get_query_results(QueryExecutionId=query_execution_id)
                    return True, results
                elif status in ['FAILED', 'CANCELLED']:
                    error_reason = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                    return False, error_reason
                
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error checking query status: {e}")
                return False, str(e)
        
        return False, f"Query timeout after {timeout} seconds"

    def get_overview_metrics(self):
        """Generate overview metrics"""
        logger.info("Generating overview metrics...")
        
        revenue_col = self.validate_column('revenue')
        category_code_col = self.validate_column('category_code')
        item_code_col = self.validate_column('item_code')
        year_col = self.validate_column('year')
        month_col = self.validate_column('month')
        
        overview = {}
        
        # Basic record count
        count_query = f"SELECT COUNT(*) as total_records FROM {self.database_name}.features_complete"
        result = self.execute_query(count_query, "total_records")
        if result and len(result) > 0:
            overview['total_records'] = result[0]
        
        # Revenue metrics
        if revenue_col:
            revenue_query = f"""
                SELECT 
                    SUM({revenue_col}) as total_revenue,
                    AVG({revenue_col}) as avg_revenue,
                    COUNT(*) as transaction_count
                FROM {self.database_name}.features_complete 
                WHERE {revenue_col} IS NOT NULL AND {revenue_col} > 0
            """
            result = self.execute_query(revenue_query, "total_revenue")
            if result and len(result) > 0:
                overview['total_revenue'] = result[0]
        
        # Date range
        if year_col and month_col:
            date_query = f"""
                SELECT 
                    MIN(CAST({year_col} AS VARCHAR) || '-' || 
                        LPAD(CAST({month_col} AS VARCHAR), 2, '0') || '-01') as earliest_date,
                    MAX(CAST({year_col} AS VARCHAR) || '-' || 
                        LPAD(CAST({month_col} AS VARCHAR), 2, '0') || '-28') as latest_date
                FROM {self.database_name}.features_complete
                WHERE {year_col} IS NOT NULL AND {month_col} IS NOT NULL
            """
            result = self.execute_query(date_query, "date_range")
            if result and len(result) > 0:
                overview['date_range'] = result[0]
        
        # Unique items
        if item_code_col:
            unique_query = f"""
                SELECT COUNT(DISTINCT {item_code_col}) as unique_items 
                FROM {self.database_name}.features_complete
                WHERE {item_code_col} IS NOT NULL
            """
            result = self.execute_query(unique_query, "unique_items")
            if result and len(result) > 0:
                overview['unique_items'] = result[0]
        
        self.dashboard_data['overview'] = overview
        logger.info(f"Overview metrics completed: {len(overview)} metrics generated")

    def get_revenue_trends(self):
        """Generate revenue trends"""
        logger.info("Generating revenue trends...")
        
        revenue_col = self.validate_column('revenue')
        year_col = self.validate_column('year')
        month_col = self.validate_column('month')
        avg_price_col = self.validate_column('avg_price')
        total_quantity_col = self.validate_column('total_quantity')
        
        if not revenue_col:
            logger.error("Revenue column not found")
            self.dashboard_data['revenue_trends'] = {}
            return
        
        revenue_trends = {}
        
        # Monthly revenue trend
        if year_col and month_col:
            monthly_query = f"""
                SELECT 
                    {year_col} as year,
                    {month_col} as month,
                    SUM({revenue_col}) as monthly_revenue,
                    COUNT(*) as transaction_count
            """
            
            if avg_price_col:
                monthly_query += f",\n                    AVG({avg_price_col}) as avg_price"
            
            monthly_query += f"""
                FROM {self.database_name}.features_complete
                WHERE {revenue_col} IS NOT NULL AND {revenue_col} > 0
                    AND {year_col} IS NOT NULL AND {month_col} IS NOT NULL
                GROUP BY {year_col}, {month_col}
                ORDER BY {year_col}, {month_col}
            """
            
            result = self.execute_query(monthly_query, "monthly_revenue")
            if result is not None:
                revenue_trends['monthly_revenue'] = result
                logger.info(f"Monthly revenue: {len(result)} months")
        
        # Seasonal analysis
        if month_col:
            seasonal_query = f"""
                SELECT 
                    CASE 
                        WHEN {month_col} IN (3, 4, 5) THEN 'Spring'
                        WHEN {month_col} IN (6, 7, 8) THEN 'Summer'
                        WHEN {month_col} IN (9, 10, 11) THEN 'Autumn'
                        ELSE 'Winter'
                    END as Season,
                    SUM({revenue_col}) as seasonal_revenue,
                    COUNT(*) as seasonal_transactions
                FROM {self.database_name}.features_complete
                WHERE {revenue_col} IS NOT NULL AND {revenue_col} > 0
                    AND {month_col} IS NOT NULL
                GROUP BY 1
                ORDER BY seasonal_revenue DESC
            """
            
            result = self.execute_query(seasonal_query, "seasonal_revenue")
            if result is not None:
                revenue_trends['seasonal_revenue'] = result
                logger.info(f"Seasonal revenue: {len(result)} seasons")
        
        self.dashboard_data['revenue_trends'] = revenue_trends
        logger.info(f"Revenue trends completed: {len(revenue_trends)} datasets")

    def get_category_analysis(self):
        """Generate category analysis"""
        logger.info("Generating category analysis...")
        
        category_name_col = self.validate_column('category_name_encoded')
        category_code_col = self.validate_column('category_code')
        revenue_col = self.validate_column('revenue')
        avg_price_col = self.validate_column('avg_price')
        total_quantity_col = self.validate_column('total_quantity')
        year_col = self.validate_column('year')
        
        # Use category_code if category_name_encoded is not available
        category_col = category_name_col or category_code_col
        
        if not category_col or not revenue_col:
            logger.error("Required columns missing for category analysis")
            self.dashboard_data['category_analysis'] = {}
            return
        
        category_analysis = {}
        
        # Top categories by revenue
        top_categories_query = f"""
            SELECT 
                {category_col} as Category_Name,
                SUM({revenue_col}) as total_revenue,
                COUNT(*) as transaction_count
        """
        
        if avg_price_col:
            top_categories_query += f",\n                AVG({avg_price_col}) as avg_price"
        
        if total_quantity_col:
            top_categories_query += f",\n                SUM({total_quantity_col}) as total_quantity"
        
        top_categories_query += f"""
            FROM {self.database_name}.features_complete
            WHERE {category_col} IS NOT NULL 
                AND {revenue_col} IS NOT NULL AND {revenue_col} > 0
            GROUP BY {category_col}
            ORDER BY total_revenue DESC
            LIMIT 15
        """
        
        result = self.execute_query(top_categories_query, "top_categories_by_revenue")
        if result is not None:
            category_analysis['top_categories_by_revenue'] = result
            logger.info(f"Top categories: {len(result)} categories")
        
        self.dashboard_data['category_analysis'] = category_analysis
        logger.info(f"Category analysis completed: {len(category_analysis)} datasets")

    def get_price_analysis(self):
        """Generate price analysis"""
        logger.info("Generating price analysis...")
        
        avg_price_col = self.validate_column('avg_price')
        revenue_col = self.validate_column('revenue')
        price_volatility_col = self.validate_column('price_volatility')
        year_col = self.validate_column('year')
        month_col = self.validate_column('month')
        
        if not avg_price_col or not revenue_col:
            logger.error("Required columns missing for price analysis")
            self.dashboard_data['price_analysis'] = {}
            return
        
        price_analysis = {}
        
        # Price distribution
        price_dist_query = f"""
            SELECT 
                CASE 
                    WHEN {avg_price_col} < 5 THEN 'Low (< 5 RMB)'
                    WHEN {avg_price_col} < 15 THEN 'Medium (5-15 RMB)'
                    WHEN {avg_price_col} < 30 THEN 'High (15-30 RMB)'
                    ELSE 'Premium (> 30 RMB)'
                END as price_range,
                COUNT(*) as transaction_count,
                SUM({revenue_col}) as range_revenue,
                AVG({avg_price_col}) as avg_price_in_range
            FROM {self.database_name}.features_complete
            WHERE {avg_price_col} IS NOT NULL AND {revenue_col} IS NOT NULL
                AND {avg_price_col} > 0 AND {revenue_col} > 0
            GROUP BY 1
            ORDER BY avg_price_in_range
        """
        
        result = self.execute_query(price_dist_query, "price_distribution")
        if result is not None:
            price_analysis['price_distribution'] = result
            logger.info(f"Price distribution: {len(result)} ranges")
        
        self.dashboard_data['price_analysis'] = price_analysis
        logger.info(f"Price analysis completed: {len(price_analysis)} datasets")

    def get_market_insights(self):
        """Generate market insights"""
        logger.info("Generating market insights...")
        
        revenue_col = self.validate_column('revenue')
        avg_price_col = self.validate_column('avg_price')
        category_name_col = self.validate_column('category_name_encoded')
        category_code_col = self.validate_column('category_code')
        is_weekend_col = self.validate_column('is_weekend')
        
        if not revenue_col:
            logger.error("Revenue column missing for market insights")
            self.dashboard_data['market_insights'] = {}
            return
        
        market_insights = {}
        
        # Top performing items
        category_col = category_name_col or category_code_col
        if category_col:
            top_items_query = f"""
                SELECT 
                    {category_col} as Category_Name,
                    SUM({revenue_col}) as total_revenue,
                    COUNT(*) as trading_days
            """
            
            if avg_price_col:
                top_items_query += f",\n                    AVG({avg_price_col}) as avg_price"
            
            top_items_query += f"""
                FROM {self.database_name}.features_complete
                WHERE {revenue_col} IS NOT NULL AND {category_col} IS NOT NULL
                    AND {revenue_col} > 0
                GROUP BY {category_col}
                ORDER BY total_revenue DESC
                LIMIT 20
            """
            
            result = self.execute_query(top_items_query, "top_performing_items")
            if result is not None:
                market_insights['top_performing_items'] = result
                logger.info(f"Top performing items: {len(result)} items")
        
        # Weekend vs Weekday analysis
        if is_weekend_col:
            weekend_query = f"""
                SELECT 
                    CASE WHEN {is_weekend_col} = 1 THEN 'Weekend' ELSE 'Weekday' END as day_type,
                    COUNT(*) as transaction_count,
                    SUM({revenue_col}) as total_revenue
            """
            
            if avg_price_col:
                weekend_query += f",\n                    AVG({avg_price_col}) as avg_price"
            
            weekend_query += f"""
                FROM {self.database_name}.features_complete
                WHERE {revenue_col} IS NOT NULL AND {is_weekend_col} IS NOT NULL
                    AND {revenue_col} > 0
                GROUP BY {is_weekend_col}
            """
            
            result = self.execute_query(weekend_query, "weekend_vs_weekday")
            if result is not None:
                market_insights['weekend_vs_weekday'] = result
                logger.info(f"Weekend vs weekday: {len(result)} categories")
        
        self.dashboard_data['market_insights'] = market_insights
        logger.info(f"Market insights completed: {len(market_insights)} datasets")

    def get_forecasting_features(self):
        """Generate forecasting features"""
        logger.info("Generating forecasting features...")
        
        category_name_col = self.validate_column('category_name_encoded')
        category_code_col = self.validate_column('category_code')
        avg_price_col = self.validate_column('avg_price')
        year_col = self.validate_column('year')
        
        category_col = category_name_col or category_code_col
        
        if not category_col or not avg_price_col:
            logger.error("Required columns missing for forecasting features")
            self.dashboard_data['forecasting_features'] = {}
            return
        
        forecasting_features = {}
        
        # Feature correlations
        if year_col:
            correlations_query = f"""
                SELECT 
                    {category_col} as Category_Name,
                    AVG({avg_price_col}) as avg_price,
                    COUNT(*) as data_points
                FROM {self.database_name}.features_complete
                WHERE {category_col} IS NOT NULL AND {year_col} >= 2023
                    AND {avg_price_col} IS NOT NULL AND {avg_price_col} > 0
                GROUP BY {category_col}
                ORDER BY avg_price DESC
                LIMIT 30
            """
            
            result = self.execute_query(correlations_query, "feature_correlations")
            if result is not None:
                forecasting_features['feature_correlations'] = result
                logger.info(f"Feature correlations: {len(result)} categories")
        
        self.dashboard_data['forecasting_features'] = forecasting_features
        logger.info(f"Forecasting features completed: {len(forecasting_features)} datasets")

    def generate_dashboard_summary(self):
        """Generate dashboard summary"""
        logger.info("Generating dashboard summary...")
        
        summary = {
            'generated_at': datetime.now().isoformat(),
            'data_period': {},
            'key_metrics': {},
            'top_insights': []
        }
        
        # Extract key metrics
        if 'overview' in self.dashboard_data:
            overview = self.dashboard_data['overview']
            if 'total_revenue' in overview:
                summary['key_metrics'].update(overview['total_revenue'])
            if 'total_records' in overview:
                summary['key_metrics'].update(overview['total_records'])
            if 'date_range' in overview:
                summary['data_period'] = overview['date_range']
        
        # Top insights
        if 'category_analysis' in self.dashboard_data and 'top_categories_by_revenue' in self.dashboard_data['category_analysis']:
            top_categories = self.dashboard_data['category_analysis']['top_categories_by_revenue'][:3]
            for i, category in enumerate(top_categories, 1):
                summary['top_insights'].append({
                    'rank': i,
                    'category': category.get('Category_Name'),
                    'revenue': category.get('total_revenue'),
                    'transactions': category.get('transaction_count')
                })
        
        self.dashboard_data['summary'] = summary

    def save_dashboard_data(self, output_path: str):
        """Save dashboard data to files"""
        logger.info(f"Saving dashboard data to {output_path}")
        
        os.makedirs(output_path, exist_ok=True)
        
        # Load existing data if it exists
        existing_data = {}
        json_file = os.path.join(output_path, 'dashboard_data.json')
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    existing_data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing data: {e}")
        
        # Merge with existing data
        merged_data = existing_data.copy()
        merged_data.update(self.dashboard_data)
        
        # Save complete data as JSON
        with open(json_file, 'w') as f:
            json.dump(merged_data, f, indent=2, default=str)
        
        # Save individual components
        for component, data in merged_data.items():
            component_file = os.path.join(output_path, f'{component}.json')
            with open(component_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Dashboard data saved to {output_path}")

    def generate_complete_dashboard(self, output_path: str = 'dashboard_data'):
        """Generate complete dashboard"""
        logger.info("Starting BI Dashboard Data Generation")
        logger.info("=" * 70)
        
        try:
            # Validate connectivity
            logger.info("STEP 0/7: Validating database connectivity...")
            available_cols = self.get_available_columns()
            if not available_cols:
                logger.error("Cannot connect to database")
                return False
            
            logger.info(f"Database connected - {len(available_cols)} columns available")
            
            # Generate components
            components = [
                ("overview metrics", self.get_overview_metrics),
                ("revenue trends", self.get_revenue_trends),
                ("category analysis", self.get_category_analysis),
                ("price analysis", self.get_price_analysis),
                ("market insights", self.get_market_insights),
                ("forecasting features", self.get_forecasting_features),
                ("dashboard summary", self.generate_dashboard_summary)
            ]
            
            for i, (name, method) in enumerate(components, 1):
                logger.info(f"STEP {i}/7: Generating {name}...")
                try:
                    method()
                    logger.info(f"  {name} completed successfully")
                except Exception as e:
                    logger.error(f"  {name} failed: {e}")
            
            # Save data
            logger.info("Saving dashboard data...")
            self.save_dashboard_data(output_path)
            
            # Print summary
            self.print_dashboard_summary()
            
            logger.info("Dashboard generation completed!")
            return True
            
        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            return False

    def print_dashboard_summary(self):
        """Print dashboard summary"""
        print("\n" + "=" * 70)
        print("BUSINESS INTELLIGENCE DASHBOARD SUMMARY")
        print("=" * 70)
        
        if self.dashboard_data:
            print(f"Components Generated:")
            for component_name, component_data in self.dashboard_data.items():
                if isinstance(component_data, dict):
                    count = len(component_data)
                    print(f"  {component_name.replace('_', ' ').title()}: {count} datasets")
                else:
                    print(f"  {component_name.replace('_', ' ').title()}: Available")
        else:
            print("No dashboard data generated")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BI Dashboard Generator - Schema Corrected')
    parser.add_argument('--config', default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--output', default='dashboard_data', help='Output directory')
    parser.add_argument('--component', help='Generate specific component only')
    
    args = parser.parse_args()
    
    dashboard_generator = BIDashboardGenerator(args.config)
    
    if args.component:
        component_methods = {
            'overview': 'get_overview_metrics',
            'revenue_trends': 'get_revenue_trends',
            'category_analysis': 'get_category_analysis',
            'price_analysis': 'get_price_analysis',
            'market_insights': 'get_market_insights',
            'forecasting_features': 'get_forecasting_features'
        }
        
        if args.component in component_methods:
            method_name = component_methods[args.component]
            getattr(dashboard_generator, method_name)()
            dashboard_generator.save_dashboard_data(args.output)
            logger.info(f"{args.component} component generated successfully!")
        else:
            available = ', '.join(component_methods.keys())
            logger.error(f"Unknown component: {args.component}. Available: {available}")
            return 1
    else:
        success = dashboard_generator.generate_complete_dashboard(args.output)
        return 0 if success else 1


if __name__ == "__main__":
    exit(main())
