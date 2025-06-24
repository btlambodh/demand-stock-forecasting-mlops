#!/usr/bin/env python3
"""
Business Intelligence Dashboard Data Generator
Updated with correct column names from SageMaker deployment

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
    """Business Intelligence Dashboard Data Generator with correct column names from SageMaker"""
    
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
        
        # Column names from SageMaker deployment feature order
        self.columns = {
            'date': 'Date',
            'total_quantity': 'Total_Quantity',
            'avg_price': 'Avg_Price', 
            'transaction_count': 'Transaction_Count',
            'price_volatility': 'Price_Volatility',
            'min_price': 'Min_Price',
            'max_price': 'Max_Price',
            'discount_count': 'Discount_Count',
            'revenue': 'Revenue',
            'discount_rate': 'Discount_Rate',
            'price_range': 'Price_Range',
            'wholesale_price': 'Wholesale Price (RMB/kg)',
            'loss_rate': 'Loss Rate (%)',
            'year': 'Year',
            'month': 'Month',
            'quarter': 'Quarter',
            'day_of_year': 'DayOfYear',
            'day_of_week': 'DayOfWeek',
            'week_of_year': 'WeekOfYear',
            'is_weekend': 'IsWeekend',
            'is_national_day': 'IsNationalDay',
            'is_labor_day': 'IsLaborDay',
            'category_code': 'Category Code',
            'category_name_encoded': 'Category Name_Encoded',
            'season_spring': 'Season_Spring',
            'season_summer': 'Season_Summer', 
            'season_winter': 'Season_Winter'
        }

    def execute_query(self, query: str, query_name: str) -> Optional[List[Dict]]:
        """Execute Athena query and return results"""
        try:
            logger.info(f"Executing query: {query_name}")
            
            # Submit query
            response = self.athena_client.start_query_execution(
                QueryString=query,
                QueryExecutionContext={'Database': self.database_name},
                ResultConfiguration={'OutputLocation': self.query_results_location},
                WorkGroup=self.workgroup
            )
            
            query_execution_id = response['QueryExecutionId']
            
            # Wait for completion
            success, result = self.wait_for_query_completion(query_execution_id)
            
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
                                if value and value.replace('.', '').replace('-', '').isdigit():
                                    try:
                                        if '.' in value:
                                            record[headers[i]] = float(value)
                                        else:
                                            record[headers[i]] = int(value)
                                    except:
                                        record[headers[i]] = value
                                else:
                                    record[headers[i]] = value
                        data.append(record)
                    
                    logger.info(f"SUCCESS {query_name}: Retrieved {len(data)} records")
                    return data
                else:
                    logger.warning(f"WARNING {query_name}: No data returned (empty result set)")
                    return []
            else:
                logger.error(f"ERROR {query_name}: Query failed - {result}")
                # Print the actual query for debugging
                logger.error(f"Failed query was: {query}")
                return None
                
        except Exception as e:
            logger.error(f"ERROR {query_name}: Exception - {e}")
            logger.error(f"Failed query was: {query}")
            return None

    def wait_for_query_completion(self, query_execution_id: str, timeout: int = 60):
        """Wait for Athena query to complete"""
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

    def get_overview_metrics(self):
        """Get high-level overview metrics with correct column names"""
        logger.info("Generating overview metrics...")
        
        queries = {
            'total_records': f"""
                SELECT COUNT(*) as total_records 
                FROM {self.database_name}.features_complete
            """,
            'unique_items': f"""
                SELECT COUNT(DISTINCT "{self.columns['category_code']}") as unique_items 
                FROM {self.database_name}.features_complete
                WHERE "{self.columns['category_code']}" IS NOT NULL
            """,
            'date_range': f"""
                SELECT 
                    MIN("{self.columns['date']}") as earliest_date, 
                    MAX("{self.columns['date']}") as latest_date 
                FROM {self.database_name}.features_complete
            """,
            'total_revenue': f"""
                SELECT 
                    SUM("{self.columns['revenue']}") as total_revenue,
                    AVG("{self.columns['revenue']}") as avg_revenue,
                    COUNT(*) as transaction_count
                FROM {self.database_name}.features_complete 
                WHERE "{self.columns['revenue']}" IS NOT NULL
            """,
            'unique_categories': f"""
                SELECT COUNT(DISTINCT "{self.columns['category_name_encoded']}") as unique_categories
                FROM {self.database_name}.features_complete 
                WHERE "{self.columns['category_name_encoded']}" IS NOT NULL
            """
        }
        
        overview = {}
        for metric_name, query in queries.items():
            result = self.execute_query(query, f"overview_{metric_name}")
            if result and len(result) > 0:
                overview[metric_name] = result[0]
        
        self.dashboard_data['overview'] = overview

    def get_revenue_trends(self):
        """Get revenue trends over time with correct column names"""
        logger.info("Generating revenue trends...")
        
        queries = {
            'monthly_revenue': f"""
                SELECT 
                    "{self.columns['year']}" as year,
                    "{self.columns['month']}" as month,
                    SUM("{self.columns['revenue']}") as monthly_revenue,
                    AVG("{self.columns['avg_price']}") as avg_price,
                    SUM("{self.columns['total_quantity']}") as total_quantity,
                    COUNT(*) as transaction_count
                FROM {self.database_name}.features_complete
                WHERE "{self.columns['revenue']}" IS NOT NULL
                GROUP BY "{self.columns['year']}", "{self.columns['month']}"
                ORDER BY "{self.columns['year']}", "{self.columns['month']}"
            """,
            'daily_revenue_trend': f"""
                SELECT 
                    "{self.columns['year']}" as year,
                    "{self.columns['month']}" as month,
                    "{self.columns['day_of_year']}" as day_of_year,
                    SUM("{self.columns['revenue']}") as daily_revenue,
                    COUNT(*) as daily_transactions,
                    AVG("{self.columns['avg_price']}") as avg_daily_price
                FROM {self.database_name}.features_complete
                WHERE "{self.columns['revenue']}" IS NOT NULL 
                  AND "{self.columns['year']}" = 2023
                  AND "{self.columns['month']}" >= 3
                GROUP BY "{self.columns['year']}", "{self.columns['month']}", "{self.columns['day_of_year']}"
                ORDER BY "{self.columns['year']}", "{self.columns['month']}", "{self.columns['day_of_year']}"
            """,
            'seasonal_revenue': f"""
                SELECT 
                    CASE 
                        WHEN "{self.columns['month']}" IN (3, 4, 5) THEN 'Spring'
                        WHEN "{self.columns['month']}" IN (6, 7, 8) THEN 'Summer'
                        WHEN "{self.columns['month']}" IN (9, 10, 11) THEN 'Autumn'
                        ELSE 'Winter'
                    END as Season,
                    SUM("{self.columns['revenue']}") as seasonal_revenue,
                    AVG("{self.columns['avg_price']}") as avg_seasonal_price,
                    COUNT(*) as seasonal_transactions
                FROM {self.database_name}.features_complete
                WHERE "{self.columns['revenue']}" IS NOT NULL
                  AND "{self.columns['month']}" IS NOT NULL
                GROUP BY 1
                ORDER BY seasonal_revenue DESC
            """
        }
        
        revenue_trends = {}
        for trend_name, query in queries.items():
            logger.info(f"Executing revenue trend query: {trend_name}")
            result = self.execute_query(query, f"revenue_{trend_name}")
            if result is not None:
                revenue_trends[trend_name] = result
                logger.info(f"SUCCESS {trend_name}: Retrieved {len(result)} records")
            else:
                logger.error(f"FAILED {trend_name}: Query returned None")
        
        if revenue_trends:
            self.dashboard_data['revenue_trends'] = revenue_trends
            logger.info(f"Revenue trends component completed with {len(revenue_trends)} datasets")
        else:
            logger.error("No revenue trends data generated - all queries failed")
            self.dashboard_data['revenue_trends'] = {}

    def get_category_analysis(self):
        """Get category performance analysis with correct column names"""
        logger.info("Generating category analysis...")
        
        queries = {
            'top_categories_by_revenue': f"""
                SELECT 
                    "{self.columns['category_name_encoded']}" as Category_Name,
                    SUM("{self.columns['revenue']}") as total_revenue,
                    AVG("{self.columns['avg_price']}") as avg_price,
                    SUM("{self.columns['total_quantity']}") as total_quantity,
                    COUNT(*) as transaction_count,
                    AVG("{self.columns['price_volatility']}") as avg_volatility
                FROM {self.database_name}.features_complete
                WHERE "{self.columns['category_name_encoded']}" IS NOT NULL AND "{self.columns['revenue']}" IS NOT NULL
                GROUP BY "{self.columns['category_name_encoded']}"
                ORDER BY total_revenue DESC
                LIMIT 15
            """,
            'category_growth': f"""
                SELECT 
                    "{self.columns['category_name_encoded']}" as Category_Name,
                    "{self.columns['year']}" as year,
                    SUM("{self.columns['revenue']}") as yearly_revenue,
                    COUNT(*) as yearly_transactions
                FROM {self.database_name}.features_complete
                WHERE "{self.columns['category_name_encoded']}" IS NOT NULL AND "{self.columns['revenue']}" IS NOT NULL
                GROUP BY "{self.columns['category_name_encoded']}", "{self.columns['year']}"
                ORDER BY "{self.columns['category_name_encoded']}", "{self.columns['year']}"
            """,
            'category_seasonality': f"""
                SELECT 
                    "{self.columns['category_name_encoded']}" as Category_Name,
                    CASE 
                        WHEN "{self.columns['season_spring']}" = 1 THEN 'Spring'
                        WHEN "{self.columns['season_summer']}" = 1 THEN 'Summer'
                        WHEN "{self.columns['season_winter']}" = 1 THEN 'Winter'
                        ELSE 'Autumn'
                    END as Season,
                    SUM("{self.columns['revenue']}") as seasonal_revenue,
                    AVG("{self.columns['avg_price']}") as avg_seasonal_price
                FROM {self.database_name}.features_complete
                WHERE "{self.columns['category_name_encoded']}" IS NOT NULL 
                  AND "{self.columns['revenue']}" IS NOT NULL
                GROUP BY "{self.columns['category_name_encoded']}", 2
                ORDER BY "{self.columns['category_name_encoded']}", seasonal_revenue DESC
            """
        }
        
        category_analysis = {}
        for analysis_name, query in queries.items():
            logger.info(f"Executing category analysis query: {analysis_name}")
            result = self.execute_query(query, f"category_{analysis_name}")
            if result is not None:
                category_analysis[analysis_name] = result
                logger.info(f"SUCCESS {analysis_name}: Retrieved {len(result)} records")
            else:
                logger.error(f"FAILED {analysis_name}: Query returned None")
        
        if category_analysis:
            self.dashboard_data['category_analysis'] = category_analysis
            logger.info(f"Category analysis component completed with {len(category_analysis)} datasets")
        else:
            logger.error("No category analysis data generated - all queries failed")
            self.dashboard_data['category_analysis'] = {}

    def get_price_analysis(self):
        """Get price analysis and trends with correct column names"""
        logger.info("Generating price analysis...")
        
        queries = {
            'price_distribution': f"""
                SELECT 
                    CASE 
                        WHEN "{self.columns['avg_price']}" < 5 THEN 'Low (< 5 RMB)'
                        WHEN "{self.columns['avg_price']}" < 15 THEN 'Medium (5-15 RMB)'
                        WHEN "{self.columns['avg_price']}" < 30 THEN 'High (15-30 RMB)'
                        ELSE 'Premium (> 30 RMB)'
                    END as price_range,
                    COUNT(*) as transaction_count,
                    SUM("{self.columns['revenue']}") as range_revenue,
                    AVG("{self.columns['avg_price']}") as avg_price_in_range
                FROM {self.database_name}.features_complete
                WHERE "{self.columns['avg_price']}" IS NOT NULL AND "{self.columns['revenue']}" IS NOT NULL
                GROUP BY 1
                ORDER BY avg_price_in_range
            """,
            'most_volatile_items': f"""
                SELECT 
                    "{self.columns['category_code']}" as Item_Code,
                    "{self.columns['category_name_encoded']}" as Category_Name,
                    AVG("{self.columns['price_volatility']}") as avg_volatility,
                    AVG("{self.columns['avg_price']}") as avg_price,
                    COUNT(*) as data_points
                FROM {self.database_name}.features_complete
                WHERE "{self.columns['price_volatility']}" IS NOT NULL 
                  AND "{self.columns['category_name_encoded']}" IS NOT NULL
                GROUP BY "{self.columns['category_code']}", "{self.columns['category_name_encoded']}"
                ORDER BY avg_volatility DESC
                LIMIT 20
            """,
            'price_trends_by_month': f"""
                SELECT 
                    "{self.columns['year']}" as year,
                    "{self.columns['month']}" as month,
                    AVG("{self.columns['avg_price']}") as avg_monthly_price,
                    AVG("{self.columns['price_volatility']}") as avg_monthly_volatility,
                    MIN("{self.columns['avg_price']}") as min_monthly_price,
                    MAX("{self.columns['avg_price']}") as max_monthly_price
                FROM {self.database_name}.features_complete
                WHERE "{self.columns['avg_price']}" IS NOT NULL
                GROUP BY "{self.columns['year']}", "{self.columns['month']}"
                ORDER BY "{self.columns['year']}", "{self.columns['month']}"
            """
        }
        
        price_analysis = {}
        for analysis_name, query in queries.items():
            logger.info(f"Executing price analysis query: {analysis_name}")
            result = self.execute_query(query, f"price_{analysis_name}")
            if result is not None:
                price_analysis[analysis_name] = result
                logger.info(f"SUCCESS {analysis_name}: Retrieved {len(result)} records")
            else:
                logger.error(f"FAILED {analysis_name}: Query returned None")
        
        if price_analysis:
            self.dashboard_data['price_analysis'] = price_analysis
            logger.info(f"Price analysis component completed with {len(price_analysis)} datasets")
        else:
            logger.error("No price analysis data generated - all queries failed")
            self.dashboard_data['price_analysis'] = {}

    def get_market_insights(self):
        """Get market insights and patterns with correct column names"""
        logger.info("Generating market insights...")
        
        queries = {
            'top_performing_items': f"""
                SELECT 
                    "{self.columns['category_code']}" as Item_Code,
                    "{self.columns['category_name_encoded']}" as Category_Name,
                    SUM("{self.columns['revenue']}") as total_revenue,
                    AVG("{self.columns['avg_price']}") as avg_price,
                    SUM("{self.columns['total_quantity']}") as total_quantity,
                    COUNT(*) as trading_days
                FROM {self.database_name}.features_complete
                WHERE "{self.columns['revenue']}" IS NOT NULL AND "{self.columns['category_name_encoded']}" IS NOT NULL
                GROUP BY "{self.columns['category_code']}", "{self.columns['category_name_encoded']}"
                ORDER BY total_revenue DESC
                LIMIT 25
            """,
            'weekend_vs_weekday': f"""
                SELECT 
                    CASE WHEN "{self.columns['is_weekend']}" = 1 THEN 'Weekend' ELSE 'Weekday' END as day_type,
                    COUNT(*) as transaction_count,
                    SUM("{self.columns['revenue']}") as total_revenue,
                    AVG("{self.columns['avg_price']}") as avg_price,
                    AVG("{self.columns['total_quantity']}") as avg_quantity
                FROM {self.database_name}.features_complete
                WHERE "{self.columns['revenue']}" IS NOT NULL AND "{self.columns['is_weekend']}" IS NOT NULL
                GROUP BY "{self.columns['is_weekend']}"
            """,
            'holiday_impact': f"""
                SELECT 
                    CASE 
                        WHEN "{self.columns['is_national_day']}" = 1 THEN 'National Day'
                        WHEN "{self.columns['is_labor_day']}" = 1 THEN 'Labor Day'
                        ELSE 'Regular Day'
                    END as day_type,
                    COUNT(*) as transaction_count,
                    AVG("{self.columns['avg_price']}") as avg_price,
                    SUM("{self.columns['revenue']}") as total_revenue
                FROM {self.database_name}.features_complete
                WHERE "{self.columns['revenue']}" IS NOT NULL
                GROUP BY 1
                ORDER BY total_revenue DESC
            """,
            'loss_rate_impact': f"""
                SELECT 
                    CASE 
                        WHEN "{self.columns['loss_rate']}" < 5 THEN 'Low Loss (< 5%)'
                        WHEN "{self.columns['loss_rate']}" < 15 THEN 'Medium Loss (5-15%)'
                        WHEN "{self.columns['loss_rate']}" < 25 THEN 'High Loss (15-25%)'
                        ELSE 'Very High Loss (> 25%)'
                    END as loss_category,
                    COUNT(*) as item_count,
                    AVG("{self.columns['avg_price']}") as avg_price,
                    SUM("{self.columns['revenue']}") as total_revenue,
                    AVG("{self.columns['loss_rate']}") as avg_loss_rate
                FROM {self.database_name}.features_complete
                WHERE "{self.columns['loss_rate']}" IS NOT NULL AND "{self.columns['revenue']}" IS NOT NULL
                GROUP BY 1
                ORDER BY avg_loss_rate
            """
        }
        
        market_insights = {}
        for insight_name, query in queries.items():
            logger.info(f"Executing market insight query: {insight_name}")
            result = self.execute_query(query, f"insights_{insight_name}")
            if result is not None:
                market_insights[insight_name] = result
                logger.info(f"SUCCESS {insight_name}: Retrieved {len(result)} records")
            else:
                logger.error(f"FAILED {insight_name}: Query returned None")
        
        if market_insights:
            self.dashboard_data['market_insights'] = market_insights
            logger.info(f"Market insights component completed with {len(market_insights)} datasets")
        else:
            logger.error("No market insights data generated - all queries failed")
            self.dashboard_data['market_insights'] = {}

    def get_forecasting_features(self):
        """Get features relevant for forecasting models with correct column names"""
        logger.info("Generating forecasting features...")
        
        queries = {
            'feature_correlations': f"""
                SELECT 
                    "{self.columns['category_name_encoded']}" as Category_Name,
                    AVG("{self.columns['price_volatility']}") as avg_volatility,
                    AVG("{self.columns['avg_price']}") as avg_price,
                    COUNT(*) as data_points
                FROM {self.database_name}.features_complete
                WHERE "{self.columns['category_name_encoded']}" IS NOT NULL 
                  AND "{self.columns['year']}" = 2023
                GROUP BY "{self.columns['category_name_encoded']}"
                ORDER BY avg_volatility DESC
            """,
            'recent_trends': f"""
                SELECT 
                    "{self.columns['year']}" as year,
                    "{self.columns['month']}" as month,
                    COUNT(*) as monthly_transactions,
                    AVG("{self.columns['avg_price']}") as avg_monthly_price,
                    SUM("{self.columns['total_quantity']}") as monthly_quantity,
                    AVG("{self.columns['price_volatility']}") as monthly_volatility
                FROM {self.database_name}.features_complete
                WHERE "{self.columns['year']}" = 2023
                GROUP BY "{self.columns['year']}", "{self.columns['month']}"
                ORDER BY "{self.columns['year']}", "{self.columns['month']}" DESC
            """,
            'model_performance_indicators': f"""
                SELECT 
                    "{self.columns['category_name_encoded']}" as Category_Name,
                    AVG("{self.columns['price_range']}") as avg_price_range,
                    AVG("{self.columns['discount_rate']}") as avg_discount_rate,
                    COUNT(DISTINCT "{self.columns['category_code']}") as unique_items,
                    STDDEV("{self.columns['avg_price']}") as price_std
                FROM {self.database_name}.features_complete
                WHERE "{self.columns['category_name_encoded']}" IS NOT NULL
                GROUP BY "{self.columns['category_name_encoded']}"
                ORDER BY price_std DESC
            """
        }
        
        forecasting_features = {}
        for feature_name, query in queries.items():
            logger.info(f"Executing forecasting feature query: {feature_name}")
            result = self.execute_query(query, f"forecasting_{feature_name}")
            if result is not None:
                forecasting_features[feature_name] = result
                logger.info(f"SUCCESS {feature_name}: Retrieved {len(result)} records")
            else:
                logger.error(f"FAILED {feature_name}: Query returned None")
        
        if forecasting_features:
            self.dashboard_data['forecasting_features'] = forecasting_features
            logger.info(f"Forecasting features component completed with {len(forecasting_features)} datasets")
        else:
            logger.error("No forecasting features data generated - all queries failed")
            self.dashboard_data['forecasting_features'] = {}

    def generate_dashboard_summary(self):
        """Generate executive dashboard summary"""
        logger.info("Generating dashboard summary...")
        
        summary = {
            'generated_at': datetime.now().isoformat(),
            'data_period': {},
            'key_metrics': {},
            'top_insights': []
        }
        
        # Extract key metrics from overview
        if 'overview' in self.dashboard_data:
            overview = self.dashboard_data['overview']
            if 'total_revenue' in overview:
                summary['key_metrics'].update(overview['total_revenue'])
            if 'total_records' in overview:
                summary['key_metrics'].update(overview['total_records'])
            if 'date_range' in overview:
                summary['data_period'] = overview['date_range']
        
        # Top insights from category analysis
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
        
        # Load existing data if it exists (for component-specific generation)
        existing_data = {}
        json_file = os.path.join(output_path, 'dashboard_data.json')
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    existing_data = json.load(f)
                logger.info("Loaded existing dashboard data for merging")
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
        
        # Save CSV files for easy consumption
        for component, data in merged_data.items():
            if isinstance(data, dict):
                for sub_component, sub_data in data.items():
                    if isinstance(sub_data, list) and len(sub_data) > 0:
                        try:
                            df = pd.DataFrame(sub_data)
                            csv_file = os.path.join(output_path, f'{component}_{sub_component}.csv')
                            df.to_csv(csv_file, index=False)
                        except Exception as e:
                            logger.warning(f"Could not save CSV for {component}_{sub_component}: {e}")
        
        logger.info(f"SUCCESS Dashboard data saved to {output_path}")
        
        # Print summary of what was saved
        if merged_data:
            logger.info("Data components saved:")
            for component in merged_data.keys():
                if isinstance(merged_data[component], dict):
                    count = len(merged_data[component])
                    logger.info(f"  SUCCESS {component}: {count} datasets")
                else:
                    logger.info(f"  SUCCESS {component}: Available")
        else:
            logger.warning("No data was generated to save")

    def generate_complete_dashboard(self, output_path: str = 'dashboard_data'):
        """Generate complete dashboard data"""
        logger.info("Starting Business Intelligence Dashboard Data Generation")
        logger.info("=" * 70)
        
        try:
            # Generate all dashboard components with progress tracking
            logger.info("STEP 1/7: Generating overview metrics...")
            self.get_overview_metrics()
            
            logger.info("STEP 2/7: Generating revenue trends...")
            self.get_revenue_trends()
            
            logger.info("STEP 3/7: Generating category analysis...")
            self.get_category_analysis()
            
            logger.info("STEP 4/7: Generating price analysis...")
            self.get_price_analysis()
            
            logger.info("STEP 5/7: Generating market insights...")
            self.get_market_insights()
            
            logger.info("STEP 6/7: Generating forecasting features...")
            self.get_forecasting_features()
            
            logger.info("STEP 7/7: Generating dashboard summary...")
            self.generate_dashboard_summary()
            
            # Save data
            logger.info("Saving all dashboard data...")
            self.save_dashboard_data(output_path)
            
            # Print summary
            self.print_dashboard_summary()
            
            logger.info("=" * 70)
            logger.info("SUCCESS Dashboard data generation completed successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR Dashboard generation failed: {e}")
            return False

    def print_dashboard_summary(self):
        """Print dashboard summary to console"""
        print("\n" + "=" * 70)
        print("BUSINESS INTELLIGENCE DASHBOARD SUMMARY")
        print("=" * 70)
        
        if 'summary' in self.dashboard_data:
            summary = self.dashboard_data['summary']
            
            print(f"Data Period: {summary.get('data_period', {}).get('earliest_date', 'N/A')} to {summary.get('data_period', {}).get('latest_date', 'N/A')}")
            print(f"Total Revenue: {summary.get('key_metrics', {}).get('total_revenue', 'N/A'):,.2f} RMB")
            print(f"Total Transactions: {summary.get('key_metrics', {}).get('transaction_count', 'N/A'):,}")
            print(f"Average Revenue: {summary.get('key_metrics', {}).get('avg_revenue', 'N/A'):.2f} RMB")
            
            print(f"\nTop Performing Categories:")
            for insight in summary.get('top_insights', []):
                print(f"   {insight['rank']}. {insight['category']}: {insight['revenue']:,.2f} RMB ({insight['transactions']:,} transactions)")
        
        print(f"\nData Components Generated:")
        
        # Expected vs actual datasets
        expected_datasets = {
            'overview': 5,
            'revenue_trends': 3,  # monthly_revenue, daily_revenue_trend, seasonal_revenue
            'category_analysis': 3,  # top_categories_by_revenue, category_growth, category_seasonality
            'price_analysis': 3,  # price_distribution, most_volatile_items, price_trends_by_month
            'market_insights': 4,  # top_performing_items, weekend_vs_weekday, holiday_impact, loss_rate_impact
            'forecasting_features': 3  # feature_correlations, recent_trends, model_performance_indicators
        }
        
        for component in expected_datasets.keys():
            expected = expected_datasets[component]
            if component in self.dashboard_data:
                if isinstance(self.dashboard_data[component], dict):
                    actual = len(self.dashboard_data[component])
                    status = "SUCCESS" if actual == expected else f"PARTIAL ({actual}/{expected})"
                    print(f"   {status} {component.replace('_', ' ').title()}: {actual} datasets")
                    
                    # Show which specific datasets are missing
                    if actual < expected and isinstance(self.dashboard_data[component], dict):
                        available = list(self.dashboard_data[component].keys())
                        print(f"     Available: {', '.join(available)}")
                else:
                    print(f"   SUCCESS {component.replace('_', ' ').title()}: Available")
            else:
                print(f"   MISSING {component.replace('_', ' ').title()}: 0/{expected} datasets")


def main():
    """Main function for dashboard generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Business Intelligence Dashboard Data Generator')
    parser.add_argument('--config', default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--output', default='dashboard_data', help='Output directory for dashboard data')
    parser.add_argument('--component', help='Generate specific component only (overview, revenue_trends, etc.)')
    
    args = parser.parse_args()
    
    # Initialize dashboard generator
    dashboard_generator = BIDashboardGenerator(args.config)
    
    if args.component:
        # Generate specific component
        logger.info(f"Generating specific component: {args.component}")
        
        # Map component names to method names
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
            logger.info(f"SUCCESS {args.component} component generated successfully!")
        else:
            available_components = ', '.join(component_methods.keys())
            logger.error(f"Unknown component: {args.component}")
            logger.error(f"Available components: {available_components}")
            return 1
    else:
        # Generate complete dashboard
        success = dashboard_generator.generate_complete_dashboard(args.output)
        return 0 if success else 1


if __name__ == "__main__":
    exit(main())