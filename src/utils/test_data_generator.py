#!/usr/bin/env python3
"""
Test Data Generator for Chinese Produce Market Forecasting
Creates comprehensive test data with all required features in the correct format

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import os


class TestDataGenerator:
    """Generate comprehensive test data for endpoint testing with proper structure"""
    
    def __init__(self):
        """Initialize test data generator"""
        self.item_codes = ['ITEM001', 'ITEM002', 'ITEM003', 'ITEM004', 'ITEM005']
        self.categories = {
            'ITEM001': {'code': 'CAT001', 'name': 'Vegetables'},
            'ITEM002': {'code': 'CAT001', 'name': 'Vegetables'},
            'ITEM003': {'code': 'CAT002', 'name': 'Fruits'},
            'ITEM004': {'code': 'CAT002', 'name': 'Fruits'},
            'ITEM005': {'code': 'CAT003', 'name': 'Herbs'}
        }
    
    def generate_features_data(self, scenario: str = "default") -> dict:
        """Generate features data in the exact format required"""
        
        # Base date for testing
        test_date = datetime.now()
        
        if scenario == "default":
            # Your exact structure
            features = {
                "Total_Quantity": 150.0,
                "Avg_Price": 18.5,
                "Transaction_Count": 25,
                "Price_Volatility": 1.2,
                "Min_Price": 16.0,
                "Max_Price": 21.0,
                "Discount_Count": 3,
                "Revenue": 2775.0,
                "Discount_Rate": 0.12,
                "Price_Range": 5.0,
                "Wholesale Price (RMB/kg)": 14.0,
                "Loss Rate (%)": 8.5,
                "Month": 7,
                "DayOfWeek": 1,
                "IsWeekend": 0,
                "Year": 2024,
                "Quarter": 3,
                "DayOfYear": 202,
                "WeekOfYear": 29
            }
        
        elif scenario == "high_volume":
            # High volume sales scenario
            total_quantity = 500.0
            avg_price = 22.0
            transaction_count = 75
            wholesale_price = avg_price * 0.7
            
            features = {
                "Total_Quantity": total_quantity,
                "Avg_Price": avg_price,
                "Transaction_Count": transaction_count,
                "Price_Volatility": 2.1,
                "Min_Price": avg_price * 0.85,
                "Max_Price": avg_price * 1.15,
                "Discount_Count": 8,
                "Revenue": total_quantity * avg_price,
                "Discount_Rate": 8 / transaction_count,
                "Price_Range": (avg_price * 1.15) - (avg_price * 0.85),
                "Wholesale Price (RMB/kg)": wholesale_price,
                "Loss Rate (%)": 12.3,
                "Month": 12,  # December - holiday season
                "DayOfWeek": 5,  # Saturday
                "IsWeekend": 1,
                "Year": 2024,
                "Quarter": 4,
                "DayOfYear": 365,
                "WeekOfYear": 52
            }
        
        elif scenario == "low_volume":
            # Low volume sales scenario
            total_quantity = 25.0
            avg_price = 12.0
            transaction_count = 5
            wholesale_price = avg_price * 0.7
            
            features = {
                "Total_Quantity": total_quantity,
                "Avg_Price": avg_price,
                "Transaction_Count": transaction_count,
                "Price_Volatility": 0.3,
                "Min_Price": avg_price * 0.9,
                "Max_Price": avg_price * 1.1,
                "Discount_Count": 1,
                "Revenue": total_quantity * avg_price,
                "Discount_Rate": 1 / transaction_count,
                "Price_Range": (avg_price * 1.1) - (avg_price * 0.9),
                "Wholesale Price (RMB/kg)": wholesale_price,
                "Loss Rate (%)": 5.2,
                "Month": 2,  # February - low season
                "DayOfWeek": 1,  # Tuesday
                "IsWeekend": 0,
                "Year": 2024,
                "Quarter": 1,
                "DayOfYear": 45,
                "WeekOfYear": 7
            }
        
        elif scenario == "high_volatility":
            # High price volatility scenario
            total_quantity = 200.0
            avg_price = 25.0
            transaction_count = 40
            wholesale_price = avg_price * 0.6  # Lower wholesale margin due to volatility
            
            features = {
                "Total_Quantity": total_quantity,
                "Avg_Price": avg_price,
                "Transaction_Count": transaction_count,
                "Price_Volatility": 4.5,  # High volatility
                "Min_Price": avg_price * 0.7,  # Wide price range
                "Max_Price": avg_price * 1.4,
                "Discount_Count": 15,
                "Revenue": total_quantity * avg_price,
                "Discount_Rate": 15 / transaction_count,
                "Price_Range": (avg_price * 1.4) - (avg_price * 0.7),
                "Wholesale Price (RMB/kg)": wholesale_price,
                "Loss Rate (%)": 18.7,  # Higher loss due to volatility
                "Month": 6,  # June - summer volatility
                "DayOfWeek": 3,  # Thursday
                "IsWeekend": 0,
                "Year": 2024,
                "Quarter": 2,
                "DayOfYear": 165,
                "WeekOfYear": 24
            }
        
        elif scenario == "holiday_peak":
            # Holiday peak scenario
            total_quantity = 800.0
            avg_price = 35.0
            transaction_count = 120
            wholesale_price = avg_price * 0.65  # Holiday markup
            
            features = {
                "Total_Quantity": total_quantity,
                "Avg_Price": avg_price,
                "Transaction_Count": transaction_count,
                "Price_Volatility": 1.8,
                "Min_Price": avg_price * 0.9,
                "Max_Price": avg_price * 1.2,
                "Discount_Count": 5,  # Fewer discounts during peak
                "Revenue": total_quantity * avg_price,
                "Discount_Rate": 5 / transaction_count,
                "Price_Range": (avg_price * 1.2) - (avg_price * 0.9),
                "Wholesale Price (RMB/kg)": wholesale_price,
                "Loss Rate (%)": 15.0,
                "Month": 10,  # October - National Day
                "DayOfWeek": 0,  # Monday
                "IsWeekend": 0,
                "Year": 2024,
                "Quarter": 4,
                "DayOfYear": 275,
                "WeekOfYear": 40
            }
        
        elif scenario == "spring_fresh":
            # Spring fresh produce scenario
            total_quantity = 320.0
            avg_price = 16.8
            transaction_count = 55
            wholesale_price = avg_price * 0.75
            
            features = {
                "Total_Quantity": total_quantity,
                "Avg_Price": avg_price,
                "Transaction_Count": transaction_count,
                "Price_Volatility": 0.9,
                "Min_Price": avg_price * 0.88,
                "Max_Price": avg_price * 1.12,
                "Discount_Count": 7,
                "Revenue": total_quantity * avg_price,
                "Discount_Rate": 7 / transaction_count,
                "Price_Range": (avg_price * 1.12) - (avg_price * 0.88),
                "Wholesale Price (RMB/kg)": wholesale_price,
                "Loss Rate (%)": 6.8,  # Lower loss in spring
                "Month": 4,  # April - spring
                "DayOfWeek": 6,  # Sunday
                "IsWeekend": 1,
                "Year": 2024,
                "Quarter": 2,
                "DayOfYear": 105,
                "WeekOfYear": 15
            }
        
        else:
            # Random scenario
            total_quantity = round(np.random.uniform(50, 400), 1)
            avg_price = round(np.random.uniform(10, 30), 1)
            transaction_count = int(np.random.uniform(5, 80))
            wholesale_price = round(avg_price * np.random.uniform(0.6, 0.8), 1)
            discount_count = int(np.random.uniform(0, transaction_count * 0.4))
            
            features = {
                "Total_Quantity": total_quantity,
                "Avg_Price": avg_price,
                "Transaction_Count": transaction_count,
                "Price_Volatility": round(np.random.uniform(0.2, 3.5), 1),
                "Min_Price": round(avg_price * np.random.uniform(0.8, 0.95), 1),
                "Max_Price": round(avg_price * np.random.uniform(1.05, 1.3), 1),
                "Discount_Count": discount_count,
                "Revenue": round(total_quantity * avg_price, 1),
                "Discount_Rate": round(discount_count / transaction_count, 3) if transaction_count > 0 else 0.0,
                "Price_Range": round((avg_price * np.random.uniform(1.05, 1.3)) - (avg_price * np.random.uniform(0.8, 0.95)), 1),
                "Wholesale Price (RMB/kg)": wholesale_price,
                "Loss Rate (%)": round(np.random.uniform(3, 25), 1),
                "Month": np.random.randint(1, 13),
                "DayOfWeek": np.random.randint(0, 7),
                "IsWeekend": 1 if np.random.randint(0, 7) >= 5 else 0,
                "Year": 2024,
                "Quarter": np.random.randint(1, 5),
                "DayOfYear": np.random.randint(1, 366),
                "WeekOfYear": np.random.randint(1, 53)
            }
        
        # Round all float values to 1 decimal place for consistency
        for key, value in features.items():
            if isinstance(value, float):
                features[key] = round(value, 1)
        
        return {"features": features}
    
    def generate_batch_features_data(self, num_instances: int = 5) -> dict:
        """Generate batch test data with multiple instances"""
        scenarios = ["default", "high_volume", "low_volume", "high_volatility", "holiday_peak", "spring_fresh"]
        
        instances = []
        for i in range(num_instances):
            scenario = scenarios[i % len(scenarios)] if i < len(scenarios) else "random"
            single_data = self.generate_features_data(scenario)
            instances.append(single_data["features"])
        
        return {"instances": instances}
    
    def save_test_data(self, output_dir: str = "data/example"):
        """Save all test data variants to files in the specified format"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Test scenarios to generate
        scenarios = {
            "basic_test.json": "default",
            "high_volume_test.json": "high_volume",
            "low_volume_test.json": "low_volume",
            "high_volatility_test.json": "high_volatility",
            "holiday_peak_test.json": "holiday_peak",
            "spring_fresh_test.json": "spring_fresh"
        }
        
        # Generate individual scenario tests
        for filename, scenario in scenarios.items():
            data = self.generate_features_data(scenario)
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Generated {filename}")
        
        # Generate batch test data
        batch_data = self.generate_batch_features_data(10)
        batch_filepath = os.path.join(output_dir, "batch_test.json")
        with open(batch_filepath, 'w') as f:
            json.dump(batch_data, f, indent=2)
        print(f"✅ Generated batch_test.json")
        
        # Generate random test cases
        for i in range(5):
            random_data = self.generate_features_data("random")
            random_filepath = os.path.join(output_dir, f"random_test_{i+1}.json")
            with open(random_filepath, 'w') as f:
                json.dump(random_data, f, indent=2)
            print(f"✅ Generated random_test_{i+1}.json")
        
        # Create test summary and usage guide
        summary = {
            "test_data_generated": datetime.now().isoformat(),
            "output_directory": output_dir,
            "files_created": {
                "basic_test.json": {
                    "description": "Default test case matching your exact specification",
                    "use_case": "Basic endpoint verification"
                },
                "high_volume_test.json": {
                    "description": "High volume sales scenario (500+ units)",
                    "use_case": "Test handling of large transactions"
                },
                "low_volume_test.json": {
                    "description": "Low volume sales scenario (25 units)",
                    "use_case": "Test handling of small transactions"
                },
                "high_volatility_test.json": {
                    "description": "High price volatility scenario",
                    "use_case": "Test model behavior with volatile pricing"
                },
                "holiday_peak_test.json": {
                    "description": "Holiday peak demand scenario",
                    "use_case": "Test seasonal peak predictions"
                },
                "spring_fresh_test.json": {
                    "description": "Spring fresh produce scenario",
                    "use_case": "Test seasonal fresh produce patterns"
                },
                "batch_test.json": {
                    "description": "Batch prediction with multiple instances",
                    "use_case": "Test batch processing capabilities"
                },
                "random_test_*.json": {
                    "description": "Random test scenarios for robustness testing",
                    "use_case": "Test model robustness with varied inputs"
                }
            },
            "data_structure": {
                "format": "{ \"features\": { ... } }",
                "required_fields": [
                    "Total_Quantity", "Avg_Price", "Transaction_Count", "Price_Volatility",
                    "Min_Price", "Max_Price", "Discount_Count", "Revenue", "Discount_Rate",
                    "Price_Range", "Wholesale Price (RMB/kg)", "Loss Rate (%)", "Month",
                    "DayOfWeek", "IsWeekend", "Year", "Quarter", "DayOfYear", "WeekOfYear"
                ]
            },
            "usage_instructions": {
                "sagemaker_test_command": [
                    "python src/deployment/sagemaker_deploy.py --config config.yaml --action test --endpoint-name YOUR_ENDPOINT --test-data data/example/basic_test.json",
                    "python src/deployment/sagemaker_deploy.py --config config.yaml --action test --endpoint-name YOUR_ENDPOINT --test-data data/example/high_volume_test.json"
                ],
                "curl_example": "curl -X POST https://your-endpoint.amazonaws.com/invocations -H 'Content-Type: application/json' -d @data/example/basic_test.json"
            }
        }
        
        summary_filepath = os.path.join(output_dir, "test_summary.json")
        with open(summary_filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Generated test_summary.json")
        
        print(f"\n🎉 Test data generation completed!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Files generated: {len(os.listdir(output_dir))}")
        print(f"\n🧪 Quick test command:")
        print(f"python src/deployment/sagemaker_deploy.py --config config.yaml --action test --endpoint-name YOUR_ENDPOINT --test-data {output_dir}/basic_test.json")
        
        return output_dir


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate test data for Chinese Produce Forecasting')
    parser.add_argument('--output-dir', default='data/example', 
                       help='Output directory for test files (default: data/example)')
    parser.add_argument('--scenario', 
                       choices=['default', 'high_volume', 'low_volume', 'high_volatility', 
                               'holiday_peak', 'spring_fresh', 'random', 'all'],
                       default='all', help='Type of test scenario to generate')
    parser.add_argument('--batch-size', type=int, default=10, 
                       help='Number of instances for batch testing')
    
    args = parser.parse_args()
    
    generator = TestDataGenerator()
    
    if args.scenario == 'all':
        # Generate all test scenarios
        generator.save_test_data(args.output_dir)
    else:
        # Generate specific scenario
        os.makedirs(args.output_dir, exist_ok=True)
        
        if args.scenario == 'batch':
            data = generator.generate_batch_features_data(args.batch_size)
            filename = 'batch_test.json'
        else:
            data = generator.generate_features_data(args.scenario)
            filename = f'{args.scenario}_test.json'
        
        filepath = os.path.join(args.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Generated {filename} in {args.output_dir}")
        print(f"🧪 Test with: python src/deployment/sagemaker_deploy.py --config config.yaml --action test --endpoint-name YOUR_ENDPOINT --test-data {filepath}")


if __name__ == "__main__":
    main()