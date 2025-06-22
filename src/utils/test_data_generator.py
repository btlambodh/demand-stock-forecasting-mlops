#!/usr/bin/env python3
"""
Test Data Generator for Chinese Produce Market Forecasting
Creates comprehensive test data matching the exact format expected by SageMaker endpoints

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
        self.base_features = [
            "Total_Quantity", "Avg_Price", "Transaction_Count", "Price_Volatility",
            "Min_Price", "Max_Price", "Discount_Count", "Revenue", "Discount_Rate",
            "Price_Range", "Wholesale Price (RMB/kg)", "Loss Rate (%)", 
            "Month", "DayOfWeek", "IsWeekend", "Year", "Quarter", "DayOfYear", "WeekOfYear"
        ]
    
    def generate_basic_test_data(self) -> dict:
        """Generate basic test data matching your exact specification"""
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
        return {"features": features}
    
    def generate_scenario_data(self, scenario: str) -> dict:
        """Generate test data for specific scenarios"""
        
        if scenario == "high_volume":
            # High volume sales scenario - weekend peak
            total_quantity = 450.0
            avg_price = 22.3
            transaction_count = 68
            wholesale_price = 17.8
            discount_count = 12
            
            features = {
                "Total_Quantity": total_quantity,
                "Avg_Price": avg_price,
                "Transaction_Count": transaction_count,
                "Price_Volatility": 2.1,
                "Min_Price": 19.5,
                "Max_Price": 25.8,
                "Discount_Count": discount_count,
                "Revenue": round(total_quantity * avg_price, 1),
                "Discount_Rate": round(discount_count / transaction_count, 3),
                "Price_Range": 6.3,
                "Wholesale Price (RMB/kg)": wholesale_price,
                "Loss Rate (%)": 11.2,
                "Month": 12,  # December - holiday season
                "DayOfWeek": 6,  # Sunday
                "IsWeekend": 1,
                "Year": 2024,
                "Quarter": 4,
                "DayOfYear": 358,
                "WeekOfYear": 51
            }
        
        elif scenario == "low_volume":
            # Low volume sales scenario - midweek quiet period
            total_quantity = 35.0
            avg_price = 14.2
            transaction_count = 8
            wholesale_price = 11.1
            discount_count = 1
            
            features = {
                "Total_Quantity": total_quantity,
                "Avg_Price": avg_price,
                "Transaction_Count": transaction_count,
                "Price_Volatility": 0.4,
                "Min_Price": 13.2,
                "Max_Price": 15.1,
                "Discount_Count": discount_count,
                "Revenue": round(total_quantity * avg_price, 1),
                "Discount_Rate": round(discount_count / transaction_count, 3),
                "Price_Range": 1.9,
                "Wholesale Price (RMB/kg)": wholesale_price,
                "Loss Rate (%)": 4.8,
                "Month": 2,  # February - low season
                "DayOfWeek": 2,  # Wednesday
                "IsWeekend": 0,
                "Year": 2024,
                "Quarter": 1,
                "DayOfYear": 48,
                "WeekOfYear": 7
            }
        
        elif scenario == "high_volatility":
            # High price volatility scenario - supply chain disruption
            total_quantity = 180.0
            avg_price = 28.7
            transaction_count = 35
            wholesale_price = 16.2
            discount_count = 18
            
            features = {
                "Total_Quantity": total_quantity,
                "Avg_Price": avg_price,
                "Transaction_Count": transaction_count,
                "Price_Volatility": 5.3,  # Very high volatility
                "Min_Price": 21.5,
                "Max_Price": 36.9,
                "Discount_Count": discount_count,
                "Revenue": round(total_quantity * avg_price, 1),
                "Discount_Rate": round(discount_count / transaction_count, 3),
                "Price_Range": 15.4,
                "Wholesale Price (RMB/kg)": wholesale_price,
                "Loss Rate (%)": 22.1,  # High loss due to supply issues
                "Month": 3,  # March - seasonal transition
                "DayOfWeek": 4,  # Friday
                "IsWeekend": 0,
                "Year": 2024,
                "Quarter": 1,
                "DayOfYear": 78,
                "WeekOfYear": 11
            }
        
        elif scenario == "holiday_peak":
            # National Day holiday peak demand
            total_quantity = 620.0
            avg_price = 31.8
            transaction_count = 95
            wholesale_price = 24.2
            discount_count = 4  # Fewer discounts during peak demand
            
            features = {
                "Total_Quantity": total_quantity,
                "Avg_Price": avg_price,
                "Transaction_Count": transaction_count,
                "Price_Volatility": 1.9,
                "Min_Price": 28.5,
                "Max_Price": 34.7,
                "Discount_Count": discount_count,
                "Revenue": round(total_quantity * avg_price, 1),
                "Discount_Rate": round(discount_count / transaction_count, 3),
                "Price_Range": 6.2,
                "Wholesale Price (RMB/kg)": wholesale_price,
                "Loss Rate (%)": 16.8,
                "Month": 10,  # October - National Day
                "DayOfWeek": 0,  # Monday (holiday)
                "IsWeekend": 0,
                "Year": 2024,
                "Quarter": 4,
                "DayOfYear": 274,
                "WeekOfYear": 39
            }
        
        elif scenario == "spring_fresh":
            # Spring fresh produce season
            total_quantity = 285.0
            avg_price = 16.9
            transaction_count = 52
            wholesale_price = 12.8
            discount_count = 8
            
            features = {
                "Total_Quantity": total_quantity,
                "Avg_Price": avg_price,
                "Transaction_Count": transaction_count,
                "Price_Volatility": 0.8,
                "Min_Price": 15.2,
                "Max_Price": 18.6,
                "Discount_Count": discount_count,
                "Revenue": round(total_quantity * avg_price, 1),
                "Discount_Rate": round(discount_count / transaction_count, 3),
                "Price_Range": 3.4,
                "Wholesale Price (RMB/kg)": wholesale_price,
                "Loss Rate (%)": 6.3,  # Lower loss in spring
                "Month": 4,  # April - spring season
                "DayOfWeek": 5,  # Saturday
                "IsWeekend": 1,
                "Year": 2024,
                "Quarter": 2,
                "DayOfYear": 105,
                "WeekOfYear": 15
            }
        
        elif scenario == "summer_peak":
            # Summer peak demand
            total_quantity = 380.0
            avg_price = 24.6
            transaction_count = 71
            wholesale_price = 19.1
            discount_count = 9
            
            features = {
                "Total_Quantity": total_quantity,
                "Avg_Price": avg_price,
                "Transaction_Count": transaction_count,
                "Price_Volatility": 1.6,
                "Min_Price": 22.1,
                "Max_Price": 27.3,
                "Discount_Count": discount_count,
                "Revenue": round(total_quantity * avg_price, 1),
                "Discount_Rate": round(discount_count / transaction_count, 3),
                "Price_Range": 5.2,
                "Wholesale Price (RMB/kg)": wholesale_price,
                "Loss Rate (%)": 13.7,
                "Month": 8,  # August - peak summer
                "DayOfWeek": 6,  # Sunday
                "IsWeekend": 1,
                "Year": 2024,
                "Quarter": 3,
                "DayOfYear": 227,
                "WeekOfYear": 32
            }
        
        elif scenario == "weekend_shopping":
            # Weekend shopping pattern
            total_quantity = 195.0
            avg_price = 20.4
            transaction_count = 43
            wholesale_price = 15.8
            discount_count = 6
            
            features = {
                "Total_Quantity": total_quantity,
                "Avg_Price": avg_price,
                "Transaction_Count": transaction_count,
                "Price_Volatility": 1.1,
                "Min_Price": 18.7,
                "Max_Price": 22.9,
                "Discount_Count": discount_count,
                "Revenue": round(total_quantity * avg_price, 1),
                "Discount_Rate": round(discount_count / transaction_count, 3),
                "Price_Range": 4.2,
                "Wholesale Price (RMB/kg)": wholesale_price,
                "Loss Rate (%)": 9.4,
                "Month": 6,  # June
                "DayOfWeek": 5,  # Saturday
                "IsWeekend": 1,
                "Year": 2024,
                "Quarter": 2,
                "DayOfYear": 165,
                "WeekOfYear": 23
            }
        
        else:  # random scenario
            # Generate realistic random data
            total_quantity = round(np.random.uniform(50, 500), 1)
            avg_price = round(np.random.uniform(12, 35), 1)
            transaction_count = int(np.random.uniform(5, 85))
            wholesale_price = round(avg_price * np.random.uniform(0.65, 0.85), 1)
            discount_count = int(np.random.uniform(0, transaction_count * 0.3))
            
            month = np.random.randint(1, 13)
            day_of_week = np.random.randint(0, 7)
            day_of_year = np.random.randint(1, 366)
            
            features = {
                "Total_Quantity": total_quantity,
                "Avg_Price": avg_price,
                "Transaction_Count": transaction_count,
                "Price_Volatility": round(np.random.uniform(0.3, 4.0), 1),
                "Min_Price": round(avg_price * np.random.uniform(0.85, 0.95), 1),
                "Max_Price": round(avg_price * np.random.uniform(1.05, 1.25), 1),
                "Discount_Count": discount_count,
                "Revenue": round(total_quantity * avg_price, 1),
                "Discount_Rate": round(discount_count / transaction_count, 3) if transaction_count > 0 else 0.0,
                "Price_Range": round((avg_price * 1.15) - (avg_price * 0.9), 1),
                "Wholesale Price (RMB/kg)": wholesale_price,
                "Loss Rate (%)": round(np.random.uniform(4, 20), 1),
                "Month": month,
                "DayOfWeek": day_of_week,
                "IsWeekend": 1 if day_of_week >= 5 else 0,
                "Year": 2024,
                "Quarter": ((month - 1) // 3) + 1,
                "DayOfYear": day_of_year,
                "WeekOfYear": min(52, (day_of_year // 7) + 1)
            }
        
        return {"features": features}
    
    def generate_batch_data(self, num_instances: int = 10) -> dict:
        """Generate batch test data with multiple realistic scenarios"""
        scenarios = [
            "basic", "high_volume", "low_volume", "high_volatility", 
            "holiday_peak", "spring_fresh", "summer_peak", "weekend_shopping"
        ]
        
        instances = []
        
        # Always include basic scenario first
        basic_data = self.generate_basic_test_data()
        instances.append(basic_data["features"])
        
        # Add other scenarios
        for i in range(1, num_instances):
            if i - 1 < len(scenarios) - 1:
                scenario = scenarios[i]
                scenario_data = self.generate_scenario_data(scenario)
            else:
                scenario_data = self.generate_scenario_data("random")
            instances.append(scenario_data["features"])
        
        return {"instances": instances}
    
    def create_all_test_files(self, output_dir: str = "data/example"):
        """Create all test data files in the correct format"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Test file definitions
        test_files = {
            "basic_test.json": {
                "data": self.generate_basic_test_data(),
                "description": "Basic test case matching exact specification"
            },
            "high_volume_test.json": {
                "data": self.generate_scenario_data("high_volume"),
                "description": "High volume sales scenario (450+ units)"
            },
            "low_volume_test.json": {
                "data": self.generate_scenario_data("low_volume"),
                "description": "Low volume sales scenario (35 units)"
            },
            "high_volatility_test.json": {
                "data": self.generate_scenario_data("high_volatility"),
                "description": "High price volatility scenario"
            },
            "holiday_peak_test.json": {
                "data": self.generate_scenario_data("holiday_peak"),
                "description": "National Day holiday peak demand"
            },
            "spring_fresh_test.json": {
                "data": self.generate_scenario_data("spring_fresh"),
                "description": "Spring fresh produce season"
            },
            "summer_peak_test.json": {
                "data": self.generate_scenario_data("summer_peak"),
                "description": "Summer peak demand scenario"
            },
            "weekend_shopping_test.json": {
                "data": self.generate_scenario_data("weekend_shopping"),
                "description": "Weekend shopping pattern"
            },
            "batch_test.json": {
                "data": self.generate_batch_data(8),
                "description": "Batch prediction with 8 diverse scenarios"
            }
        }
        
        # Generate individual test files
        for filename, file_info in test_files.items():
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(file_info["data"], f, indent=2)
            print(f"✅ Generated {filename}")
        
        # Generate additional random test cases
        for i in range(5):
            random_data = self.generate_scenario_data("random")
            random_filename = f"random_test_{i+1}.json"
            random_filepath = os.path.join(output_dir, random_filename)
            with open(random_filepath, 'w') as f:
                json.dump(random_data, f, indent=2)
            print(f"✅ Generated {random_filename}")
        
        # Create comprehensive test summary
        self._create_test_summary(output_dir, test_files)
        
        print(f"\n🎉 All test data files created successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Total files generated: {len(os.listdir(output_dir))}")
        
        return output_dir
    
    def _create_test_summary(self, output_dir: str, test_files: dict):
        """Create comprehensive test summary and usage guide"""
        summary = {
            "test_data_generated": datetime.now().isoformat(),
            "generator_version": "2.0",
            "output_directory": output_dir,
            "total_files": len(test_files) + 5,  # +5 for random tests
            "data_format": {
                "single_prediction": {
                    "structure": "{ \"features\": { ... } }",
                    "example_files": ["basic_test.json", "high_volume_test.json"]
                },
                "batch_prediction": {
                    "structure": "{ \"instances\": [{ ... }, { ... }] }",
                    "example_files": ["batch_test.json"]
                }
            },
            "test_scenarios": {
                filename.replace('.json', ''): info["description"] 
                for filename, info in test_files.items()
            },
            "feature_structure": {
                "required_features": self.base_features,
                "feature_count": len(self.base_features),
                "note": "These basic features will be expanded to 90 features by the inference script"
            },
            "sagemaker_usage": {
                "deploy_command": "python fixed_sagemaker_deploy.py --config config.yaml --action deploy --model-path models/best_model.pkl --model-name chinese-produce-forecaster --endpoint-name produce-forecast-test",
                "test_commands": [
                    f"python fixed_sagemaker_deploy.py --config config.yaml --action test --endpoint-name produce-forecast-test --test-data {output_dir}/basic_test.json",
                    f"python fixed_sagemaker_deploy.py --config config.yaml --action test --endpoint-name produce-forecast-test --test-data {output_dir}/batch_test.json",
                    f"python fixed_sagemaker_deploy.py --config config.yaml --action test --endpoint-name produce-forecast-test --test-data {output_dir}/high_volatility_test.json"
                ],
                "cleanup_command": "python fixed_sagemaker_deploy.py --config config.yaml --action delete --endpoint-name produce-forecast-test"
            },
            "direct_api_usage": {
                "curl_example": f"curl -X POST https://your-endpoint.amazonaws.com/invocations -H 'Content-Type: application/json' -d @{output_dir}/basic_test.json",
                "boto3_example": "import boto3; runtime = boto3.client('sagemaker-runtime'); response = runtime.invoke_endpoint(EndpointName='your-endpoint', ContentType='application/json', Body=json.dumps(test_data))"
            },
            "validation_notes": {
                "feature_engineering": "Input features are automatically expanded to 90 features by the inference script",
                "data_types": "All numeric values should be float or int",
                "missing_values": "Any missing basic features will be filled with realistic defaults",
                "expected_output": "JSON response with predictions, confidence scores, and metadata"
            }
        }
        
        summary_filepath = os.path.join(output_dir, "test_summary.json")
        with open(summary_filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Generated test_summary.json")
        
        # Create a simple README for the test data
        readme_content = f"""# Test Data for Chinese Produce Market Forecasting

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Available Test Files

### Single Prediction Tests
- `basic_test.json` - Standard test case
- `high_volume_test.json` - High volume scenario (450+ units)
- `low_volume_test.json` - Low volume scenario (35 units)
- `high_volatility_test.json` - Price volatility scenario
- `holiday_peak_test.json` - Holiday demand peak
- `spring_fresh_test.json` - Spring season scenario
- `summer_peak_test.json` - Summer peak scenario
- `weekend_shopping_test.json` - Weekend pattern
- `random_test_*.json` - Random scenarios for robustness testing

### Batch Prediction Tests
- `batch_test.json` - Multiple instances for batch testing

## Quick Test Commands

```bash
# Test basic scenario
python fixed_sagemaker_deploy.py --config config.yaml --action test --endpoint-name YOUR_ENDPOINT --test-data {output_dir}/basic_test.json

# Test batch predictions
python fixed_sagemaker_deploy.py --config config.yaml --action test --endpoint-name YOUR_ENDPOINT --test-data {output_dir}/batch_test.json

# Test high volatility scenario
python fixed_sagemaker_deploy.py --config config.yaml --action test --endpoint-name YOUR_ENDPOINT --test-data {output_dir}/high_volatility_test.json
```

## Data Format

All test files use the format expected by SageMaker endpoints:
- Single: `{{"features": {{"Total_Quantity": 150.0, ...}}}}`
- Batch: `{{"instances": [{{"Total_Quantity": 150.0, ...}}, ...]}}`

The inference script automatically expands these {len(self.base_features)} basic features to the 90 features required by the model.
"""
        
        readme_filepath = os.path.join(output_dir, "README.md")
        with open(readme_filepath, 'w') as f:
            f.write(readme_content)
        print(f"✅ Generated README.md")


def main():
    """Main function with enhanced command-line interface"""
    parser = argparse.ArgumentParser(
        description='Generate test data for Chinese Produce Market Forecasting SageMaker Endpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_data_generator.py                                    # Generate all test files
  python test_data_generator.py --scenario basic                   # Generate only basic test
  python test_data_generator.py --scenario batch --batch-size 15   # Generate batch with 15 instances
  python test_data_generator.py --output-dir tests/data           # Custom output directory
        """
    )
    
    parser.add_argument('--output-dir', default='data/example', 
                       help='Output directory for test files (default: data/example)')
    parser.add_argument('--scenario', 
                       choices=['all', 'basic', 'high_volume', 'low_volume', 'high_volatility', 
                               'holiday_peak', 'spring_fresh', 'summer_peak', 'weekend_shopping', 
                               'batch', 'random'],
                       default='all', help='Type of test scenario to generate (default: all)')
    parser.add_argument('--batch-size', type=int, default=8, 
                       help='Number of instances for batch testing (default: 8)')
    parser.add_argument('--random-count', type=int, default=5,
                       help='Number of random test files to generate (default: 5)')
    
    args = parser.parse_args()
    
    print("🔧 Chinese Produce Market Forecasting - Test Data Generator")
    print("=" * 60)
    
    generator = TestDataGenerator()
    
    if args.scenario == 'all':
        # Generate complete test suite
        print("📊 Generating complete test data suite...")
        output_dir = generator.create_all_test_files(args.output_dir)
        
        print(f"\n🧪 Quick test commands:")
        print(f"python fixed_sagemaker_deploy.py --config config.yaml --action test --endpoint-name YOUR_ENDPOINT --test-data {output_dir}/basic_test.json")
        print(f"python fixed_sagemaker_deploy.py --config config.yaml --action test --endpoint-name YOUR_ENDPOINT --test-data {output_dir}/batch_test.json")
        
    else:
        # Generate specific scenario
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"📊 Generating {args.scenario} test data...")
        
        if args.scenario == 'batch':
            data = generator.generate_batch_data(args.batch_size)
            filename = 'batch_test.json'
        elif args.scenario == 'basic':
            data = generator.generate_basic_test_data()
            filename = 'basic_test.json'
        else:
            data = generator.generate_scenario_data(args.scenario)
            filename = f'{args.scenario}_test.json'
        
        filepath = os.path.join(args.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Generated {filename} in {args.output_dir}")
        print(f"\n🧪 Test with:")
        print(f"python fixed_sagemaker_deploy.py --config config.yaml --action test --endpoint-name YOUR_ENDPOINT --test-data {filepath}")
    
    print(f"\n✨ Test data generation completed successfully!")


if __name__ == "__main__":
    main()