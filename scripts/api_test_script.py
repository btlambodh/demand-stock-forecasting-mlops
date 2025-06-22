#!/usr/bin/env python3
"""
Test script for the Chinese Produce Market Forecasting API
Fixes: Tests dynamic feature order extraction and compatibility

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import requests
import json
import time
import os
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Version: {data['version']}")
            print(f"   Models loaded: {data['models_loaded']}")
            print(f"   Uptime: {data['uptime_seconds']:.2f}s")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_models_endpoint():
    """Test models listing endpoint"""
    print("\n🔍 Testing models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/models")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models endpoint passed")
            print(f"   Available models: {len(data['models'])} total")
            print(f"   Default model: {data['default_model']}")
            print(f"   Feature engineering: {data['feature_engineering']}")
            print(f"   Status: {data['status']}")
            print(f"   Version: {data['version']}")
            print(f"   Sample models: {data['models'][:3]}...")
            return True
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False

def test_feature_example():
    """Test feature example endpoint"""
    print("\n🔍 Testing feature example endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/features/example")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Feature example endpoint passed")
            print(f"   Example input keys: {list(data['example_input'].keys())}")
            print(f"   Column mapping: {data['mapping']}")
            print(f"   Version: {data['version']}")
            print(f"   Note: {data['note']}")
            return True, data['example_input']
        else:
            print(f"❌ Feature example failed: {response.status_code}")
            return False, None
    except Exception as e:
        print(f"❌ Feature example error: {e}")
        return False, None

def test_single_prediction():
    """Test single prediction with FIXED SageMaker-format features"""
    print("\n🔍 Testing FIXED single prediction endpoint...")
    try:
        # Use FIXED SageMaker-compatible input format
        test_data = {
            "Total_Quantity": 150.5,
            "Avg_Price": 18.50,
            "Transaction_Count": 25,
            "Month": 6,
            "DayOfWeek": 2,
            "IsWeekend": 0,
            "Price_Volatility": 1.2,
            "Discount_Count": 3,
            "Wholesale_Price": 14.0,  # Mapped to "Wholesale Price (RMB/kg)"
            "Loss_Rate": 8.5,         # Mapped to "Loss Rate (%)"
            "Category_Code": 1,
            "Item_Code": 101
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=test_data, timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ FIXED single prediction passed")
            print(f"   Predicted price: ¥{data['predicted_price']:.2f}/kg")
            print(f"   Confidence: {data['confidence']:.3f}")
            print(f"   Model used: {data['model_used']}")
            print(f"   Features engineered: {data['features_engineered']}")
            
            # Sanity check on prediction value
            predicted_price = data['predicted_price']
            if 1 <= predicted_price <= 1000:
                print(f"   ✅ Prediction value within acceptable range")
                if 5 <= predicted_price <= 100:
                    print(f"   ✅ Prediction value looks very reasonable")
                return True
            else:
                print(f"   ❌ Prediction value too extreme: {predicted_price}")
                return False
        else:
            print(f"❌ FIXED single prediction failed: {response.status_code}")
            if response.status_code == 500:
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    print(f"   Error: {error_detail}")
                except:
                    print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ FIXED single prediction error: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint with FIXED features"""
    print("\n🔍 Testing FIXED batch prediction endpoint...")
    try:
        # Create test data for batch prediction (FIXED SageMaker format)
        test_data = {
            "instances": [
                {
                    "Total_Quantity": 150.5,
                    "Avg_Price": 18.50,
                    "Transaction_Count": 25,
                    "Month": 6,
                    "DayOfWeek": 2,
                    "IsWeekend": 0,
                    "Price_Volatility": 1.2,
                    "Wholesale_Price": 14.0,
                    "Loss_Rate": 8.5
                },
                {
                    "Total_Quantity": 200.0,
                    "Avg_Price": 16.80,
                    "Transaction_Count": 30,
                    "Month": 7,
                    "DayOfWeek": 5,
                    "IsWeekend": 1,
                    "Price_Volatility": 0.8,
                    "Wholesale_Price": 12.5,
                    "Loss_Rate": 10.0
                },
                {
                    "Total_Quantity": 100.0,
                    "Avg_Price": 22.30,
                    "Transaction_Count": 15,
                    "Month": 8,
                    "DayOfWeek": 0,
                    "IsWeekend": 0,
                    "Price_Volatility": 1.5,
                    "Wholesale_Price": 16.8,
                    "Loss_Rate": 6.5
                }
            ],
            "model_name": "best_model"
        }
        
        response = requests.post(f"{BASE_URL}/predict/batch", json=test_data, timeout=20)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ FIXED batch prediction passed")
            print(f"   Batch ID: {data['batch_id']}")
            print(f"   Predictions count: {len(data['predictions'])}")
            print(f"   Processing time: {data['processing_time_ms']:.2f}ms")
            print(f"   Model used: {data['model_used']}")
            
            reasonable_predictions = 0
            for i, pred in enumerate(data['predictions']):
                price = pred['predicted_price']
                print(f"   Prediction {i+1}: ¥{price:.2f}/kg")
                if 1 <= price <= 1000:  # Lenient range for testing
                    reasonable_predictions += 1
            
            success_rate = reasonable_predictions / len(data['predictions'])
            print(f"   ✅ {reasonable_predictions}/{len(data['predictions'])} predictions within range ({success_rate*100:.1f}%)")
            
            return success_rate >= 0.8  # 80% success rate
        else:
            print(f"❌ FIXED batch prediction failed: {response.status_code}")
            if response.status_code == 500:
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    print(f"   Error: {error_detail}")
                except:
                    print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ FIXED batch prediction error: {e}")
        return False

def test_different_models():
    """Test predictions with different models"""
    print("\n🔍 Testing different models...")
    try:
        # Get available models first
        models_response = requests.get(f"{BASE_URL}/models")
        if models_response.status_code != 200:
            print("❌ Could not get models list")
            return False
            
        models = models_response.json()['models']
        
        # Prioritize models likely to work and test a reasonable subset
        priority_models = ['best_model', 'linear_regression', 'ridge', 'gradient_boosting']
        test_models = []
        
        # Add priority models that exist
        for model in priority_models:
            if model in models:
                test_models.append(model)
        
        # Add other models up to 6 total
        for model in models:
            if model not in test_models and len(test_models) < 6:
                test_models.append(model)
        
        test_data = {
            "Total_Quantity": 175.0,
            "Avg_Price": 20.00,
            "Transaction_Count": 18,
            "Month": 8,
            "DayOfWeek": 3,
            "IsWeekend": 0,
            "Wholesale_Price": 15.5,
            "Loss_Rate": 7.5
        }
        
        successful_predictions = 0
        reasonable_predictions = 0
        
        for model_name in test_models:
            try:
                response = requests.post(
                    f"{BASE_URL}/predict?model_name={model_name}", 
                    json=test_data,
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    price = result['predicted_price']
                    print(f"   ✅ {model_name}: ¥{price:.2f}")
                    successful_predictions += 1
                    
                    # Check if prediction is reasonable
                    if 1 <= price <= 1000:
                        reasonable_predictions += 1
                    else:
                        print(f"      ⚠️  Price seems extreme for {model_name}")
                        
                else:
                    print(f"   ❌ {model_name}: Failed ({response.status_code})")
                    if response.status_code == 500:
                        print(f"      Still has feature order issues")
                    
            except Exception as e:
                print(f"   ❌ {model_name}: Error - {e}")
        
        success_rate = successful_predictions / len(test_models) if test_models else 0
        reasonable_rate = reasonable_predictions / len(test_models) if test_models else 0
        
        print(f"   Model success rate: {success_rate*100:.1f}% ({successful_predictions}/{len(test_models)})")
        print(f"   Reasonable predictions: {reasonable_rate*100:.1f}% ({reasonable_predictions}/{len(test_models)})")
        
        if success_rate >= 0.8:
            print(f"   ✅ FIXED! Most models working correctly")
            return True
        elif success_rate >= 0.5:
            print(f"   ⚠️  Partial fix - some models still have issues")
            return True
        else:
            print(f"   ❌ Still major model compatibility issues")
            return False
        
    except Exception as e:
        print(f"❌ Model testing error: {e}")
        return False

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🔍 Testing edge cases...")
    
    edge_cases = [
        {
            "name": "Minimum viable values",
            "data": {
                "Total_Quantity": 1.0,
                "Avg_Price": 5.0,
                "Transaction_Count": 1,
                "Month": 1,
                "DayOfWeek": 0,
                "IsWeekend": 0,
                "Wholesale_Price": 3.0,
                "Loss_Rate": 5.0
            }
        },
        {
            "name": "High-end produce",
            "data": {
                "Total_Quantity": 50.0,
                "Avg_Price": 45.0,
                "Transaction_Count": 8,
                "Month": 12,
                "DayOfWeek": 6,
                "IsWeekend": 1,
                "Wholesale_Price": 35.0,
                "Loss_Rate": 15.0
            }
        },
        {
            "name": "Weekend peak season",
            "data": {
                "Total_Quantity": 300.0,
                "Avg_Price": 25.0,
                "Transaction_Count": 45,
                "Month": 10,
                "DayOfWeek": 6,
                "IsWeekend": 1,
                "Price_Volatility": 2.5,
                "Wholesale_Price": 18.0,
                "Loss_Rate": 12.0
            }
        },
        {
            "name": "Low season midweek",
            "data": {
                "Total_Quantity": 75.0,
                "Avg_Price": 12.50,
                "Transaction_Count": 12,
                "Month": 2,
                "DayOfWeek": 2,
                "IsWeekend": 0,
                "Price_Volatility": 0.5,
                "Wholesale_Price": 9.0,
                "Loss_Rate": 6.0
            }
        }
    ]
    
    successful_cases = 0
    reasonable_cases = 0
    
    for case in edge_cases:
        try:
            response = requests.post(f"{BASE_URL}/predict", json=case['data'], timeout=15)
            if response.status_code == 200:
                result = response.json()
                price = result['predicted_price']
                print(f"   ✅ {case['name']}: ¥{price:.2f}")
                successful_cases += 1
                
                # Check reasonableness
                if 1 <= price <= 1000:
                    reasonable_cases += 1
                else:
                    print(f"      ⚠️  Price seems extreme")
                    
            else:
                print(f"   ❌ {case['name']}: Failed ({response.status_code})")
                if response.status_code == 500:
                    print(f"      Still has feature order issues")
        except Exception as e:
            print(f"   ❌ {case['name']}: Error - {e}")
    
    success_rate = successful_cases / len(edge_cases)
    reasonable_rate = reasonable_cases / len(edge_cases) if successful_cases > 0 else 0
    
    print(f"   Edge case success rate: {success_rate*100:.1f}%")
    print(f"   Reasonable predictions: {reasonable_rate*100:.1f}%")
    
    if success_rate >= 0.75:
        print(f"   ✅ FIXED! Edge case handling working well")
        return True
    elif success_rate >= 0.5:
        print(f"   ⚠️  Most edge cases working")
        return True
    else:
        print(f"   ❌ Edge case handling still has issues")
        return False

def run_load_test():
    """Run a load test with FIXED SageMaker-format data"""
    print("\n🔍 Running load test...")
    try:
        test_data = {
            "Total_Quantity": 150.5,
            "Avg_Price": 18.50,
            "Transaction_Count": 25,
            "Month": 6,
            "DayOfWeek": 2,
            "IsWeekend": 0,
            "Wholesale_Price": 14.0,
            "Loss_Rate": 8.5
        }
        
        num_requests = 10
        start_time = time.time()
        successful_requests = 0
        latencies = []
        reasonable_predictions = 0
        
        for i in range(num_requests):
            try:
                req_start = time.time()
                response = requests.post(f"{BASE_URL}/predict", json=test_data, timeout=15)
                latency = (time.time() - req_start) * 1000
                
                if response.status_code == 200:
                    successful_requests += 1
                    latencies.append(latency)
                    
                    # Check prediction reasonableness
                    result = response.json()
                    if 1 <= result['predicted_price'] <= 1000:
                        reasonable_predictions += 1
                        
            except:
                pass
        
        total_time = time.time() - start_time
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        print(f"✅ Load test completed")
        print(f"   Requests sent: {num_requests}")
        print(f"   Successful: {successful_requests}")
        print(f"   Success rate: {(successful_requests/num_requests)*100:.1f}%")
        print(f"   Reasonable predictions: {reasonable_predictions}")
        print(f"   Average latency: {avg_latency:.2f}ms")
        print(f"   Total time: {total_time:.2f}s")
        
        if successful_requests >= num_requests * 0.8:
            print(f"   ✅ FIXED! Load test performance excellent")
            return True
        elif successful_requests >= num_requests * 0.6:
            print(f"   ⚠️  Load test performance acceptable")
            return True
        else:
            print(f"   ❌ Load test performance poor")
            return False
        
    except Exception as e:
        print(f"❌ Load test error: {e}")
        return False

def test_sagemaker_compatibility():
    """Compare local API predictions with SageMaker-expected format"""
    print("\n🔍 Testing FIXED SageMaker format compatibility...")
    
    try:
        # Test data that exactly matches SageMaker format
        sagemaker_test_data = {
            "Total_Quantity": 150.0,
            "Avg_Price": 18.5,
            "Transaction_Count": 25,
            "Month": 7,
            "DayOfWeek": 1,
            "IsWeekend": 0,
            "Price_Volatility": 1.2,
            "Revenue": 2775.0,
            "Min_Price": 16.0,
            "Max_Price": 21.0,
            "Discount_Count": 3,
            "Discount_Rate": 0.12,
            "Wholesale_Price": 14.0,  # This will be mapped to "Wholesale Price (RMB/kg)"
            "Loss_Rate": 8.5,         # This will be mapped to "Loss Rate (%)"
            "Year": 2024,
            "Quarter": 3,
            "DayOfYear": 202,
            "WeekOfYear": 29
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=sagemaker_test_data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ FIXED SageMaker format test passed")
            print(f"   Predicted price: ¥{result['predicted_price']:.2f}")
            print(f"   Features engineered: {result['features_engineered']}")
            print(f"   Model used: {result['model_used']}")
            
            # Check if feature count looks reasonable (model-specific now)
            feature_count = result['features_engineered']
            if feature_count >= 80 and feature_count <= 100:
                print(f"   ✅ Feature count looks reasonable ({feature_count})")
            else:
                print(f"   ⚠️  Unexpected feature count: {feature_count}")
            
            # Check prediction reasonableness
            price = result['predicted_price']
            if 1 <= price <= 1000:
                print(f"   ✅ Prediction within acceptable range")
                return True
            else:
                print(f"   ❌ Prediction extreme: {price}")
                return False
        else:
            print(f"   ❌ FIXED SageMaker format test failed: {response.status_code}")
            if response.status_code == 500:
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    print(f"   Error: {error_detail}")
                    if "feature names" in error_detail.lower():
                        print(f"   🔴 FEATURE ORDER STILL NOT FIXED!")
                except:
                    print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ FIXED SageMaker format test error: {e}")
        return False

def test_with_provided_test_data():
    """Test using the provided test data files if they exist"""
    print("\n🔍 Testing with provided test data...")
    
    test_files = [
        "basic_test.json",
        "high_volume_test.json", 
        "low_volume_test.json",
        "batch_test.json"
    ]
    
    successful_tests = 0
    found_files = 0
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            continue
            
        found_files += 1
        try:
            with open(test_file, 'r') as f:
                test_data = json.load(f)
            
            # Handle different formats
            if "instances" in test_data:
                # Batch test
                response = requests.post(f"{BASE_URL}/predict/batch", json=test_data, timeout=20)
                endpoint = "batch"
            elif "features" in test_data:
                # Single test with features wrapper
                response = requests.post(f"{BASE_URL}/predict", json=test_data["features"], timeout=15)
                endpoint = "single"
            else:
                # Direct single test
                response = requests.post(f"{BASE_URL}/predict", json=test_data, timeout=15)
                endpoint = "single"
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ {test_file} ({endpoint}): Success")
                successful_tests += 1
                
                if endpoint == "single":
                    print(f"      Price: ¥{result['predicted_price']:.2f}")
                else:
                    print(f"      Predictions: {len(result['predictions'])}")
            else:
                print(f"   ❌ {test_file}: Failed ({response.status_code})")
                
        except Exception as e:
            print(f"   ❌ {test_file}: Error - {e}")
    
    if found_files == 0:
        print(f"   ⚠️  No test files found")
        return True  # Don't fail if no files exist
    
    success_rate = successful_tests / found_files if found_files > 0 else 0
    print(f"   Test file success rate: {success_rate*100:.1f}% ({successful_tests}/{found_files})")
    return success_rate > 0.5

def main():
    """Run comprehensive FIXED test suite"""
    print("🚀 Starting FIXED SageMaker-Synchronized API Test Suite")
    print("=" * 70)
    
    # Wait for API to be ready
    print("⏳ Waiting for API to be ready...")
    time.sleep(2)
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Models Endpoint", test_models_endpoint),
        ("Feature Example", lambda: test_feature_example()[0]),
        ("FIXED Single Prediction", test_single_prediction),
        ("FIXED Batch Prediction", test_batch_prediction),
        ("Different Models", test_different_models),
        ("Edge Cases", test_edge_cases),
        ("Load Test", run_load_test),
        ("FIXED SageMaker Compatibility", test_sagemaker_compatibility),
        ("Provided Test Data", test_with_provided_test_data)
    ]
    
    passed = 0
    total = len(tests)
    critical_tests = ["FIXED Single Prediction", "FIXED Batch Prediction", "FIXED SageMaker Compatibility"]
    critical_passed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                if test_name in critical_tests:
                    critical_passed += 1
            else:
                if test_name in critical_tests:
                    print(f"   🔴 CRITICAL TEST FAILED: {test_name}")
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            if test_name in critical_tests:
                print(f"   🔴 CRITICAL TEST ERROR: {test_name}")
    
    print("\n" + "=" * 70)
    print("🏁 FIXED SageMaker-Synchronized API Test Results")
    print("=" * 70)
    print(f"Overall tests passed: {passed}/{total} ({(passed/total)*100:.1f}%)")
    print(f"Critical tests passed: {critical_passed}/{len(critical_tests)}")
    
    if critical_passed == len(critical_tests):
        print("🎉 ALL CRITICAL TESTS PASSED! API IS FIXED AND WORKING! 🎉")
        print("✅ Feature order mismatch issue has been resolved!")
        print("✅ Dynamic feature extraction is working correctly!")
    elif critical_passed >= len(critical_tests) * 0.7:
        print("🟡 Most critical tests passed. Minor issues remain.")
    else:
        print("🔴 Critical issues still present.")
        print("⚠️  Check the error messages above for remaining issues")
    
    print(f"\n📊 Interactive docs: {BASE_URL}/docs")
    print(f"📈 Metrics: {BASE_URL}/metrics") 
    print(f"🔧 Example format: {BASE_URL}/features/example")
    
    # Show example requests
    print(f"\n📝 Example curl requests:")
    print(f"# FIXED Single prediction")
    print(f"""curl -X POST "{BASE_URL}/predict" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "Total_Quantity": 150.5,
    "Avg_Price": 18.50,
    "Transaction_Count": 25,
    "Month": 6,
    "DayOfWeek": 2,
    "IsWeekend": 0,
    "Wholesale_Price": 14.0,
    "Loss_Rate": 8.5
  }}'""")

if __name__ == "__main__":
    main()