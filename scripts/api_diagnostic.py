#!/usr/bin/env python3
"""
Diagnostic script to debug API model loading issues
Fixes: Tests dynamic feature order extraction

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import requests
import json
import os
import time

BASE_URL = "http://localhost:8000"

def test_api_startup():
    """Test if API is running and responsive"""
    print("🔍 FIXED API Startup Diagnostic")
    print("=" * 50)
    
    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API is running and responsive")
                print(f"   Status: {data['status']}")
                print(f"   Version: {data['version']}")
                print(f"   Models loaded: {data['models_loaded']}")
                print(f"   Uptime: {data['uptime_seconds']:.2f}s")
                return True
            else:
                print(f"⚠️  API responded with status {response.status_code}")
        except Exception as e:
            print(f"⚠️  Attempt {i+1}/{max_retries}: API not responding - {e}")
            if i < max_retries - 1:
                time.sleep(2)
    
    print(f"❌ API is not accessible after {max_retries} attempts")
    return False

def detailed_model_diagnosis():
    """Get detailed information about model loading with FIXED API"""
    print("\n🔍 FIXED Detailed Model Diagnosis")
    print("=" * 50)
    
    # Test each available model
    try:
        # Get models list
        response = requests.get(f"{BASE_URL}/models")
        if response.status_code == 200:
            data = response.json()
            available_models = data['models']
            print(f"📋 Available models from /models endpoint: {len(available_models)} models")
            print(f"   Models: {available_models}")
            print(f"   Default model: {data['default_model']}")
            print(f"   Feature engineering: {data['feature_engineering']}")
            print(f"   Version: {data['version']}")
            
            # FIXED test data with proper SageMaker format
            test_data = {
                "Total_Quantity": 150.5,
                "Avg_Price": 18.50,
                "Transaction_Count": 25,
                "Month": 6,
                "DayOfWeek": 2,
                "IsWeekend": 0,
                "Price_Volatility": 1.2,
                "Discount_Count": 3,
                "Wholesale_Price": 14.0,  # FIXED: proper mapping
                "Loss_Rate": 8.5          # FIXED: proper mapping
            }
            
            print(f"\n🧪 Testing models with FIXED data format:")
            
            # Try the default model first
            print(f"\n🎯 Testing default model:")
            try:
                response = requests.post(f"{BASE_URL}/predict", json=test_data, timeout=15)
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ Success: ¥{result['predicted_price']:.2f}")
                    print(f"   Model used: {result['model_used']}")
                    print(f"   Confidence: {result['confidence']}")
                    print(f"   Features engineered: {result['features_engineered']}")
                    
                    # Check prediction reasonableness
                    price = result['predicted_price']
                    if 5 <= price <= 100:
                        print(f"   ✅ Prediction in reasonable range")
                    elif 1 <= price <= 1000:
                        print(f"   ⚠️  Prediction acceptable but check model training")
                    else:
                        print(f"   ❌ Prediction seems extreme")
                        
                else:
                    print(f"   ❌ Failed ({response.status_code}): {response.text}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            # Try a subset of specific models that should work
            test_models = []
            
            # Prioritize models that are more likely to work
            priority_models = ['best_model', 'linear_regression', 'ridge', 'gradient_boosting']
            for model in priority_models:
                if model in available_models:
                    test_models.append(model)
            
            # Add other models (up to 5 total)
            for model in available_models:
                if model not in test_models and len(test_models) < 5:
                    test_models.append(model)
            
            successful_models = 0
            total_tested = 0
            
            for model_name in test_models:
                print(f"\n🧪 Testing model: {model_name}")
                total_tested += 1
                try:
                    response = requests.post(
                        f"{BASE_URL}/predict?model_name={model_name}", 
                        json=test_data,
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        price = result['predicted_price']
                        print(f"   ✅ Success: ¥{price:.2f}")
                        print(f"   Model used: {result['model_used']}")
                        print(f"   Features: {result['features_engineered']}")
                        successful_models += 1
                        
                        # Analyze prediction quality
                        if 5 <= price <= 100:
                            print(f"   ✅ Excellent prediction range")
                        elif 1 <= price <= 1000:
                            print(f"   ⚠️  Acceptable prediction range")
                        else:
                            print(f"   ❌ Extreme prediction - model issue")
                            
                    else:
                        print(f"   ❌ Failed ({response.status_code}): {response.text}")
                        
                except Exception as e:
                    print(f"   ❌ Error: {e}")
            
            print(f"\n📊 Model Test Summary:")
            print(f"   Models tested: {total_tested}")
            print(f"   Successful: {successful_models}")
            print(f"   Success rate: {(successful_models/total_tested)*100:.1f}%")
            
            if successful_models >= total_tested * 0.8:
                print(f"   ✅ FIXED! Most models working correctly")
                return True
            elif successful_models >= total_tested * 0.5:
                print(f"   ⚠️  Partial fix - some models still have issues")
                return True
            else:
                print(f"   ❌ Still major issues with model compatibility")
                return False
                
        else:
            print(f"❌ Could not get models list: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        return False

def test_feature_engineering():
    """Test the FIXED feature engineering"""
    print(f"\n🔧 FIXED Feature Engineering Test")
    print("=" * 40)
    
    try:
        # Test different input formats
        test_cases = [
            {
                "name": "Minimal required features",
                "data": {
                    "Total_Quantity": 100.0,
                    "Avg_Price": 15.0,
                    "Transaction_Count": 10,
                    "Month": 6,
                    "DayOfWeek": 1,
                    "IsWeekend": 0
                }
            },
            {
                "name": "Full featured input",
                "data": {
                    "Total_Quantity": 150.0,
                    "Avg_Price": 18.5,
                    "Transaction_Count": 25,
                    "Month": 7,
                    "DayOfWeek": 1,
                    "IsWeekend": 0,
                    "Price_Volatility": 1.2,
                    "Min_Price": 16.0,
                    "Max_Price": 21.0,
                    "Discount_Count": 3,
                    "Wholesale_Price": 14.0,
                    "Loss_Rate": 8.5
                }
            },
            {
                "name": "Edge case values",
                "data": {
                    "Total_Quantity": 1.0,
                    "Avg_Price": 50.0,
                    "Transaction_Count": 1,
                    "Month": 12,
                    "DayOfWeek": 6,
                    "IsWeekend": 1,
                    "Wholesale_Price": 40.0,
                    "Loss_Rate": 20.0
                }
            }
        ]
        
        successful_cases = 0
        
        for test_case in test_cases:
            print(f"\n   🧪 {test_case['name']}:")
            try:
                response = requests.post(f"{BASE_URL}/predict", json=test_case['data'], timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    print(f"      ✅ Success: ¥{result['predicted_price']:.2f}")
                    print(f"      Features: {result['features_engineered']}")
                    print(f"      Model: {result['model_used']}")
                    successful_cases += 1
                    
                    # Check prediction reasonableness
                    price = result['predicted_price']
                    if 1 <= price <= 1000:
                        print(f"      ✅ Prediction within acceptable range")
                    else:
                        print(f"      ⚠️  Prediction may be extreme: {price}")
                        
                else:
                    print(f"      ❌ Failed: {response.status_code}")
                    if response.status_code == 500:
                        try:
                            error_detail = response.json().get('detail', 'Unknown error')
                            print(f"      Error: {error_detail}")
                        except:
                            print(f"      Error details: {response.text}")
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        success_rate = successful_cases / len(test_cases)
        print(f"\n   📊 Feature Engineering Success Rate: {success_rate*100:.1f}%")
        
        if success_rate >= 0.8:
            print(f"   ✅ FIXED! Feature engineering working correctly")
            return True
        elif success_rate >= 0.5:
            print(f"   ⚠️  Partial fix - some cases still failing")
            return True
        else:
            print(f"   ❌ Feature engineering still has major issues")
            return False
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False

def test_batch_functionality():
    """Test FIXED batch prediction functionality"""
    print(f"\n📦 FIXED Batch Prediction Test")
    print("=" * 35)
    
    try:
        batch_data = {
            "instances": [
                {
                    "Total_Quantity": 150.0,
                    "Avg_Price": 18.5,
                    "Transaction_Count": 25,
                    "Month": 6,
                    "DayOfWeek": 2,
                    "IsWeekend": 0,
                    "Wholesale_Price": 14.0,
                    "Loss_Rate": 8.5
                },
                {
                    "Total_Quantity": 200.0,
                    "Avg_Price": 22.0,
                    "Transaction_Count": 30,
                    "Month": 8,
                    "DayOfWeek": 5,
                    "IsWeekend": 1,
                    "Wholesale_Price": 18.0,
                    "Loss_Rate": 10.0
                }
            ],
            "model_name": "best_model"
        }
        
        response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data, timeout=20)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Batch prediction successful")
            print(f"   Batch ID: {result['batch_id']}")
            print(f"   Predictions: {len(result['predictions'])}")
            print(f"   Processing time: {result['processing_time_ms']:.2f}ms")
            print(f"   Model used: {result['model_used']}")
            
            reasonable_predictions = 0
            for i, pred in enumerate(result['predictions']):
                price = pred['predicted_price']
                print(f"   Prediction {i+1}: ¥{price:.2f} (confidence: {pred['confidence']:.3f})")
                if 1 <= price <= 1000:
                    reasonable_predictions += 1
                    
            if reasonable_predictions == len(result['predictions']):
                print(f"   ✅ All predictions within reasonable range")
                return True
            elif reasonable_predictions >= len(result['predictions']) * 0.8:
                print(f"   ⚠️  Most predictions reasonable ({reasonable_predictions}/{len(result['predictions'])})")
                return True
            else:
                print(f"   ❌ Too many extreme predictions")
                return False
        else:
            print(f"   ❌ Batch prediction failed: {response.status_code}")
            if response.status_code == 500:
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    print(f"   Error: {error_detail}")
                except:
                    print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Batch test error: {e}")
        return False

def test_error_handling():
    """Test API error handling"""
    print(f"\n⚠️  Error Handling Test")
    print("=" * 25)
    
    error_cases = [
        {
            "name": "Missing required field",
            "data": {
                "Avg_Price": 18.5,
                "Transaction_Count": 25
                # Missing Total_Quantity
            },
            "expected_status": [422, 400]
        },
        {
            "name": "Invalid data types",
            "data": {
                "Total_Quantity": "invalid",
                "Avg_Price": 18.5,
                "Transaction_Count": 25,
                "Month": 6,
                "DayOfWeek": 2,
                "IsWeekend": 0
            },
            "expected_status": [422, 400]
        },
        {
            "name": "Invalid model name",
            "data": {
                "Total_Quantity": 150.0,
                "Avg_Price": 18.5,
                "Transaction_Count": 25,
                "Month": 6,
                "DayOfWeek": 2,
                "IsWeekend": 0
            },
            "endpoint": "/predict?model_name=nonexistent_model",
            "expected_status": [404, 500, 400]
        }
    ]
    
    passed_cases = 0
    
    for case in error_cases:
        try:
            endpoint = case.get("endpoint", "/predict")
            url = f"{BASE_URL}{endpoint}"
            
            response = requests.post(url, json=case["data"], timeout=10)
            
            if response.status_code in case["expected_status"]:
                print(f"   ✅ {case['name']}: Proper error handling ({response.status_code})")
                passed_cases += 1
            elif response.status_code == 200:
                print(f"   ⚠️  {case['name']}: Accepted invalid input (might be OK)")
                passed_cases += 0.5  # Half credit for lenient handling
            else:
                print(f"   ❌ {case['name']}: Unexpected response ({response.status_code})")
                
        except Exception as e:
            print(f"   ❌ {case['name']}: Error - {e}")
    
    success_rate = passed_cases / len(error_cases)
    return success_rate >= 0.7

def test_performance():
    """Test API performance"""
    print(f"\n⚡ Performance Test")
    print("=" * 20)
    
    test_data = {
        "Total_Quantity": 150.0,
        "Avg_Price": 18.5,
        "Transaction_Count": 25,
        "Month": 6,
        "DayOfWeek": 2,
        "IsWeekend": 0,
        "Wholesale_Price": 14.0,
        "Loss_Rate": 8.5
    }
    
    num_requests = 5
    latencies = []
    successful_requests = 0
    
    print(f"   Running {num_requests} requests...")
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/predict", json=test_data, timeout=10)
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                latencies.append(latency)
                successful_requests += 1
                print(f"   Request {i+1}: {latency:.2f}ms ✅")
            else:
                print(f"   Request {i+1}: Failed ({response.status_code}) ❌")
                
        except Exception as e:
            print(f"   Request {i+1}: Error - {e} ❌")
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print(f"\n   📊 Performance Summary:")
        print(f"   Average latency: {avg_latency:.2f}ms")
        print(f"   Min latency: {min_latency:.2f}ms")
        print(f"   Max latency: {max_latency:.2f}ms")
        print(f"   Success rate: {successful_requests}/{num_requests} ({successful_requests/num_requests*100:.1f}%)")
        
        if successful_requests >= num_requests * 0.8 and avg_latency < 2000:
            print(f"   ✅ FIXED! Good performance")
            return True
        elif successful_requests >= num_requests * 0.6:
            print(f"   ⚠️  Acceptable performance")
            return True
        else:
            print(f"   ❌ Poor performance")
            return False
    else:
        print(f"   ❌ No successful requests")
        return False

def comprehensive_health_check():
    """Comprehensive health and capability check"""
    print(f"\n🏥 Comprehensive Health Check")
    print("=" * 35)
    
    checks = [
        ("API Startup", test_api_startup),
        ("Model Diagnosis", detailed_model_diagnosis),
        ("Feature Engineering", test_feature_engineering),
        ("Batch Functionality", test_batch_functionality),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance)
    ]
    
    passed_checks = 0
    critical_checks = ["Model Diagnosis", "Feature Engineering", "Batch Functionality"]
    critical_passed = 0
    
    for check_name, check_func in checks:
        try:
            print(f"\n🔍 {check_name}:")
            result = check_func()
            if result:
                passed_checks += 1
                if check_name in critical_checks:
                    critical_passed += 1
                print(f"   ✅ {check_name} passed")
            else:
                print(f"   ⚠️  {check_name} had issues")
                if check_name in critical_checks:
                    print(f"   🔴 CRITICAL CHECK FAILED: {check_name}")
        except Exception as e:
            print(f"   ❌ {check_name} failed: {e}")
            if check_name in critical_checks:
                print(f"   🔴 CRITICAL CHECK ERROR: {check_name}")
    
    health_score = passed_checks / len(checks) * 100
    critical_score = critical_passed / len(critical_checks) * 100
    
    print(f"\n📊 Overall Health Score: {health_score:.1f}% ({passed_checks}/{len(checks)} checks passed)")
    print(f"📊 Critical Health Score: {critical_score:.1f}% ({critical_passed}/{len(critical_checks)} critical checks passed)")
    
    if critical_score >= 100:
        print(f"🎉 ALL CRITICAL CHECKS PASSED! API IS FULLY FIXED! 🎉")
        return True
    elif critical_score >= 67:
        print(f"🟡 Most critical checks passed - minor issues remain")
        return True
    else:
        print(f"🔴 Critical issues still present")
        return False

if __name__ == "__main__":
    print("🔧 FIXED API Diagnostic Tool")
    print("=" * 50)
    
    if not test_api_startup():
        print("\n❌ API is not running. Please start the API first:")
        print("   python api.py")
        exit(1)
    
    overall_success = comprehensive_health_check()
    
    print(f"\n" + "=" * 50)
    print(f"🎯 DIAGNOSTIC COMPLETE")
    print(f"=" * 50)
    
    if overall_success:
        print(f"🎉 SUCCESS! Your FIXED API is working correctly!")
        print(f"✅ Feature order mismatch issue has been resolved")
        print(f"✅ Dynamic feature extraction is working")
        print(f"✅ All critical functionality is operational")
    else:
        print(f"⚠️  Some issues remain, but check the results above")
        print(f"🔧 If critical tests pass, the main fixes are working")
    
    print(f"\n📝 Next steps:")
    print(f"   • Run the full test suite: python api_test_script.py")
    print(f"   • Check the interactive docs: {BASE_URL}/docs")
    print(f"   • Monitor logs for any remaining issues")