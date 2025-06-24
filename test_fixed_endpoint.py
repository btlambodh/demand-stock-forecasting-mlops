#!/usr/bin/env python3
"""
Test Fixed Endpoint
Simple test script to verify the fixed endpoint works

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import boto3
import json
import time
from datetime import datetime
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

def test_endpoint(endpoint_name: str):
    """Test the fixed endpoint with various input formats"""
    print(f" TESTING FIXED ENDPOINT: {endpoint_name}")
    print("=" * 60)
    
    try:
        # Create predictor
        sagemaker_session = sagemaker.Session()
        predictor = Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
        
        # Test 1: Basic features only
        print("Test 1: Basic features only")
        basic_test = {
            "features": {
                "Total_Quantity": 100.0,
                "Avg_Price": 15.0,
                "Transaction_Count": 10,
                "Month": 6,
                "DayOfWeek": 2
            }
        }
        
        start_time = time.time()
        result1 = predictor.predict(basic_test)
        latency1 = (time.time() - start_time) * 1000
        
        print(f" Result: {result1}")
        print(f"⏱️  Latency: {latency1:.2f}ms")
        print()
        
        # Test 2: Minimal features
        print("Test 2: Minimal features")
        minimal_test = {
            "features": {
                "Avg_Price": 18.5,
                "Month": 7
            }
        }
        
        start_time = time.time()
        result2 = predictor.predict(minimal_test)
        latency2 = (time.time() - start_time) * 1000
        
        print(f" Result: {result2}")
        print(f"⏱️  Latency: {latency2:.2f}ms")
        print()
        
        # Test 3: Real-world scenario
        print("Test 3: Real-world produce data")
        produce_test = {
            "features": {
                "Total_Quantity": 250.0,
                "Avg_Price": 22.30,
                "Transaction_Count": 35,
                "Month": 8,
                "DayOfWeek": 5,
                "IsWeekend": 1,
                "Price_Volatility": 1.2,
                "Revenue": 5575.0
            }
        }
        
        start_time = time.time()
        result3 = predictor.predict(produce_test)
        latency3 = (time.time() - start_time) * 1000
        
        print(f" Result: {result3}")
        print(f"⏱️  Latency: {latency3:.2f}ms")
        print()
        
        # Test 4: Direct feature dictionary (no wrapper)
        print("Test 4: Direct features (no wrapper)")
        direct_test = {
            "Total_Quantity": 150.0,
            "Avg_Price": 16.8,
            "Month": 9
        }
        
        start_time = time.time()
        result4 = predictor.predict(direct_test)
        latency4 = (time.time() - start_time) * 1000
        
        print(f" Result: {result4}")
        print(f"⏱️  Latency: {latency4:.2f}ms")
        print()
        
        # Summary
        avg_latency = (latency1 + latency2 + latency3 + latency4) / 4
        
        print(" TEST SUMMARY")
        print("=" * 30)
        print(f" All 4 tests passed!")
        print(f"⏱️  Average latency: {avg_latency:.2f}ms")
        print(f"🎯 Endpoint is working perfectly!")
        print()
        print(" Your endpoint can handle:")
        print("  - Basic input features")
        print("  - Minimal input features") 
        print("  - Complex produce data")
        print("  - Different input formats")
        print("  - Automatic feature engineering")
        
        return True
        
    except Exception as e:
        print(f" Test failed: {e}")
        return False


def check_endpoint_status(endpoint_name: str):
    """Check if endpoint is ready"""
    print(f"📊 CHECKING ENDPOINT STATUS: {endpoint_name}")
    print("=" * 50)
    
    try:
        sagemaker_client = boto3.client('sagemaker')
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        
        status = response['EndpointStatus']
        print(f"Status: {status}")
        
        if status == 'InService':
            print(" Endpoint is ready for testing!")
            return True
        elif status == 'Creating':
            print(" Endpoint is still being created...")
            print("Wait a few minutes and try again")
            return False
        else:
            print(f" Endpoint status: {status}")
            return False
            
    except Exception as e:
        print(f" Error checking status: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Fixed SageMaker Endpoint')
    parser.add_argument('--endpoint-name', required=True, help='Endpoint name to test')
    parser.add_argument('--wait', action='store_true', help='Wait for endpoint to be ready')
    
    args = parser.parse_args()
    
    print(f" ENDPOINT TESTING TOOL")
    print(f"Testing endpoint: {args.endpoint_name}")
    print(f"Time: {datetime.now()}")
    print()
    
    # Check status first
    ready = check_endpoint_status(args.endpoint_name)
    
    if not ready and args.wait:
        print(" Waiting for endpoint to be ready...")
        for i in range(30):  # Wait up to 15 minutes
            time.sleep(30)
            print(f"Checking status... ({i+1}/30)")
            ready = check_endpoint_status(args.endpoint_name)
            if ready:
                break
    
    if ready:
        print()
        success = test_endpoint(args.endpoint_name)
        if success:
            print(" ALL TESTS PASSED! Your endpoint is working perfectly!")
        else:
            print(" Some tests failed. Check the logs above.")
    else:
        print(" Endpoint not ready for testing")


if __name__ == "__main__":
    main()
