#!/usr/bin/env python3
"""
Enhanced SageMaker Deployment Diagnostic Script
Provides detailed CloudWatch analysis and specific fix recommendations

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import os
import json
import boto3
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import re
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_model_file(model_path):
    """Enhanced model file checking with feature analysis"""
    print("🔍 CHECKING MODEL FILE")
    print("=" * 40)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"📊 File size: {file_size:.2f} MB")
    
    try:
        print("🔄 Loading model...")
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"📦 Model type: {type(model).__name__}")
        
        # Analyze model structure
        actual_model = None
        scaler = None
        
        if isinstance(model, dict):
            print(f"🔑 Dictionary keys: {list(model.keys())}")
            if 'model' in model:
                actual_model = model['model']
                print(f"🤖 Actual model type: {type(actual_model).__name__}")
            if 'scaler' in model:
                scaler = model['scaler']
                print(f"📏 Scaler found: {type(scaler).__name__}")
        else:
            actual_model = model
            print(f"🤖 Direct model type: {type(model).__name__}")
        
        # Check model capabilities
        if actual_model and hasattr(actual_model, 'predict'):
            print("✅ Model has predict method")
            
            # Try to get feature names if available
            if hasattr(actual_model, 'feature_names_in_'):
                features = actual_model.feature_names_in_
                print(f"📋 Expected features ({len(features)}):")
                print(f"   First 10: {list(features[:10])}")
                if len(features) > 10:
                    print(f"   Last 10: {list(features[-10:])}")
                    
                # Check for specific features that cause issues
                problematic_features = ['Autumn_Price', 'Avg_Price_Change', 'Avg_Price_Lag_1', 
                                      'Season_Autumn', 'Holiday_Demand']
                found_features = [f for f in problematic_features if f in features]
                if found_features:
                    print(f"⚠️  Model expects complex features: {found_features}")
                    print("   This requires feature engineering in inference script!")
        else:
            print("❌ Model missing predict method")
            return False
        
        # Test prediction with minimal data
        print("\n🧪 Testing prediction with minimal data...")
        try:
            sample_data = pd.DataFrame([{
                'Total_Quantity': 100.0,
                'Avg_Price': 15.0,
                'Transaction_Count': 10,
                'Month': 6,
                'DayOfWeek': 2
            }])
            
            if isinstance(model, dict) and 'model' in model:
                test_model = model['model']
                test_scaler = model.get('scaler')
                
                if test_scaler:
                    try:
                        scaled_data = test_scaler.transform(sample_data)
                        prediction = test_model.predict(scaled_data)
                    except Exception as scale_error:
                        print(f"⚠️  Scaling failed: {scale_error}")
                        prediction = test_model.predict(sample_data)
                else:
                    prediction = test_model.predict(sample_data)
            else:
                prediction = model.predict(sample_data)
            
            print(f"✅ Basic prediction successful: {prediction}")
            return True
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            print("🔧 This indicates feature mismatch - inference script needs feature engineering!")
            
            # More specific error analysis
            error_str = str(e)
            if "feature names" in error_str.lower():
                print("   💡 SOLUTION: Use inference script with automatic feature generation")
            elif "shape" in error_str.lower():
                print("   💡 SOLUTION: Model expects different number of features than provided")
            elif "columns" in error_str.lower():
                print("   💡 SOLUTION: Column names don't match training data")
                
            return False
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def get_enhanced_cloudwatch_logs(endpoint_name, region='us-east-1', hours_back=2):
    """Enhanced CloudWatch log analysis with better error detection"""
    print(f"\n🔍 ENHANCED CLOUDWATCH ANALYSIS FOR {endpoint_name}")
    print("=" * 70)
    
    try:
        cloudwatch_client = boto3.client('logs', region_name=region)
        log_group = f"/aws/sagemaker/Endpoints/{endpoint_name}"
        print(f"📋 Log group: {log_group}")
        
        # Get all log streams (not just the latest)
        try:
            streams_response = cloudwatch_client.describe_log_streams(
                logGroupName=log_group,
                orderBy='LastEventTime',
                descending=True,
                limit=10  # Check multiple streams
            )
            
            if not streams_response['logStreams']:
                print("❌ No log streams found")
                return []
            
            print(f"📊 Found {len(streams_response['logStreams'])} log streams")
            
            all_events = []
            
            # Analyze multiple streams for comprehensive error detection
            for i, stream in enumerate(streams_response['logStreams'][:3]):  # Top 3 streams
                stream_name = stream['logStreamName']
                print(f"\n📄 Stream {i+1}: {stream_name}")
                print(f"🕐 Last event: {datetime.fromtimestamp(stream.get('lastEventTime', 0)/1000)}")
                
                # Get events from this stream
                start_time = int((datetime.now() - timedelta(hours=hours_back)).timestamp() * 1000)
                
                try:
                    events_response = cloudwatch_client.get_log_events(
                        logGroupName=log_group,
                        logStreamName=stream_name,
                        limit=100,  # More events
                        startTime=start_time,
                        startFromHead=False
                    )
                    
                    stream_events = events_response['events']
                    all_events.extend(stream_events)
                    print(f"   📊 {len(stream_events)} events in this stream")
                    
                except Exception as e:
                    print(f"   ❌ Error reading stream: {e}")
                    continue
            
            if not all_events:
                print("❌ No log events found")
                return []
            
            # Sort all events by timestamp
            all_events.sort(key=lambda x: x['timestamp'])
            
            print(f"\n📜 COMPREHENSIVE LOG ANALYSIS ({len(all_events)} total events):")
            print("=" * 70)
            
            # Categorize and display events
            errors = []
            warnings = []
            info_events = []
            
            for event in all_events[-50:]:  # Last 50 events across all streams
                timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
                message = event['message'].strip()
                
                if any(keyword in message.lower() for keyword in ['error', 'failed', 'exception', 'traceback']):
                    errors.append((timestamp, message))
                elif any(keyword in message.lower() for keyword in ['warning', 'warn']):
                    warnings.append((timestamp, message))
                else:
                    info_events.append((timestamp, message))
            
            # Display errors with better formatting
            if errors:
                print(f"\n🚨 ERRORS FOUND ({len(errors)}):")
                print("-" * 50)
                for timestamp, message in errors[-10:]:  # Last 10 errors
                    print(f"❌ {timestamp}: {message}")
                    
                    # Multi-line error handling (for stack traces)
                    if 'Traceback' in message:
                        print("   📝 This appears to be a Python stack trace")
                    if 'IndentationError' in message:
                        print("   💡 FIX: Inference script has indentation issues")
                    if 'ImportError' in message or 'ModuleNotFoundError' in message:
                        print("   💡 FIX: Missing Python dependencies")
                    if 'joblib' in message.lower():
                        print("   💡 FIX: Model loading/serialization issue")
            
            # Display warnings
            if warnings:
                print(f"\n⚠️  WARNINGS FOUND ({len(warnings)}):")
                print("-" * 50)
                for timestamp, message in warnings[-5:]:  # Last 5 warnings
                    print(f"⚠️  {timestamp}: {message}")
            
            # Display key info events
            if info_events:
                print(f"\n📋 KEY INFO EVENTS ({len(info_events)}):")
                print("-" * 50)
                for timestamp, message in info_events[-10:]:  # Last 10 info events
                    if any(keyword in message.lower() for keyword in ['starting', 'loading', 'installing', 'building']):
                        print(f"ℹ️  {timestamp}: {message}")
            
            # Enhanced error pattern analysis
            analyze_error_patterns(all_events)
            
            return all_events
            
        except cloudwatch_client.exceptions.ResourceNotFoundException:
            print(f"❌ Log group not found: {log_group}")
            print("💡 This means the container never started properly")
            return []
            
    except Exception as e:
        print(f"❌ Error accessing CloudWatch: {e}")
        return []


def analyze_error_patterns(events: List[Dict]) -> None:
    """Advanced error pattern analysis"""
    print(f"\n🔍 ADVANCED ERROR PATTERN ANALYSIS:")
    print("=" * 50)
    
    all_messages = " ".join([event['message'] for event in events])
    all_messages_lower = all_messages.lower()
    
    issues_found = []
    fixes = []
    
    # Container startup issues
    if 'failed to start container' in all_messages_lower:
        issues_found.append("Container failed to start")
        fixes.append("Check Docker image compatibility and resource limits")
    
    # Python/dependency issues
    if 'importerror' in all_messages_lower or 'modulenotfounderror' in all_messages_lower:
        issues_found.append("Python import/dependency error")
        fixes.append("Update requirements.txt with correct package versions")
    
    # Inference script issues
    if 'indentationerror' in all_messages_lower:
        issues_found.append("Python indentation error in inference script")
        fixes.append("Use ultra-clean inference script generator")
    
    if 'syntaxerror' in all_messages_lower:
        issues_found.append("Python syntax error in inference script")
        fixes.append("Validate inference script syntax")
    
    # Model loading issues
    if 'joblib' in all_messages_lower and 'error' in all_messages_lower:
        issues_found.append("Model loading/serialization error")
        fixes.append("Check model file compatibility and joblib version")
    
    # Feature mismatch issues
    if 'feature names' in all_messages_lower or 'feature_names_in_' in all_messages_lower:
        issues_found.append("Feature name mismatch")
        fixes.append("Implement automatic feature engineering in inference script")
    
    if 'shape' in all_messages_lower and 'mismatch' in all_messages_lower:
        issues_found.append("Feature count mismatch")
        fixes.append("Generate all expected features in inference script")
    
    # Memory/resource issues
    if 'memory' in all_messages_lower or 'oom' in all_messages_lower:
        issues_found.append("Out of memory")
        fixes.append("Use larger instance type (ml.m5.large or ml.m5.xlarge)")
    
    # Health check issues
    if 'health check' in all_messages_lower or 'ping' in all_messages_lower:
        issues_found.append("Health check failure")
        fixes.append("Container not responding to health checks - check inference script")
    
    # Timeout issues
    if 'timeout' in all_messages_lower:
        issues_found.append("Timeout during startup")
        fixes.append("Optimize model loading or use larger instance")
    
    # Package compatibility
    if 'pandas' in all_messages_lower and 'error' in all_messages_lower:
        issues_found.append("Pandas compatibility issue")
        fixes.append("Use compatible pandas version (2.1.4)")
    
    if 'numpy' in all_messages_lower and 'error' in all_messages_lower:
        issues_found.append("NumPy compatibility issue")
        fixes.append("Use compatible numpy version (1.24.3)")
    
    if 'sklearn' in all_messages_lower and 'error' in all_messages_lower:
        issues_found.append("Scikit-learn compatibility issue")
        fixes.append("Use compatible scikit-learn version (1.3.2)")
    
    # Display findings
    if issues_found:
        print("🎯 SPECIFIC ISSUES DETECTED:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
        
        print(f"\n💡 RECOMMENDED FIXES:")
        for i, fix in enumerate(fixes, 1):
            print(f"   {i}. {fix}")
    else:
        print("🤔 No specific error patterns detected in logs")
        print("💡 This might be a subtle timing or resource issue")
    
    # Specific SageMaker recommendations
    print(f"\n🚀 SAGEMAKER-SPECIFIC RECOMMENDATIONS:")
    if 'indentationerror' in all_messages_lower:
        print("   🔧 IMMEDIATE FIX: Run the ultra-clean inference patch")
        print("      python inference_method_patch.py")
    
    if 'feature' in all_messages_lower and 'mismatch' in all_messages_lower:
        print("   🔧 FEATURE FIX: Use inference script with automatic feature generation")
        print("      Your model expects 105+ features, test data has ~5 features")
    
    if 'memory' in all_messages_lower:
        print("   🔧 RESOURCE FIX: Use larger instance type")
        print("      --instance-type ml.m5.large (or ml.m5.xlarge)")


def check_endpoint_status(endpoint_name, region='us-east-1'):
    """Check current endpoint status with detailed info"""
    print(f"\n📊 ENDPOINT STATUS CHECK: {endpoint_name}")
    print("=" * 50)
    
    try:
        sagemaker_client = boto3.client('sagemaker', region_name=region)
        
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            
            status = response['EndpointStatus']
            creation_time = response['CreationTime']
            last_modified = response['LastModifiedTime']
            
            print(f"🔹 Status: {status}")
            print(f"🔹 Created: {creation_time}")
            print(f"🔹 Last Modified: {last_modified}")
            
            if status == 'Failed':
                if 'FailureReason' in response:
                    print(f"❌ Failure Reason: {response['FailureReason']}")
                    
                    # Parse failure reason for specific guidance
                    failure_reason = response['FailureReason'].lower()
                    if 'health check' in failure_reason:
                        print("💡 Health check failure usually means inference script issues")
                    if 'memory' in failure_reason:
                        print("💡 Try larger instance type: ml.m5.large")
                    if 'timeout' in failure_reason:
                        print("💡 Model loading taking too long - optimize inference script")
            
            elif status == 'Creating':
                elapsed = datetime.now(creation_time.tzinfo) - creation_time
                print(f"🕐 Creating for: {elapsed}")
                if elapsed.total_seconds() > 1200:  # 20 minutes
                    print("⚠️  Taking longer than usual - might be stuck")
            
            return status
            
        except sagemaker_client.exceptions.ClientError as e:
            if 'ValidationException' in str(e):
                print("✅ Endpoint doesn't exist")
                return 'NotFound'
            else:
                print(f"❌ Error checking status: {e}")
                return 'Error'
                
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return 'Error'


def main():
    print("🩺 ENHANCED SAGEMAKER DEPLOYMENT DIAGNOSTIC")
    print("=" * 70)
    print("Advanced diagnosis for SageMaker endpoint failures")
    print()
    
    # Configuration
    model_path = "models/best_model.pkl"
    endpoint_name = "produce-forecast-staging"
    region = 'us-east-1'
    
    # 1. Enhanced model checking
    print("🎯 STEP 1: ENHANCED MODEL ANALYSIS")
    model_ok = check_model_file(model_path)
    
    # 2. Endpoint status
    print("\n🎯 STEP 2: ENDPOINT STATUS CHECK")
    status = check_endpoint_status(endpoint_name, region)
    
    # 3. Enhanced CloudWatch analysis
    print("\n🎯 STEP 3: COMPREHENSIVE LOG ANALYSIS")
    events = get_enhanced_cloudwatch_logs(endpoint_name, region, hours_back=3)
    
    # 4. Cleanup if needed
    if status == 'Failed':
        print(f"\n🎯 STEP 4: CLEANUP FAILED ENDPOINT")
        print("=" * 40)
        try:
            sagemaker_client = boto3.client('sagemaker', region_name=region)
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            print("✅ Failed endpoint deletion initiated")
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")
    
    # 5. Final recommendations
    print(f"\n🎯 FINAL RECOMMENDATIONS:")
    print("=" * 40)
    
    if not model_ok:
        print("🔧 PRIORITY 1: Fix model issues")
        print("   python model_validator.py --model-path models/best_model.pkl")
        print("   Consider retraining if model is corrupted")
    
    if events and any('indentationerror' in event['message'].lower() for event in events):
        print("🔧 PRIORITY 2: Fix inference script indentation")
        print("   python inference_method_patch.py")
    
    if events and any('feature' in event['message'].lower() for event in events):
        print("🔧 PRIORITY 3: Fix feature mismatch")
        print("   Use inference script with automatic feature engineering")
    
    print(f"\n📋 NEXT DEPLOYMENT COMMAND:")
    print("python src/deployment/sagemaker_deploy.py \\")
    print("  --config config.yaml \\")
    print("  --action deploy \\")
    print("  --model-path models/best_model.pkl \\")
    print("  --model-name chinese-produce-forecaster \\")
    print("  --endpoint-name produce-forecast-staging-fixed \\")
    print("  --environment staging")
    
    print(f"\n🎉 Good luck with your deployment! 🚀")


if __name__ == "__main__":
    main()