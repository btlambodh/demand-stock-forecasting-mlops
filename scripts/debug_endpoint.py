#!/usr/bin/env python3
"""
SageMaker Endpoint Debugger
Diagnose why predictions are returning 0 with proper command-line arguments

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import argparse
import json
import pandas as pd
import numpy as np
import joblib
import boto3
import yaml
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import sagemaker

def debug_local_model(model_path="models/best_model.pkl"):
    """Debug the local model to understand what it expects"""
    print("🔍 DEBUGGING LOCAL MODEL")
    print("=" * 50)
    
    try:
        # Load the model
        print(f"Loading model from: {model_path}")
        model_artifact = joblib.load(model_path)
        
        # Analyze model structure
        if isinstance(model_artifact, dict):
            print(f"✅ Model is dictionary with keys: {list(model_artifact.keys())}")
            
            if 'model' in model_artifact:
                model = model_artifact['model']
                print(f"✅ Model type: {type(model).__name__}")
                
                # Check feature names
                if hasattr(model, 'feature_names_in_'):
                    features = model.feature_names_in_
                    print(f"✅ Model expects {len(features)} features")
                    print(f"   First 10: {list(features[:10])}")
                    print(f"   Last 10: {list(features[-10:])}")
                    
                    # Save expected features
                    with open('expected_features.json', 'w') as f:
                        json.dump(list(features), f, indent=2)
                    print(f"✅ Expected features saved to expected_features.json")
                    
                    return list(features), model_artifact
                else:
                    print("⚠️ Model doesn't have feature_names_in_ attribute")
                    return None, model_artifact
            else:
                print("❌ No 'model' key found in dictionary")
                return None, model_artifact
        else:
            print(f"✅ Direct model type: {type(model_artifact).__name__}")
            if hasattr(model_artifact, 'feature_names_in_'):
                features = model_artifact.feature_names_in_
                print(f"✅ Model expects {len(features)} features")
                return list(features), model_artifact
            else:
                print("⚠️ Model doesn't have feature_names_in_ attribute")
                return None, model_artifact
                
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def create_exact_90_features_debug(df_input):
    """
    Create EXACTLY the 90 features the model expects - DEBUG VERSION
    This should match the fixed inference script exactly
    """
    print(f"🔧 Creating exact 90 features from {df_input.shape[1]} input features")
    df = df_input.copy()
    
    try:
        # Basic feature defaults
        basic_defaults = {
            "Total_Quantity": 100.0,
            "Avg_Price": 15.0,
            "Transaction_Count": 10,
            "Price_Volatility": 1.0,
            "Min_Price": 12.0,
            "Max_Price": 18.0,
            "Discount_Count": 2,
            "Revenue": 1500.0,
            "Discount_Rate": 0.1,
            "Price_Range": 6.0,
            "Wholesale Price (RMB/kg)": 12.0,
            "Loss Rate (%)": 8.0,
            "Month": 6,
            "DayOfWeek": 1,
            "IsWeekend": 0,
            "Year": 2024,
            "Quarter": 2,
            "DayOfYear": 180,
            "WeekOfYear": 26
        }
        
        # Fill missing basic features
        for col, default_val in basic_defaults.items():
            if col not in df.columns:
                df[col] = default_val
            else:
                df[col] = df[col].fillna(default_val)
        
        # ===== MISSING FEATURES THAT MODEL EXPECTS =====
        
        # 1. Avg_Quantity (missing from inference but model expects it)
        df["Avg_Quantity"] = df["Total_Quantity"] / np.maximum(df["Transaction_Count"], 1)
        
        # 2. Category Code (missing from inference but model expects it) 
        df["Category Code"] = 1  # Default category code
        
        # ===== CALCULATED FEATURES =====
        
        if "Revenue" not in df_input.columns:
            df["Revenue"] = df["Total_Quantity"] * df["Avg_Price"]
        if "Price_Range" not in df_input.columns:
            df["Price_Range"] = df["Max_Price"] - df["Min_Price"]
        if "Discount_Rate" not in df_input.columns:
            df["Discount_Rate"] = df["Discount_Count"] / np.maximum(df["Transaction_Count"], 1)
        
        # ===== TEMPORAL FEATURES =====
        
        df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
        df["DayOfYear_Sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365)
        df["DayOfYear_Cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365)
        df["DayOfWeek_Sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
        df["DayOfWeek_Cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
        
        # Chinese holidays - only the ones model expects
        df["IsNationalDay"] = ((df["Month"] == 10) & (df["DayOfYear"].between(274, 280))).astype(int)
        df["IsLaborDay"] = ((df["Month"] == 5) & (df["DayOfYear"].between(121, 125))).astype(int)
        # NOTE: NOT including IsSpringFestival as model doesn't expect it
        
        # Season mapping
        season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 
                     5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 
                     10: "Autumn", 11: "Autumn"}
        df["Season"] = df["Month"].map(season_map).fillna("Summer")
        
        # Days since epoch
        epoch_date = pd.to_datetime("2020-01-01")
        current_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(df["DayOfYear"] - 1, unit="D")
        df["Days_Since_Epoch"] = (current_date - epoch_date).dt.days
        
        # ===== PRICE FEATURES =====
        
        df["Retail_Wholesale_Ratio"] = df["Avg_Price"] / np.maximum(df["Wholesale Price (RMB/kg)"], 0.1)
        df["Price_Markup"] = df["Avg_Price"] - df["Wholesale Price (RMB/kg)"]
        df["Price_Markup_Pct"] = (df["Price_Markup"] / np.maximum(df["Wholesale Price (RMB/kg)"], 0.1)) * 100
        df["Avg_Price_Change"] = np.random.normal(0.02, 0.01, len(df))
        df["Wholesale_Price_Change"] = np.random.normal(0.015, 0.008, len(df))
        
        # ===== LAG FEATURES =====
        
        lag_periods = [1, 7, 14, 30]
        np.random.seed(42)
        
        for lag in lag_periods:
            price_lag_noise = np.random.normal(1.0, 0.05, len(df))
            quantity_lag_noise = np.random.normal(1.0, 0.08, len(df))
            df[f"Avg_Price_Lag_{lag}"] = df["Avg_Price"] * price_lag_noise
            df[f"Total_Quantity_Lag_{lag}"] = df["Total_Quantity"] * quantity_lag_noise
            df[f"Revenue_Lag_{lag}"] = df[f"Avg_Price_Lag_{lag}"] * df[f"Total_Quantity_Lag_{lag}"]
        
        # ===== ROLLING WINDOW FEATURES =====
        
        windows = [7, 14, 30]
        
        for window in windows:
            ma_variation = 0.03
            df[f"Avg_Price_MA_{window}"] = df["Avg_Price"] * np.random.uniform(1-ma_variation, 1+ma_variation, len(df))
            df[f"Total_Quantity_MA_{window}"] = df["Total_Quantity"] * np.random.uniform(1-ma_variation, 1+ma_variation, len(df))
            df[f"Revenue_MA_{window}"] = df[f"Avg_Price_MA_{window}"] * df[f"Total_Quantity_MA_{window}"]
            df[f"Avg_Price_Std_{window}"] = df["Price_Volatility"]
            df[f"Total_Quantity_Std_{window}"] = df["Total_Quantity"] * 0.1
            df[f"Avg_Price_Min_{window}"] = df["Min_Price"]
            df[f"Avg_Price_Max_{window}"] = df["Max_Price"]
        
        # ===== CATEGORY FEATURES =====
        
        df["Category_Total_Quantity"] = df["Total_Quantity"] * np.random.uniform(3, 6, len(df))
        df["Category_Avg_Price"] = df["Avg_Price"] * np.random.uniform(0.9, 1.1, len(df))
        df["Category_Revenue"] = df["Category_Total_Quantity"] * df["Category_Avg_Price"]
        df["Item_Quantity_Share"] = df["Total_Quantity"] / np.maximum(df["Category_Total_Quantity"], 1)
        df["Item_Revenue_Share"] = df["Revenue"] / np.maximum(df["Category_Revenue"], 1)
        df["Price_Relative_to_Category"] = df["Avg_Price"] / np.maximum(df["Category_Avg_Price"], 0.1)
        df["Category Name_Encoded"] = np.random.randint(1, 4, len(df))
        
        # ===== LOSS RATE FEATURES - EXACT MATCH =====
        
        df["Effective_Supply"] = df["Total_Quantity"] * (1 - df["Loss Rate (%)"] / 100)
        df["Loss_Adjusted_Revenue"] = df["Effective_Supply"] * df["Avg_Price"]
        
        # Only create the loss rate categories the model expects
        df["Loss_Rate_Category_Medium"] = ((df["Loss Rate (%)"] > 5) & (df["Loss Rate (%)"] <= 15)).astype(int)
        df["Loss_Rate_Category_High"] = (df["Loss Rate (%)"] > 15).astype(int)
        df["Loss_Rate_Category_Very_High"] = (df["Loss Rate (%)"] > 25).astype(int)
        # NOTE: NOT creating Loss_Rate_Category_Low as model doesn't expect it
        
        # ===== INTERACTION FEATURES =====
        
        df["Price_Quantity_Interaction"] = df["Avg_Price"] * df["Total_Quantity"]
        df["Price_Volatility_Quantity"] = df["Price_Volatility"] * df["Total_Quantity"]
        df["Spring_Price"] = df["Avg_Price"] * (df["Season"] == "Spring").astype(int)
        df["Summer_Price"] = df["Avg_Price"] * (df["Season"] == "Summer").astype(int)
        df["Autumn_Price"] = df["Avg_Price"] * (df["Season"] == "Autumn").astype(int)
        df["Winter_Price"] = df["Avg_Price"] * (df["Season"] == "Winter").astype(int)
        df["Holiday_Demand"] = df["Total_Quantity"] * (df["IsNationalDay"] + df["IsLaborDay"])
        
        # ===== SEASONAL DUMMY VARIABLES - EXACT MATCH =====
        
        # Only create the season dummies the model expects (NOT Season_Autumn)
        df["Season_Spring"] = (df["Season"] == "Spring").astype(int)
        df["Season_Summer"] = (df["Season"] == "Summer").astype(int)
        df["Season_Winter"] = (df["Season"] == "Winter").astype(int)
        # NOTE: NOT creating Season_Autumn as model doesn't expect it
        
        # Clean up categorical columns
        df = df.drop("Season", axis=1, errors='ignore')
        
        # ===== FINAL CLEANUP =====
        
        df = df.fillna(0)
        
        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        # ===== EXACT FEATURE SELECTION - THE 90 FEATURES MODEL EXPECTS =====
        
        # These are the EXACT 90 features from your model debug output
        expected_features = [
            'Total_Quantity', 'Avg_Quantity', 'Transaction_Count', 'Avg_Price', 'Price_Volatility', 
            'Min_Price', 'Max_Price', 'Discount_Count', 'Revenue', 'Discount_Rate', 'Price_Range', 
            'Wholesale Price (RMB/kg)', 'Loss Rate (%)', 'Year', 'Month', 'Quarter', 'DayOfYear', 
            'DayOfWeek', 'WeekOfYear', 'IsWeekend', 'Month_Sin', 'Month_Cos', 'DayOfYear_Sin', 
            'DayOfYear_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos', 'IsNationalDay', 'IsLaborDay', 
            'Days_Since_Epoch', 'Retail_Wholesale_Ratio', 'Price_Markup', 'Price_Markup_Pct', 
            'Avg_Price_Change', 'Wholesale_Price_Change', 'Avg_Price_Lag_1', 'Total_Quantity_Lag_1', 
            'Revenue_Lag_1', 'Avg_Price_Lag_7', 'Total_Quantity_Lag_7', 'Revenue_Lag_7', 
            'Avg_Price_Lag_14', 'Total_Quantity_Lag_14', 'Revenue_Lag_14', 'Avg_Price_Lag_30', 
            'Total_Quantity_Lag_30', 'Revenue_Lag_30', 'Avg_Price_MA_7', 'Total_Quantity_MA_7', 
            'Revenue_MA_7', 'Avg_Price_Std_7', 'Total_Quantity_Std_7', 'Avg_Price_Min_7', 
            'Avg_Price_Max_7', 'Avg_Price_MA_14', 'Total_Quantity_MA_14', 'Revenue_MA_14', 
            'Avg_Price_Std_14', 'Total_Quantity_Std_14', 'Avg_Price_Min_14', 'Avg_Price_Max_14', 
            'Avg_Price_MA_30', 'Total_Quantity_MA_30', 'Revenue_MA_30', 'Avg_Price_Std_30', 
            'Total_Quantity_Std_30', 'Avg_Price_Min_30', 'Avg_Price_Max_30', 'Category Code', 
            'Category_Total_Quantity', 'Category_Avg_Price', 'Category_Revenue', 'Item_Quantity_Share', 
            'Item_Revenue_Share', 'Price_Relative_to_Category', 'Effective_Supply', 'Loss_Adjusted_Revenue', 
            'Price_Quantity_Interaction', 'Price_Volatility_Quantity', 'Spring_Price', 'Summer_Price', 
            'Autumn_Price', 'Winter_Price', 'Holiday_Demand', 'Season_Spring', 'Season_Summer', 
            'Season_Winter', 'Loss_Rate_Category_Medium', 'Loss_Rate_Category_High', 
            'Loss_Rate_Category_Very_High', 'Category Name_Encoded'
        ]
        
        # Ensure all expected features exist
        for feature in expected_features:
            if feature not in df.columns:
                print(f"⚠️ Creating missing feature: {feature}")
                if 'Price' in feature:
                    df[feature] = df["Avg_Price"].iloc[0] if len(df) > 0 else 15.0
                elif 'Quantity' in feature:
                    df[feature] = df["Total_Quantity"].iloc[0] if len(df) > 0 else 100.0
                elif 'Rate' in feature or '%' in feature:
                    df[feature] = 0.1
                else:
                    df[feature] = 0.0
        
        # Select only the expected features in the correct order
        df_final = df[expected_features].copy()
        
        print(f"✅ EXACT feature engineering complete: {df_final.shape[1]} features (expected: 90)")
        print(f"✅ Feature count match: {df_final.shape[1] == 90}")
        
        if df_final.shape[1] != 90:
            print(f"❌ Feature count mismatch! Generated {df_final.shape[1]}, expected 90")
        
        return df_final
        
    except Exception as e:
        print(f"❌ Error in exact feature engineering: {e}")
        # Return basic features if advanced engineering fails
        basic_features = ["Total_Quantity", "Avg_Price", "Transaction_Count", "Month", "DayOfWeek"]
        available_features = [f for f in basic_features if f in df.columns]
        return df[available_features] if available_features else df

def test_local_prediction(model_artifact, expected_features=None, test_file="data/example/basic_test.json"):
    """Test prediction locally with the FIXED feature engineering"""
    print("\n🧪 TESTING LOCAL PREDICTION")
    print("=" * 50)
    
    # Load test data
    try:
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        print(f"✅ Loaded test data from {test_file}")
        input_features = test_data['features']
        print(f"   Input features: {len(input_features)} features")
        print(f"   Sample: {list(input_features.keys())[:5]}...")
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return False, None
    
    # Create DataFrame
    df_input = pd.DataFrame([input_features])
    print(f"✅ Created input DataFrame: {df_input.shape}")
    
    # Apply the FIXED feature engineering
    try:
        df_engineered = create_exact_90_features_debug(df_input)
        print(f"✅ Feature engineering completed: {df_engineered.shape}")
        
        # Check if we have expected features
        if expected_features:
            missing_features = set(expected_features) - set(df_engineered.columns)
            extra_features = set(df_engineered.columns) - set(expected_features)
            
            print(f"📊 Feature comparison:")
            print(f"   Expected: {len(expected_features)} features")
            print(f"   Generated: {len(df_engineered.columns)} features")
            print(f"   Missing: {len(missing_features)} features")
            print(f"   Extra: {len(extra_features)} features")
            
            if missing_features:
                print(f"❌ Missing features: {list(missing_features)[:5]}...")
            if extra_features:
                print(f"⚠️ Extra features: {list(extra_features)[:5]}...")
            
            # If features match exactly, continue
            if len(missing_features) == 0 and len(extra_features) == 0:
                print("✅ Perfect feature match!")
            else:
                print("⚠️ Feature mismatch detected")
        
        # Make prediction
        if isinstance(model_artifact, dict):
            model = model_artifact['model']
            scaler = model_artifact.get('scaler')
        else:
            model = model_artifact
            scaler = None
        
        # Try prediction
        try:
            if scaler:
                print("🔧 Applying scaler...")
                X_scaled = scaler.transform(df_engineered)
                prediction = model.predict(X_scaled)
                print(f"✅ Prediction with scaling: {prediction}")
            else:
                print("🔧 Predicting without scaling...")
                prediction = model.predict(df_engineered)
                print(f"✅ Prediction without scaling: {prediction}")
            
            # Check prediction details
            print(f"📊 Prediction details:")
            print(f"   Type: {type(prediction)}")
            print(f"   Shape: {prediction.shape if hasattr(prediction, 'shape') else 'scalar'}")
            print(f"   Value: {prediction}")
            
            if hasattr(prediction, '__len__') and len(prediction) > 0:
                print(f"   Min: {np.min(prediction)}")
                print(f"   Max: {np.max(prediction)}")
                print(f"   Mean: {np.mean(prediction)}")
            
            if np.all(prediction == 0):
                print("❌ PROBLEM: All predictions are 0!")
                
                # Debug zero predictions
                print(f"\n🔍 DEBUGGING ZERO PREDICTIONS:")
                print(f"   Input data range: {df_engineered.min().min()} to {df_engineered.max().max()}")
                print(f"   Any NaN values: {df_engineered.isnull().sum().sum()}")
                print(f"   Any infinite values: {np.isinf(df_engineered.values).sum()}")
                
                # Check if all features are zero
                zero_features = (df_engineered == 0).all()
                if zero_features.any():
                    zero_list = zero_features[zero_features].index.tolist()
                    print(f"   Features that are all zero: {zero_list[:10]}...")
                
                return False, prediction
            else:
                print("✅ Predictions look good!")
                return True, prediction
                
        except Exception as pred_error:
            print(f"❌ Prediction failed: {pred_error}")
            return False, None
            
    except Exception as fe_error:
        print(f"❌ Feature engineering failed: {fe_error}")
        return False, None

def check_endpoint_logs(endpoint_name, region="us-east-1"):
    """Check CloudWatch logs for the endpoint"""
    print(f"\n📋 CHECKING ENDPOINT LOGS")
    print("=" * 50)
    
    try:
        # Get CloudWatch logs
        logs_client = boto3.client('logs', region_name=region)
        log_group = f"/aws/sagemaker/Endpoints/{endpoint_name}"
        
        # Get recent log events
        response = logs_client.describe_log_streams(
            logGroupName=log_group,
            orderBy='LastEventTime',
            descending=True,
            limit=3
        )
        
        if response['logStreams']:
            print(f"✅ Found {len(response['logStreams'])} log streams for {endpoint_name}")
            
            for i, stream in enumerate(response['logStreams'][:2]):  # Check first 2 streams
                stream_name = stream['logStreamName']
                print(f"\n📄 Log Stream {i+1}: {stream_name}")
                
                try:
                    events_response = logs_client.get_log_events(
                        logGroupName=log_group,
                        logStreamName=stream_name,
                        limit=10,
                        startFromHead=False
                    )
                    
                    for event in events_response['events'][-5:]:  # Last 5 events
                        timestamp = pd.to_datetime(event['timestamp'], unit='ms')
                        message = event['message'].strip()
                        if "ERROR" in message.upper() or "CRITICAL" in message.upper() or "TRACEBACK" in message.upper():
                            print(f"❌ {timestamp}: {message}")
                        elif "SUCCESS" in message.upper() or "LOADED" in message.upper():
                            print(f"✅ {timestamp}: {message}")
                        else:
                            print(f"ℹ️  {timestamp}: {message}")
                
                except Exception as e:
                    print(f"⚠️ Error reading stream {stream_name}: {e}")
            
        else:
            print("❌ No log streams found")
            print(f"💡 Check manually: AWS Console → CloudWatch → Log Groups → {log_group}")
            
    except Exception as e:
        print(f"❌ Error checking logs: {e}")
        print("💡 Make sure the endpoint name is correct and you have CloudWatch permissions")

def test_endpoint_with_detailed_response(endpoint_name, test_file="data/example/basic_test.json", region="us-east-1"):
    """Test endpoint and show detailed response"""
    print(f"\n🧪 TESTING ENDPOINT: {endpoint_name}")
    print("=" * 50)
    
    try:
        # Initialize SageMaker session
        sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
        
        # Create predictor
        predictor = Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
        
        # Load test data
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        print(f"✅ Loaded test data from {test_file}")
        print(f"   Input: {test_data}")
        
        # Make prediction
        result = predictor.predict(test_data)
        
        print(f"✅ Endpoint response:")
        print(json.dumps(result, indent=2))
        
        # Analyze response
        if 'predictions' in result:
            predictions = result['predictions']
            print(f"\n📊 Prediction Analysis:")
            print(f"   Type: {type(predictions)}")
            print(f"   Length: {len(predictions) if isinstance(predictions, list) else 'scalar'}")
            print(f"   Values: {predictions}")
            
            if isinstance(predictions, list) and len(predictions) > 0:
                pred_array = np.array(predictions)
                print(f"   Min: {np.min(pred_array)}")
                print(f"   Max: {np.max(pred_array)}")
                print(f"   Mean: {np.mean(pred_array)}")
                
                if np.all(pred_array == 0):
                    print("❌ PROBLEM: All predictions are 0!")
                    return False
                else:
                    print("✅ Predictions have variation")
                    return True
            else:
                print("❌ PROBLEM: No predictions in response")
                return False
        elif 'error_message' in result:
            print(f"❌ ENDPOINT ERROR: {result['error_message']}")
            return False
        else:
            print("❌ PROBLEM: Unexpected response format")
            return False
            
    except Exception as e:
        print(f"❌ Endpoint test failed: {e}")
        if "not found" in str(e).lower():
            print(f"💡 Endpoint '{endpoint_name}' may not exist or may be in a different region")
            print(f"💡 Try: aws sagemaker list-endpoints --region {region}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Debug SageMaker Endpoint Issues')
    parser.add_argument('--endpoint-name', required=True, help='SageMaker endpoint name')
    parser.add_argument('--model-path', default='models/best_model.pkl', help='Path to local model file')
    parser.add_argument('--test-data', default='data/example/basic_test.json', help='Path to test data JSON file')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--skip-local', action='store_true', help='Skip local model testing')
    parser.add_argument('--skip-endpoint', action='store_true', help='Skip endpoint testing')
    parser.add_argument('--skip-logs', action='store_true', help='Skip log checking')
    
    args = parser.parse_args()
    
    print("🔍 SAGEMAKER ENDPOINT DEBUGGER - FIXED VERSION")
    print("=" * 70)
    print(f"Endpoint: {args.endpoint_name}")
    print(f"Model: {args.model_path}")
    print(f"Test Data: {args.test_data}")
    print(f"Region: {args.region}")
    print()
    
    local_success = True
    endpoint_success = True
    
    # Step 1: Debug local model
    if not args.skip_local:
        expected_features, model_artifact = debug_local_model(args.model_path)
        
        if model_artifact is None:
            print("❌ Cannot proceed without loading the model")
            return
        
        # Test local prediction
        local_success, local_prediction = test_local_prediction(model_artifact, expected_features, args.test_data)
    
    # Step 2: Test endpoint
    if not args.skip_endpoint:
        endpoint_success = test_endpoint_with_detailed_response(args.endpoint_name, args.test_data, args.region)
    
    # Step 3: Check logs
    if not args.skip_logs:
        check_endpoint_logs(args.endpoint_name, args.region)
    
    # Summary
    print(f"\n📋 DIAGNOSTIC SUMMARY")
    print("=" * 40)
    print(f"Local model test: {'✅ PASS' if local_success else '❌ FAIL'}")
    print(f"Endpoint test: {'✅ PASS' if endpoint_success else '❌ FAIL'}")
    
    if not local_success and not args.skip_local:
        print("\n🔧 ISSUE IS WITH LOCAL MODEL:")
        print("   - Check model training process")
        print("   - Verify target variable is not all zeros")
        print("   - Check feature scaling/transformation")
        print("   - Retrain model if necessary")
    elif not endpoint_success and not args.skip_endpoint:
        print("\n🔧 ISSUE IS WITH ENDPOINT:")
        print("   - Check CloudWatch logs for errors")
        print("   - Redeploy with fixed inference script")
        print("   - Verify endpoint name and region")
        print(f"   - Try: python fixed_sagemaker_deploy.py --config config.yaml --action deploy --model-path {args.model_path} --model-name chinese-produce-forecaster --endpoint-name {args.endpoint_name}-fixed")
    else:
        print("\n✅ DIAGNOSTICS COMPLETED")
        print("   Check the detailed output above for specific issues")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Deploy fixed endpoint: python fixed_sagemaker_deploy.py --config config.yaml --action deploy --model-path {args.model_path} --model-name chinese-produce-forecaster --endpoint-name {args.endpoint_name}-fixed")
    print(f"   2. Test fixed endpoint: python fixed_sagemaker_deploy.py --config config.yaml --action test --endpoint-name {args.endpoint_name}-fixed --test-data {args.test_data}")
    print(f"   3. Delete old endpoint: python fixed_sagemaker_deploy.py --config config.yaml --action delete --endpoint-name {args.endpoint_name}")

if __name__ == "__main__":
    main()