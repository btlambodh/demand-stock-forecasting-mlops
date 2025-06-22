#!/usr/bin/env python3
"""
SageMaker Endpoint Debugger
Diagnose why predictions are returning 0

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

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

def test_local_prediction(model_artifact, expected_features=None):
    """Test prediction locally with the same feature engineering"""
    print("\n🧪 TESTING LOCAL PREDICTION")
    print("=" * 50)
    
    # Load test data
    test_file = "data/example/high_volume_test.json"
    try:
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        print(f"✅ Loaded test data from {test_file}")
        input_features = test_data['features']
        print(f"   Input features: {len(input_features)} features")
        print(f"   Sample: {list(input_features.keys())[:5]}...")
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Create DataFrame
    df_input = pd.DataFrame([input_features])
    print(f"✅ Created input DataFrame: {df_input.shape}")
    
    # Apply the same feature engineering as in inference script
    try:
        df_engineered = create_comprehensive_features_from_basic_input(df_input)
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
                print(f"❌ Missing features: {list(missing_features)[:10]}...")
            if extra_features:
                print(f"⚠️ Extra features: {list(extra_features)[:10]}...")
        
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
            print(f"   Min: {np.min(prediction)}")
            print(f"   Max: {np.max(prediction)}")
            
            if np.all(prediction == 0):
                print("❌ PROBLEM: All predictions are 0!")
                
                # Check input data
                print(f"\n🔍 DEBUGGING ZERO PREDICTIONS:")
                print(f"   Input data range: {df_engineered.min().min()} to {df_engineered.max().max()}")
                print(f"   Any NaN values: {df_engineered.isnull().sum().sum()}")
                print(f"   Any infinite values: {np.isinf(df_engineered.values).sum()}")
                
                # Check if all features are zero
                zero_features = (df_engineered == 0).all()
                if zero_features.any():
                    print(f"   Features that are all zero: {zero_features[zero_features].index.tolist()}")
                
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

def create_comprehensive_features_from_basic_input(df_input):
    """Copy of the feature engineering function from inference script"""
    df = df_input.copy()
    
    # Ensure all basic features exist with sensible defaults
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
    
    # Derive additional basic features if missing
    if "Revenue" not in df_input.columns:
        df["Revenue"] = df["Total_Quantity"] * df["Avg_Price"]
    if "Price_Range" not in df_input.columns:
        df["Price_Range"] = df["Max_Price"] - df["Min_Price"]
    if "Discount_Rate" not in df_input.columns:
        df["Discount_Rate"] = df["Discount_Count"] / np.maximum(df["Transaction_Count"], 1)
    
    # TEMPORAL FEATURES
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["DayOfYear_Sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365)
    df["DayOfYear_Cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365)
    df["DayOfWeek_Sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_Cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    
    # Chinese holidays
    df["IsNationalDay"] = ((df["Month"] == 10) & (df["DayOfYear"].between(274, 280))).astype(int)
    df["IsLaborDay"] = ((df["Month"] == 5) & (df["DayOfYear"].between(121, 125))).astype(int)
    df["IsSpringFestival"] = ((df["Month"] == 2) & (df["DayOfYear"].between(32, 46))).astype(int)
    
    # Agricultural seasons
    season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 
                 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 
                 10: "Autumn", 11: "Autumn"}
    df["Season"] = df["Month"].map(season_map).fillna("Summer")
    
    # Days since epoch for trend
    epoch_date = pd.to_datetime("2020-01-01")
    current_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(df["DayOfYear"] - 1, unit="D")
    df["Days_Since_Epoch"] = (current_date - epoch_date).dt.days
    
    # PRICE FEATURES
    df["Retail_Wholesale_Ratio"] = df["Avg_Price"] / np.maximum(df["Wholesale Price (RMB/kg)"], 0.1)
    df["Price_Markup"] = df["Avg_Price"] - df["Wholesale Price (RMB/kg)"]
    df["Price_Markup_Pct"] = (df["Price_Markup"] / np.maximum(df["Wholesale Price (RMB/kg)"], 0.1)) * 100
    df["Avg_Price_Change"] = np.random.normal(0.02, 0.01, len(df))
    df["Wholesale_Price_Change"] = np.random.normal(0.015, 0.008, len(df))
    
    # LAG FEATURES - simulated for inference
    lag_periods = [1, 7, 14, 30]
    np.random.seed(42)
    
    for lag in lag_periods:
        price_lag_noise = np.random.normal(1.0, 0.05, len(df))
        quantity_lag_noise = np.random.normal(1.0, 0.08, len(df))
        df[f"Avg_Price_Lag_{lag}"] = df["Avg_Price"] * price_lag_noise
        df[f"Total_Quantity_Lag_{lag}"] = df["Total_Quantity"] * quantity_lag_noise
        df[f"Revenue_Lag_{lag}"] = df[f"Avg_Price_Lag_{lag}"] * df[f"Total_Quantity_Lag_{lag}"]
    
    # ROLLING WINDOW FEATURES - simulated for inference
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
    
    # CATEGORY FEATURES - simulated
    df["Category_Total_Quantity"] = df["Total_Quantity"] * np.random.uniform(3, 6, len(df))
    df["Category_Avg_Price"] = df["Avg_Price"] * np.random.uniform(0.9, 1.1, len(df))
    df["Category_Revenue"] = df["Category_Total_Quantity"] * df["Category_Avg_Price"]
    df["Item_Quantity_Share"] = df["Total_Quantity"] / np.maximum(df["Category_Total_Quantity"], 1)
    df["Item_Revenue_Share"] = df["Revenue"] / np.maximum(df["Category_Revenue"], 1)
    df["Price_Relative_to_Category"] = df["Avg_Price"] / np.maximum(df["Category_Avg_Price"], 0.1)
    df["Category Name_Encoded"] = np.random.randint(1, 4, len(df))
    
    # LOSS RATE FEATURES
    df["Effective_Supply"] = df["Total_Quantity"] * (1 - df["Loss Rate (%)"] / 100)
    df["Loss_Adjusted_Revenue"] = df["Effective_Supply"] * df["Avg_Price"]
    df["Loss_Rate_Category_High"] = (df["Loss Rate (%)"] > 15).astype(int)
    df["Loss_Rate_Category_Low"] = (df["Loss Rate (%)"] <= 5).astype(int)
    df["Loss_Rate_Category_Medium"] = ((df["Loss Rate (%)"] > 5) & (df["Loss Rate (%)"] <= 15)).astype(int)
    df["Loss_Rate_Category_Very_High"] = (df["Loss Rate (%)"] > 25).astype(int)
    
    # INTERACTION FEATURES
    df["Price_Quantity_Interaction"] = df["Avg_Price"] * df["Total_Quantity"]
    df["Price_Volatility_Quantity"] = df["Price_Volatility"] * df["Total_Quantity"]
    df["Spring_Price"] = df["Avg_Price"] * (df["Season"] == "Spring").astype(int)
    df["Summer_Price"] = df["Avg_Price"] * (df["Season"] == "Summer").astype(int)
    df["Autumn_Price"] = df["Avg_Price"] * (df["Season"] == "Autumn").astype(int)
    df["Winter_Price"] = df["Avg_Price"] * (df["Season"] == "Winter").astype(int)
    df["Holiday_Demand"] = df["Total_Quantity"] * (df["IsNationalDay"] + df["IsLaborDay"] + df["IsSpringFestival"])
    
    # SEASONAL DUMMY VARIABLES
    df["Season_Autumn"] = (df["Season"] == "Autumn").astype(int)
    df["Season_Spring"] = (df["Season"] == "Spring").astype(int)
    df["Season_Summer"] = (df["Season"] == "Summer").astype(int)
    df["Season_Winter"] = (df["Season"] == "Winter").astype(int)
    
    # Clean up categorical columns
    categorical_to_remove = ["Season"]
    for col in categorical_to_remove:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Final cleanup
    df = df.fillna(0)
    
    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

def check_endpoint_logs(endpoint_name="produce-forecast-staging"):
    """Check CloudWatch logs for the endpoint"""
    print(f"\n📋 CHECKING ENDPOINT LOGS")
    print("=" * 50)
    
    try:
        # Get CloudWatch logs
        logs_client = boto3.client('logs', region_name='us-east-1')
        log_group = f"/aws/sagemaker/Endpoints/{endpoint_name}"
        
        # Get recent log events
        response = logs_client.describe_log_streams(
            logGroupName=log_group,
            orderBy='LastEventTime',
            descending=True,
            limit=1
        )
        
        if response['logStreams']:
            stream_name = response['logStreams'][0]['logStreamName']
            
            events_response = logs_client.get_log_events(
                logGroupName=log_group,
                logStreamName=stream_name,
                limit=20,
                startFromHead=False
            )
            
            print(f"✅ Recent log entries from {stream_name}:")
            for event in events_response['events'][-10:]:  # Last 10 events
                timestamp = pd.to_datetime(event['timestamp'], unit='ms')
                message = event['message'].strip()
                print(f"   {timestamp}: {message}")
        else:
            print("❌ No log streams found")
            
    except Exception as e:
        print(f"❌ Error checking logs: {e}")
        print("💡 Check manually: AWS Console → CloudWatch → Log Groups → /aws/sagemaker/Endpoints/")

def test_endpoint_with_detailed_response(endpoint_name="produce-forecast-staging"):
    """Test endpoint and show detailed response"""
    print(f"\n🧪 TESTING ENDPOINT WITH DETAILED RESPONSE")
    print("=" * 50)
    
    try:
        # Initialize SageMaker session
        sagemaker_session = sagemaker.Session()
        
        # Create predictor
        predictor = Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
        
        # Load test data
        with open("data/example/high_volume_test.json", 'r') as f:
            test_data = json.load(f)
        
        print(f"✅ Loaded test data")
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
        else:
            print("❌ PROBLEM: No 'predictions' key in response")
            return False
            
    except Exception as e:
        print(f"❌ Endpoint test failed: {e}")
        return False

def main():
    print("🔍 SAGEMAKER ENDPOINT DEBUGGER")
    print("=" * 60)
    print("Diagnosing why predictions are returning 0")
    print()
    
    # Step 1: Debug local model
    expected_features, model_artifact = debug_local_model()
    
    if model_artifact is None:
        print("❌ Cannot proceed without loading the model")
        return
    
    # Step 2: Test local prediction
    local_success, local_prediction = test_local_prediction(model_artifact, expected_features)
    
    # Step 3: Test endpoint
    endpoint_success = test_endpoint_with_detailed_response()
    
    # Step 4: Check logs
    check_endpoint_logs()
    
    # Summary
    print(f"\n📋 DIAGNOSTIC SUMMARY")
    print("=" * 40)
    print(f"Local model test: {'✅ PASS' if local_success else '❌ FAIL'}")
    print(f"Endpoint test: {'✅ PASS' if endpoint_success else '❌ FAIL'}")
    
    if not local_success:
        print("\n🔧 ISSUE IS WITH LOCAL MODEL:")
        print("   - Check model training process")
        print("   - Verify target variable is not all zeros")
        print("   - Check feature scaling/transformation")
    elif not endpoint_success:
        print("\n🔧 ISSUE IS WITH ENDPOINT:")
        print("   - Check CloudWatch logs for errors")
        print("   - Verify inference script feature engineering")
        print("   - Check model serialization/deserialization")
    else:
        print("\n✅ BOTH LOCAL AND ENDPOINT TESTS PASSED")
        print("   The issue might be resolved!")

if __name__ == "__main__":
    main()