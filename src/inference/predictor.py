#!/usr/bin/env python3
"""
Model Predictor for Chinese Produce Market Forecasting
Handles model loading, preprocessing, and prediction logic

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings

import yaml
import pandas as pd
import numpy as np
import joblib
import boto3
from sklearn.preprocessing import StandardScaler
import mlflow
from mlflow.tracking import MlflowClient

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelPredictor:
    """Production model predictor with comprehensive preprocessing and prediction logic"""
    
    def __init__(self, config_path: str):
        """Initialize predictor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.aws_config = self.config['aws']
        self.model_config = self.config['models']
        
        # Initialize AWS clients
        self.s3_client = boto3.client('s3', region_name=self.aws_config['region'])
        
        # Initialize MLflow
        self.setup_mlflow()
        self.mlflow_client = MlflowClient()
        
        # Model storage
        self.loaded_models = {}
        self.feature_columns = []
        self.feature_metadata = {}
        
        # Performance tracking
        self.prediction_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'average_latency': 0.0,
            'last_prediction_time': None
        }
        
        logger.info("ModelPredictor initialized successfully")

    def setup_mlflow(self):
        """Setup MLflow connection"""
        try:
            s3_bucket = self.aws_config['s3']['bucket_name']
            mlflow_uri = f"s3://{s3_bucket}/mlflow"
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("chinese-produce-forecasting")
            logger.info(f"MLflow configured: {mlflow_uri}")
        except Exception as e:
            logger.warning(f"MLflow setup failed, using local tracking: {e}")
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("chinese-produce-forecasting")

    def load_production_models(self) -> Dict[str, Any]:
        """Load all production models from MLflow registry"""
        logger.info("Loading production models from MLflow registry")
        
        models = {}
        
        try:
            # Get all registered models
            registered_models = self.mlflow_client.list_registered_models()
            
            for reg_model in registered_models:
                model_name = reg_model.name
                
                # Skip if not a produce forecasting model
                if 'chinese_produce' not in model_name.lower():
                    continue
                
                try:
                    # Get production version
                    prod_versions = self.mlflow_client.get_latest_versions(
                        model_name, stages=["Production"]
                    )
                    
                    if prod_versions:
                        prod_version = prod_versions[0]
                        model_info = self.load_model_from_mlflow(prod_version)
                        
                        if model_info:
                            models[model_name] = model_info
                            logger.info(f"Loaded production model: {model_name} v{prod_version.version}")
                    
                    # If no production model, try staging
                    else:
                        staging_versions = self.mlflow_client.get_latest_versions(
                            model_name, stages=["Staging"]
                        )
                        
                        if staging_versions:
                            staging_version = staging_versions[0]
                            model_info = self.load_model_from_mlflow(staging_version)
                            
                            if model_info:
                                models[model_name] = model_info
                                logger.info(f"Loaded staging model: {model_name} v{staging_version.version}")
                
                except Exception as e:
                    logger.error(f"Error loading model {model_name}: {e}")
                    continue
            
            # Set best model if available
            if models:
                self.select_best_model(models)
            
            # Load feature metadata
            self.load_feature_metadata()
            
            self.loaded_models = models
            logger.info(f"Successfully loaded {len(models)} production models")
            
        except Exception as e:
            logger.error(f"Error loading models from MLflow: {e}")
            # Fallback to local models
            models = self.load_local_models()
        
        return models

    def load_model_from_mlflow(self, model_version) -> Optional[Dict[str, Any]]:
        """Load a specific model version from MLflow"""
        try:
            # Get model URI
            model_uri = f"models:/{model_version.name}/{model_version.version}"
            
            # Load model
            model = mlflow.sklearn.load_model(model_uri)
            
            # Get run details for metadata
            run = self.mlflow_client.get_run(model_version.run_id)
            
            model_info = {
                'model': model,
                'scaler': None,  # Will be loaded if available
                'version': model_version.version,
                'run_id': model_version.run_id,
                'stage': model_version.current_stage,
                'metrics': run.data.metrics,
                'params': run.data.params,
                'tags': run.data.tags,
                'model_type': run.data.tags.get('model_type', 'unknown'),
                'has_scaler': run.data.tags.get('has_scaler', 'false').lower() == 'true'
            }
            
            # Try to load scaler if it exists
            if model_info['has_scaler']:
                try:
                    # Download complete artifact from MLflow
                    artifact_path = self.mlflow_client.download_artifacts(
                        model_version.run_id, 
                        path="",
                        dst_path="/tmp"
                    )
                    
                    # Look for complete artifact file
                    for root, dirs, files in os.walk(artifact_path):
                        for file in files:
                            if file.endswith('_complete_artifact.pkl'):
                                complete_artifact_path = os.path.join(root, file)
                                complete_artifact = joblib.load(complete_artifact_path)
                                
                                if isinstance(complete_artifact, dict) and 'scaler' in complete_artifact:
                                    model_info['scaler'] = complete_artifact['scaler']
                                    logger.info(f"Loaded scaler for {model_version.name}")
                                break
                        
                except Exception as e:
                    logger.warning(f"Could not load scaler for {model_version.name}: {e}")
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {e}")
            return None

    def load_local_models(self) -> Dict[str, Any]:
        """Fallback: Load models from local directory"""
        logger.info("Loading models from local directory")
        
        models = {}
        models_dir = "models"
        
        if not os.path.exists(models_dir):
            logger.warning("No local models directory found")
            return models
        
        try:
            # Load evaluation results
            eval_file = os.path.join(models_dir, "evaluation.json")
            if os.path.exists(eval_file):
                with open(eval_file, 'r') as f:
                    evaluation_data = json.load(f)
            else:
                evaluation_data = {}
            
            # Load each model file
            for filename in os.listdir(models_dir):
                if filename.endswith('_model.pkl'):
                    model_name = filename.replace('_model.pkl', '')
                    model_path = os.path.join(models_dir, filename)
                    
                    try:
                        model_artifact = joblib.load(model_path)
                        
                        # Extract model and scaler
                        if isinstance(model_artifact, dict):
                            model = model_artifact.get('model')
                            scaler = model_artifact.get('scaler')
                        else:
                            model = model_artifact
                            scaler = None
                        
                        model_info = {
                            'model': model,
                            'scaler': scaler,
                            'version': 'local',
                            'stage': 'production',
                            'metrics': evaluation_data.get(model_name, {}),
                            'model_type': type(model).__name__,
                            'has_scaler': scaler is not None,
                            'source': 'local'
                        }
                        
                        models[f"chinese_produce_{model_name}"] = model_info
                        logger.info(f"Loaded local model: {model_name}")
                        
                    except Exception as e:
                        logger.error(f"Error loading local model {model_name}: {e}")
            
            # Set best model
            if models:
                self.select_best_model(models)
            
        except Exception as e:
            logger.error(f"Error loading local models: {e}")
        
        return models

    def select_best_model(self, models: Dict[str, Any]):
        """Select the best model based on performance metrics"""
        try:
            best_model_name = None
            best_mape = float('inf')
            
            for model_name, model_info in models.items():
                metrics = model_info.get('metrics', {})
                
                # Try different MAPE metric names
                mape = None
                for mape_key in ['test_mape', 'val_mape', 'validation_mape']:
                    if mape_key in metrics:
                        mape = metrics[mape_key]
                        break
                
                if mape is not None and mape < best_mape:
                    best_mape = mape
                    best_model_name = model_name
            
            if best_model_name:
                # Add best model alias
                models['best_model'] = models[best_model_name].copy()
                models['best_model']['alias'] = 'best_model'
                logger.info(f"Best model selected: {best_model_name} (MAPE: {best_mape:.3f}%)")
            
        except Exception as e:
            logger.error(f"Error selecting best model: {e}")

    def load_feature_metadata(self):
        """Load feature metadata for preprocessing"""
        try:
            # Try to load from models directory
            feature_info_file = "models/feature_info.json"
            if os.path.exists(feature_info_file):
                with open(feature_info_file, 'r') as f:
                    self.feature_metadata = json.load(f)
                    self.feature_columns = self.feature_metadata.get('feature_columns', [])
                    logger.info(f"Loaded feature metadata: {len(self.feature_columns)} features")
                    return
            
            # Try to get from S3
            bucket = self.aws_config['s3']['bucket_name']
            s3_key = "models/feature_info.json"
            
            response = self.s3_client.get_object(Bucket=bucket, Key=s3_key)
            self.feature_metadata = json.loads(response['Body'].read())
            self.feature_columns = self.feature_metadata.get('feature_columns', [])
            logger.info(f"Loaded feature metadata from S3: {len(self.feature_columns)} features")
            
        except Exception as e:
            logger.warning(f"Could not load feature metadata: {e}")
            # Set default feature columns based on common features
            self.feature_columns = self.get_default_feature_columns()

    def get_default_feature_columns(self) -> List[str]:
        """Get default feature columns if metadata is not available"""
        return [
            'Total_Quantity', 'Avg_Price', 'Transaction_Count', 'Price_Volatility',
            'Min_Price', 'Max_Price', 'Revenue', 'Discount_Rate', 'Price_Range',
            'Month', 'Quarter', 'DayOfYear', 'DayOfWeek', 'WeekOfYear', 'IsWeekend',
            'Month_Sin', 'Month_Cos', 'DayOfYear_Sin', 'DayOfYear_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos',
            'Retail_Wholesale_Ratio', 'Price_Markup', 'Price_Markup_Pct',
            'Avg_Price_Lag_1', 'Avg_Price_Lag_7', 'Total_Quantity_Lag_1', 'Total_Quantity_Lag_7',
            'Avg_Price_MA_7', 'Avg_Price_MA_30', 'Total_Quantity_MA_7',
            'Category_Avg_Price', 'Item_Revenue_Share', 'Price_Relative_to_Category',
            'Loss_Rate_Percent'
        ]

    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for prediction"""
        logger.debug(f"Preprocessing features: {df.shape}")
        
        try:
            # Make a copy to avoid modifying original
            processed_df = df.copy()
            
            # Add derived features if missing
            processed_df = self.add_derived_features(processed_df)
            
            # Handle missing values
            processed_df = self.handle_missing_values(processed_df)
            
            # Ensure all required features are present
            processed_df = self.ensure_required_features(processed_df)
            
            # Remove any non-feature columns
            feature_df = self.select_feature_columns(processed_df)
            
            # Handle infinite values
            feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
            feature_df = feature_df.fillna(feature_df.median())
            
            logger.debug(f"Preprocessing completed: {feature_df.shape}")
            return feature_df
            
        except Exception as e:
            logger.error(f"Error in feature preprocessing: {e}")
            raise

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features if missing"""
        derived_df = df.copy()
        
        # Revenue calculation
        if 'Revenue' not in derived_df.columns and 'Total_Quantity' in derived_df.columns and 'Avg_Price' in derived_df.columns:
            derived_df['Revenue'] = derived_df['Total_Quantity'] * derived_df['Avg_Price']
        
        # Discount rate calculation
        if 'Discount_Rate' not in derived_df.columns:
            derived_df['Discount_Rate'] = 0.0  # Default
        
        # Price range calculation
        if 'Price_Range' not in derived_df.columns and 'Max_Price' in derived_df.columns and 'Min_Price' in derived_df.columns:
            derived_df['Price_Range'] = derived_df['Max_Price'] - derived_df['Min_Price']
        
        # Cyclical features
        if 'Month' in derived_df.columns:
            if 'Month_Sin' not in derived_df.columns:
                derived_df['Month_Sin'] = np.sin(2 * np.pi * derived_df['Month'] / 12)
            if 'Month_Cos' not in derived_df.columns:
                derived_df['Month_Cos'] = np.cos(2 * np.pi * derived_df['Month'] / 12)
        
        if 'DayOfWeek' in derived_df.columns:
            if 'DayOfWeek_Sin' not in derived_df.columns:
                derived_df['DayOfWeek_Sin'] = np.sin(2 * np.pi * derived_df['DayOfWeek'] / 7)
            if 'DayOfWeek_Cos' not in derived_df.columns:
                derived_df['DayOfWeek_Cos'] = np.cos(2 * np.pi * derived_df['DayOfWeek'] / 7)
        
        # Day of year cyclical features
        if 'DayOfYear' not in derived_df.columns and 'Month' in derived_df.columns:
            # Approximate day of year from month
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            derived_df['DayOfYear'] = derived_df['Month'].apply(
                lambda m: sum(days_in_month[:m-1]) + 15 if m > 0 else 15
            )
        
        if 'DayOfYear' in derived_df.columns:
            if 'DayOfYear_Sin' not in derived_df.columns:
                derived_df['DayOfYear_Sin'] = np.sin(2 * np.pi * derived_df['DayOfYear'] / 365)
            if 'DayOfYear_Cos' not in derived_df.columns:
                derived_df['DayOfYear_Cos'] = np.cos(2 * np.pi * derived_df['DayOfYear'] / 365)
        
        # Quarter calculation
        if 'Quarter' not in derived_df.columns and 'Month' in derived_df.columns:
            derived_df['Quarter'] = ((derived_df['Month'] - 1) // 3) + 1
        
        # Week of year approximation
        if 'WeekOfYear' not in derived_df.columns and 'DayOfYear' in derived_df.columns:
            derived_df['WeekOfYear'] = (derived_df['DayOfYear'] / 7).astype(int) + 1
        
        return derived_df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies"""
        filled_df = df.copy()
        
        # Default values for common features
        default_values = {
            'Price_Volatility': 0.0,
            'Discount_Rate': 0.0,
            'Price_Range': 0.0,
            'Retail_Wholesale_Ratio': 1.0,
            'Price_Markup': 0.0,
            'Price_Markup_Pct': 0.0,
            'Item_Revenue_Share': 0.0,
            'Price_Relative_to_Category': 1.0,
            'Loss_Rate_Percent': 10.0,
            'IsWeekend': 0
        }
        
        # Fill with defaults
        for col, default_val in default_values.items():
            if col in filled_df.columns:
                filled_df[col] = filled_df[col].fillna(default_val)
        
        # Forward fill lag features
        lag_columns = [col for col in filled_df.columns if 'Lag_' in col]
        for col in lag_columns:
            if col in filled_df.columns:
                # Use current values as approximation for missing lags
                base_col = col.replace('_Lag_1', '').replace('_Lag_7', '').replace('_Lag_14', '').replace('_Lag_30', '')
                if base_col in filled_df.columns:
                    filled_df[col] = filled_df[col].fillna(filled_df[base_col])
        
        # Moving averages - use current values
        ma_columns = [col for col in filled_df.columns if '_MA_' in col]
        for col in ma_columns:
            if col in filled_df.columns:
                base_col = col.replace('_MA_7', '').replace('_MA_14', '').replace('_MA_30', '')
                if base_col in filled_df.columns:
                    filled_df[col] = filled_df[col].fillna(filled_df[base_col])
        
        # Fill remaining numeric columns with median
        numeric_cols = filled_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if filled_df[col].isnull().any():
                median_val = filled_df[col].median()
                if pd.isna(median_val):
                    # If median is NaN, use 0
                    filled_df[col] = filled_df[col].fillna(0)
                else:
                    filled_df[col] = filled_df[col].fillna(median_val)
        
        return filled_df

    def ensure_required_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required features are present"""
        enriched_df = df.copy()
        
        # Add missing features with default values
        for feature in self.feature_columns:
            if feature not in enriched_df.columns:
                # Set appropriate default based on feature type
                if 'Sin' in feature or 'Cos' in feature:
                    enriched_df[feature] = 0.0
                elif 'Ratio' in feature or 'Relative' in feature:
                    enriched_df[feature] = 1.0
                elif 'Rate' in feature or 'Share' in feature:
                    enriched_df[feature] = 0.0
                elif 'Lag_' in feature or '_MA_' in feature:
                    enriched_df[feature] = 0.0
                else:
                    enriched_df[feature] = 0.0
        
        return enriched_df

    def select_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select only the feature columns needed for prediction"""
        # Remove non-feature columns
        exclude_cols = [
            'Date', 'Item Code', 'Category Name', 'Category Code',
            'Item Name', 'DateTime', 'Time'
        ]
        
        available_cols = [col for col in df.columns if col not in exclude_cols]
        
        # If we have feature columns defined, use them
        if self.feature_columns:
            # Use intersection of available columns and expected features
            selected_cols = [col for col in self.feature_columns if col in available_cols]
            
            # Add any additional numeric columns that might be useful
            numeric_cols = df[available_cols].select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in selected_cols and col not in exclude_cols:
                    selected_cols.append(col)
        else:
            # Use all numeric columns
            selected_cols = df[available_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        return df[selected_cols]

    def predict(self, df: pd.DataFrame, model_name: str = "best_model") -> Dict[str, Any]:
        """Make predictions using specified model"""
        start_time = datetime.now()
        
        try:
            # Check if model is available
            if model_name not in self.loaded_models:
                available_models = list(self.loaded_models.keys())
                raise ValueError(f"Model '{model_name}' not found. Available: {available_models}")
            
            model_info = self.loaded_models[model_name]
            model = model_info['model']
            scaler = model_info.get('scaler')
            
            # Preprocess features
            processed_features = self.preprocess_features(df)
            
            # Apply scaling if required
            if scaler is not None:
                X_scaled = scaler.transform(processed_features)
                logger.debug("Applied feature scaling")
            else:
                X_scaled = processed_features.values
            
            # Make predictions
            predictions = model.predict(X_scaled)
            
            # Calculate confidence/uncertainty
            confidence = self.calculate_prediction_confidence(
                model, X_scaled, predictions, model_info
            )
            
            # Update stats
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds()
            
            self.prediction_stats['total_predictions'] += len(predictions)
            self.prediction_stats['successful_predictions'] += len(predictions)
            self.prediction_stats['last_prediction_time'] = end_time.isoformat()
            
            # Update average latency
            if self.prediction_stats['average_latency'] == 0:
                self.prediction_stats['average_latency'] = latency
            else:
                # Exponential moving average
                alpha = 0.1
                self.prediction_stats['average_latency'] = (
                    alpha * latency + (1 - alpha) * self.prediction_stats['average_latency']
                )
            
            result = {
                'predictions': predictions.tolist(),
                'confidence': confidence.tolist(),
                'model_used': model_name,
                'model_type': model_info.get('model_type', 'unknown'),
                'model_version': model_info.get('version', 'unknown'),
                'input_shape': list(processed_features.shape),
                'processing_time_seconds': latency,
                'timestamp': end_time.isoformat()
            }
            
            logger.info(f"Prediction completed", 
                       model=model_name,
                       predictions=len(predictions),
                       latency=latency)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            # Update error stats
            self.prediction_stats['total_predictions'] += len(df) if df is not None else 1
            raise

    def calculate_prediction_confidence(self, model, X_scaled, predictions, model_info) -> np.ndarray:
        """Calculate prediction confidence based on model type and performance"""
        try:
            # Get model performance metrics
            metrics = model_info.get('metrics', {})
            base_confidence = 1.0 - (metrics.get('val_mape', 20.0) / 100.0)  # Convert MAPE to confidence
            base_confidence = max(0.1, min(0.95, base_confidence))  # Clamp between 0.1 and 0.95
            
            # Model-specific confidence calculation
            if hasattr(model, 'predict_proba'):
                # Classification model (unlikely for price prediction, but handle it)
                proba = model.predict_proba(X_scaled)
                confidence = np.max(proba, axis=1)
            
            elif hasattr(model, 'decision_function'):
                # SVM or similar
                decision_scores = np.abs(model.decision_function(X_scaled))
                # Normalize to 0-1 range
                confidence = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-8)
                confidence = confidence * base_confidence
            
            elif hasattr(model, 'feature_importances_'):
                # Tree-based models
                n_samples = len(predictions)
                
                # Use feature importance variance as uncertainty indicator
                feature_importance_std = np.std(model.feature_importances_)
                uncertainty = min(0.3, feature_importance_std)  # Cap uncertainty
                
                confidence = np.full(n_samples, base_confidence - uncertainty)
            
            else:
                # Linear models or others - use base confidence
                confidence = np.full(len(predictions), base_confidence)
            
            # Ensure confidence is in valid range
            confidence = np.clip(confidence, 0.1, 0.95)
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            # Return default confidence
            return np.full(len(predictions), 0.8)

    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """Get information about loaded models"""
        if model_name:
            if model_name in self.loaded_models:
                info = self.loaded_models[model_name].copy()
                # Remove actual model objects for serialization
                info.pop('model', None)
                info.pop('scaler', None)
                return info
            else:
                return {}
        else:
            # Return info for all models
            all_info = {}
            for name, model_info in self.loaded_models.items():
                info = model_info.copy()
                info.pop('model', None)
                info.pop('scaler', None)
                all_info[name] = info
            return all_info

    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        return self.prediction_stats.copy()

    def validate_input(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate input data"""
        errors = []
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("Input DataFrame is empty")
            return False, errors
        
        # Check for required minimum features
        required_features = ['Avg_Price', 'Total_Quantity', 'Month', 'DayOfWeek']
        missing_required = [feat for feat in required_features if feat not in df.columns]
        
        if missing_required:
            errors.append(f"Missing required features: {missing_required}")
        
        # Check for invalid values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df.columns:
                # Check for negative prices
                if 'Price' in col and (df[col] < 0).any():
                    errors.append(f"Negative values found in {col}")
                
                # Check for negative quantities
                if 'Quantity' in col and (df[col] < 0).any():
                    errors.append(f"Negative values found in {col}")
                
                # Check for extreme values
                if df[col].abs().max() > 1e6:
                    errors.append(f"Extreme values found in {col}")
        
        # Check date-related features
        if 'Month' in df.columns:
            invalid_months = df[(df['Month'] < 1) | (df['Month'] > 12)]
            if not invalid_months.empty:
                errors.append("Invalid month values (must be 1-12)")
        
        if 'DayOfWeek' in df.columns:
            invalid_days = df[(df['DayOfWeek'] < 0) | (df['DayOfWeek'] > 6)]
            if not invalid_days.empty:
                errors.append("Invalid day of week values (must be 0-6)")
        
        return len(errors) == 0, errors


def main():
    """Test the predictor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Model Predictor')
    parser.add_argument('--config', default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--test-data', help='Path to test data CSV file')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = ModelPredictor(args.config)
        
        # Load models
        models = predictor.load_production_models()
        print(f"Loaded {len(models)} models: {list(models.keys())}")
        
        # Create test data if not provided
        if args.test_data and os.path.exists(args.test_data):
            test_df = pd.read_csv(args.test_data)
        else:
            # Create sample test data
            test_df = pd.DataFrame({
                'Total_Quantity': [150.5, 200.0, 80.3],
                'Avg_Price': [12.80, 15.20, 9.50],
                'Transaction_Count': [25, 30, 15],
                'Price_Volatility': [0.15, 0.20, 0.10],
                'Month': [6, 7, 8],
                'DayOfWeek': [2, 4, 1],
                'IsWeekend': [0, 0, 0]
            })
        
        # Validate input
        is_valid, errors = predictor.validate_input(test_df)
        if not is_valid:
            print(f"Input validation errors: {errors}")
            return
        
        # Make predictions
        if models:
            result = predictor.predict(test_df, "best_model")
            
            print("\nPrediction Results:")
            print(f"Model: {result['model_used']}")
            print(f"Predictions: {result['predictions']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Processing time: {result['processing_time_seconds']:.3f}s")
        else:
            print("No models available for prediction")
        
        # Print stats
        stats = predictor.get_prediction_stats()
        print(f"\nPrediction Stats: {stats}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()