#!/usr/bin/env python3
"""
Basic Model Training Script for Chinese Produce Market Forecasting
Now with comprehensive file logging

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import argparse
import logging
import os
import json
from datetime import datetime
from typing import Dict, Tuple
import sys

import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def setup_logging(model_version: str, log_dir: str = "logs") -> logging.Logger:
    """Set up comprehensive logging with file and console handlers"""
    
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('ModelTrainer')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler - detailed log
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"training_{model_version}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Create a summary log file
    summary_filename = f"training_summary_{model_version}_{timestamp}.log"
    summary_filepath = os.path.join(log_dir, summary_filename)
    
    summary_handler = logging.FileHandler(summary_filepath, mode='w', encoding='utf-8')
    summary_handler.setLevel(logging.WARNING)  # Only important messages
    summary_handler.setFormatter(simple_formatter)
    logger.addHandler(summary_handler)
    
    # Log the setup
    logger.info("="*60)
    logger.info("LOGGING SETUP COMPLETED")
    logger.info("="*60)
    logger.info(f"Model Version: {model_version}")
    logger.info(f"Detailed Log: {log_filepath}")
    logger.info(f"Summary Log: {summary_filepath}")
    logger.info(f"Log Directory: {os.path.abspath(log_dir)}")
    logger.info("="*60)
    
    return logger


class ModelTrainer:
    """Basic model training for produce price forecasting"""
    
    def __init__(self, config_path: str, model_version: str, logger: logging.Logger):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_version = model_version
        self.logger = logger

    def load_processed_data(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """Load processed feature data"""
        self.logger.info("Loading processed feature data...")
        
        data = {}
        
        train_path = os.path.join(data_path, 'train.parquet')
        val_path = os.path.join(data_path, 'validation.parquet') 
        test_path = os.path.join(data_path, 'test.parquet')
        
        if os.path.exists(train_path):
            data['train'] = pd.read_parquet(train_path)
            self.logger.info(f"Loaded training data: {data['train'].shape}")
            self.logger.debug(f"Training data columns: {list(data['train'].columns)}")
        
        if os.path.exists(val_path):
            data['validation'] = pd.read_parquet(val_path)
            self.logger.info(f"Loaded validation data: {data['validation'].shape}")
            
        if os.path.exists(test_path):
            data['test'] = pd.read_parquet(test_path)
            self.logger.info(f"Loaded test data: {data['test'].shape}")
        
        # Log data statistics
        for split_name, df in data.items():
            self.logger.debug(f"{split_name} data info:")
            self.logger.debug(f"  - Shape: {df.shape}")
            self.logger.debug(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            self.logger.debug(f"  - Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        return data

    def prepare_features_targets(self, df: pd.DataFrame, forecast_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and targets for training"""
        self.logger.info(f"Preparing features for {forecast_horizon}-day forecast...")
        
        # Target column for the specified horizon
        target_col = f'Avg_Price_Target_{forecast_horizon}d'
        
        if target_col not in df.columns:
            self.logger.error(f"Target column {target_col} not found in data")
            available_targets = [col for col in df.columns if 'Target' in col]
            self.logger.error(f"Available target columns: {available_targets}")
            raise ValueError(f"Target column {target_col} not found in data")
        
        # Exclude target columns and non-feature columns from features
        exclude_cols = [
            'Date', 'Item Code', 'Category Name'
        ] + [col for col in df.columns if 'Target' in col]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.logger.debug(f"Excluded columns: {exclude_cols}")
        self.logger.debug(f"Feature columns: {feature_cols[:10]}...")  # First 10 features
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Log initial shapes
        self.logger.debug(f"Initial X shape: {X.shape}, y shape: {y.shape}")
        
        # Remove rows with missing targets
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        self.logger.debug(f"After removing missing targets: X shape: {X.shape}, y shape: {y.shape}")
        
        # Handle infinite values
        inf_mask = X.isin([np.inf, -np.inf]).any(axis=1)
        if inf_mask.sum() > 0:
            self.logger.warning(f"Found {inf_mask.sum()} rows with infinite values, replacing with NaN")
            X = X.replace([np.inf, -np.inf], np.nan)
        
        # Check for missing values in features
        missing_counts = X.isnull().sum()
        if missing_counts.sum() > 0:
            self.logger.warning(f"Found missing values in features:")
            for col, count in missing_counts[missing_counts > 0].items():
                self.logger.warning(f"  {col}: {count} missing values")
        
        # Fill remaining missing values
        X = X.fillna(X.median())
        
        self.logger.info(f"Features prepared: {X.shape}, Target: {y.shape}")
        self.logger.info(f"Feature columns: {len(feature_cols)}")
        self.logger.warning(f"FEATURE PREPARATION COMPLETED - {len(feature_cols)} features ready")
        
        return X, y

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Train multiple models"""
        self.logger.warning("STARTING MODEL TRAINING PHASE")
        self.logger.info("Training models...")
        
        # Initialize models
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6, 
                learning_rate=0.1,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                self.logger.warning(f"TRAINING {name.upper()}...")
                start_time = datetime.now()
                
                # Scale features for linear models
                if name in ['linear_regression', 'ridge']:
                    self.logger.info(f"Scaling features for {name}")
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Predictions
                    train_pred = model.predict(X_train_scaled)
                    val_pred = model.predict(X_val_scaled)
                    
                    # Store scaler
                    model_artifact = {'model': model, 'scaler': scaler}
                else:
                    # Train model without scaling
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    train_pred = model.predict(X_train)
                    val_pred = model.predict(X_val)
                    
                    model_artifact = {'model': model}
                
                # Calculate training time
                training_time = (datetime.now() - start_time).total_seconds()
                self.logger.info(f"{name} training completed in {training_time:.2f} seconds")
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_train, train_pred, y_val, val_pred)
                
                # Feature importance (if available)
                importance = None
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    self.logger.debug(f"{name} feature importance calculated")
                elif hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_)
                    self.logger.debug(f"{name} coefficient importance calculated")
                
                results[name] = {
                    'model_artifact': model_artifact,
                    'metrics': metrics,
                    'feature_importance': importance,
                    'training_time': training_time
                }
                
                self.logger.warning(f"{name} - Validation MAPE: {metrics['val_mape']:.3f}%")
                self.logger.info(f"{name} - Validation RMSE: {metrics['val_rmse']:.3f}")
                self.logger.info(f"{name} - Validation R²: {metrics['val_r2']:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")
                self.logger.exception(f"Full traceback for {name}:")
                results[name] = {'error': str(e)}
        
        self.logger.warning("MODEL TRAINING PHASE COMPLETED")
        return results

    def calculate_metrics(self, y_train: pd.Series, train_pred: np.ndarray,
                         y_val: pd.Series, val_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        def mape(y_true, y_pred):
            """Mean Absolute Percentage Error"""
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        def smape(y_true, y_pred):
            """Symmetric Mean Absolute Percentage Error"""
            return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
        
        metrics = {
            # Training metrics
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_mse': mean_squared_error(y_train, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'train_mape': mape(y_train, train_pred),
            
            # Validation metrics
            'val_mae': mean_absolute_error(y_val, val_pred),
            'val_mse': mean_squared_error(y_val, val_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'val_r2': r2_score(y_val, val_pred),
            'val_mape': mape(y_val, val_pred),
            'val_smape': smape(y_val, val_pred)
        }
        
        return metrics

    def evaluate_on_test(self, results: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate models on test set"""
        self.logger.warning("STARTING TEST SET EVALUATION")
        self.logger.info("Evaluating models on test set...")
        
        test_results = {}
        
        for model_name, result in results.items():
            if 'error' in result:
                self.logger.warning(f"Skipping {model_name} due to training error")
                continue
                
            try:
                self.logger.info(f"Evaluating {model_name} on test set...")
                
                model_artifact = result['model_artifact']
                model = model_artifact['model']
                
                # Handle scaled models
                if 'scaler' in model_artifact:
                    scaler = model_artifact['scaler']
                    X_test_scaled = scaler.transform(X_test)
                    test_pred = model.predict(X_test_scaled)
                    self.logger.debug(f"{model_name}: Applied scaling for test predictions")
                else:
                    test_pred = model.predict(X_test)
                
                # Calculate test metrics
                def mape(y_true, y_pred):
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                
                test_metrics = {
                    'test_mae': mean_absolute_error(y_test, test_pred),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                    'test_r2': r2_score(y_test, test_pred),
                    'test_mape': mape(y_test, test_pred)
                }
                
                # Combine with existing metrics
                all_metrics = {**result.get('metrics', {}), **test_metrics}
                
                test_results[model_name] = {
                    'metrics': all_metrics,
                    'feature_importance': result.get('feature_importance', None),
                    'training_time': result.get('training_time', 0)
                }
                
                self.logger.warning(f"{model_name} - Test MAPE: {test_metrics['test_mape']:.3f}%")
                self.logger.info(f"{model_name} - Test RMSE: {test_metrics['test_rmse']:.3f}")
                self.logger.info(f"{model_name} - Test R²: {test_metrics['test_r2']:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
                self.logger.exception(f"Full traceback for {model_name} evaluation:")
                test_results[model_name] = {'error': str(e)}
        
        self.logger.warning("TEST SET EVALUATION COMPLETED")
        return test_results

    def save_models_and_results(self, results: Dict, evaluation_results: Dict, 
                               feature_cols: list, output_path: str) -> Dict[str, str]:
        """Save trained models and results"""
        self.logger.warning("SAVING MODELS AND RESULTS")
        self.logger.info("Saving models and results...")
        
        os.makedirs(output_path, exist_ok=True)
        
        saved_files = {}
        
        # Save models
        for name, result in results.items():
            if 'error' not in result:
                try:
                    model_artifact = result['model_artifact']
                    model_path = os.path.join(output_path, f'{name}_model.pkl')
                    joblib.dump(model_artifact, model_path)
                    saved_files[name] = model_path
                    
                    # Log model file size
                    file_size = os.path.getsize(model_path) / 1024**2  # MB
                    self.logger.info(f"Saved {name} model ({file_size:.2f} MB)")
                    
                except Exception as e:
                    self.logger.error(f"Error saving {name}: {e}")
        
        # Save evaluation results
        eval_data = {}
        for name, result in evaluation_results.items():
            if 'error' not in result:
                eval_data[name] = result['metrics']
        
        eval_path = os.path.join(output_path, 'evaluation.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_data, f, indent=2)
        saved_files['evaluation'] = eval_path
        self.logger.info(f"Saved evaluation results: {eval_path}")
        
        # Save feature information
        feature_info = {
            'feature_columns': feature_cols,
            'num_features': len(feature_cols),
            'model_version': self.model_version,
            'training_date': datetime.now().isoformat(),
            'training_summary': {
                'models_trained': len([r for r in results.values() if 'error' not in r]),
                'models_failed': len([r for r in results.values() if 'error' in r])
            }
        }
        
        feature_path = os.path.join(output_path, 'feature_info.json')
        with open(feature_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        saved_files['feature_info'] = feature_path
        self.logger.info(f"Saved feature info: {feature_path}")
        
        self.logger.warning(f"SAVED {len(saved_files)} FILES TO {output_path}")
        return saved_files

    def run_training_pipeline(self, data_path: str, output_path: str, 
                            forecast_horizon: int = 1) -> Dict:
        """Run complete training pipeline"""
        self.logger.warning("="*60)
        self.logger.warning("STARTING COMPLETE TRAINING PIPELINE")
        self.logger.warning("="*60)
        
        training_summary = {
            'model_version': self.model_version,
            'forecast_horizon': forecast_horizon,
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'models_trained': 0,
            'best_model': None,
            'best_performance': None
        }
        
        try:
            # Load data
            self.logger.info("Step 1: Loading processed data...")
            data = self.load_processed_data(data_path)
            
            if 'train' not in data or 'validation' not in data:
                raise ValueError("Training and validation data not found")
            
            # Prepare features
            self.logger.info("Step 2: Preparing features and targets...")
            X_train, y_train = self.prepare_features_targets(data['train'], forecast_horizon)
            X_val, y_val = self.prepare_features_targets(data['validation'], forecast_horizon)
            
            if 'test' in data:
                X_test, y_test = self.prepare_features_targets(data['test'], forecast_horizon)
            else:
                X_test, y_test = None, None
                self.logger.warning("No test data available")
            
            # Get feature columns
            feature_cols = list(X_train.columns)
            
            # Train models
            self.logger.info("Step 3: Training models...")
            results = self.train_models(X_train, y_train, X_val, y_val)
            
            # Count successful models
            successful_models = [name for name, result in results.items() if 'error' not in result]
            training_summary['models_trained'] = len(successful_models)
            
            # Evaluate on test set
            if X_test is not None and y_test is not None:
                self.logger.info("Step 4: Evaluating on test set...")
                evaluation_results = self.evaluate_on_test(results, X_test, y_test)
            else:
                # Use validation results if no test set
                self.logger.info("Step 4: Using validation results (no test set)...")
                evaluation_results = {}
                for name, result in results.items():
                    if 'error' not in result:
                        evaluation_results[name] = {
                            'metrics': result.get('metrics', {}),
                            'feature_importance': result.get('feature_importance', None),
                            'training_time': result.get('training_time', 0)
                        }
            
            # Select best model based on validation MAPE
            self.logger.info("Step 5: Selecting best model...")
            best_model_name = None
            best_mape = float('inf')
            
            for name, result in evaluation_results.items():
                if 'error' not in result:
                    val_mape = result.get('metrics', {}).get('val_mape', float('inf'))
                    if val_mape < best_mape:
                        best_mape = val_mape
                        best_model_name = name
            
            if best_model_name:
                training_summary['best_model'] = best_model_name
                training_summary['best_performance'] = evaluation_results[best_model_name]['metrics']
                training_summary['status'] = 'success'
                self.logger.warning(f"BEST MODEL SELECTED: {best_model_name} (MAPE: {best_mape:.3f}%)")
            
            # Save models and results
            self.logger.info("Step 6: Saving models and results...")
            saved_files = self.save_models_and_results(
                results, evaluation_results, feature_cols, output_path
            )
            training_summary['saved_files'] = saved_files
            
            self.logger.warning("="*60)
            self.logger.warning("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.warning("="*60)
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            self.logger.exception("Full traceback:")
            training_summary['error'] = str(e)
            raise
        
        return training_summary


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train models for Chinese Produce Forecasting')
    parser.add_argument('--config', required=True, help='Path to config.yaml')
    parser.add_argument('--data-path', default='data/processed', help='Path to processed data')
    parser.add_argument('--output-path', default='models', help='Path for model output')
    parser.add_argument('--model-version', required=True, help='Model version identifier')
    parser.add_argument('--forecast-horizon', type=int, default=1, help='Forecast horizon in days')
    
    args = parser.parse_args()
    
    # Set up logging first
    logger = setup_logging(args.model_version)
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(args.config, args.model_version, logger)
        
        # Create output directory
        os.makedirs(args.output_path, exist_ok=True)
        
        # Run training pipeline
        results = trainer.run_training_pipeline(
            args.data_path, 
            args.output_path, 
            args.forecast_horizon
        )
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL TRAINING SUMMARY")
        print("="*60)
        print(f"Model Version: {results['model_version']}")
        print(f"Status: {results['status'].upper()}")
        print(f"Models Trained: {results['models_trained']}")
        print(f"Best Model: {results.get('best_model', 'None')}")
        
        if results.get('best_performance'):
            print(f"\nBest Model Performance:")
            for metric, value in results['best_performance'].items():
                if 'mape' in metric.lower():
                    print(f"  {metric}: {value:.3f}%")
                elif 'r2' in metric.lower():
                    print(f"  {metric}: {value:.3f}")
                else:
                    print(f"  {metric}: {value:.3f}")
        
        print(f"\nOutput Path: {args.output_path}")
        print(f"Log Files: logs/training_{args.model_version}_*.log")
        
        logger.warning("="*60)
        logger.warning("MAIN FUNCTION COMPLETED SUCCESSFULLY")
        logger.warning("="*60)
        
    except Exception as e:
        logger.error(f"Main function failed: {e}")
        logger.exception("Full traceback:")
        print(f"\n❌ Training failed: {e}")
        print(f"Check log files in logs/ directory for details")
        sys.exit(1)


if __name__ == "__main__":
    main()