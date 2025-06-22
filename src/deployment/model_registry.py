#!/usr/bin/env python3
"""
Model Registry and Versioning for Chinese Produce Market Forecasting
MLflow integration with AWS backend for enterprise model management

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import argparse
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys

import pandas as pd
import numpy as np
import yaml
import joblib
import boto3
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """Enterprise-grade model registry with MLflow and AWS integration"""
    
    def __init__(self, config_path: str):
        """Initialize model registry with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.aws_config = self.config['aws']
        self.s3_client = boto3.client('s3', region_name=self.aws_config['region'])
        
        # Setup MLflow tracking
        self.setup_mlflow()
        self.client = MlflowClient()
        
        # Model performance thresholds
        self.performance_thresholds = self.config['evaluation']['thresholds']
        
        logger.info("Model Registry initialized successfully")

    def setup_mlflow(self):
        """Setup MLflow with AWS backend"""
        try:
            # Set MLflow tracking URI to S3
            s3_bucket = self.aws_config['s3']['bucket_name']
            mlflow_uri = f"s3://{s3_bucket}/mlflow"
            mlflow.set_tracking_uri(mlflow_uri)
            
            # Set experiment
            experiment_name = "chinese-produce-forecasting"
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                    logger.info(f"Created MLflow experiment: {experiment_name}")
                else:
                    experiment_id = experiment.experiment_id
                    logger.info(f"Using existing MLflow experiment: {experiment_name}")
            except Exception:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created MLflow experiment: {experiment_name}")
            
            mlflow.set_experiment(experiment_name)
            
            logger.info(f"MLflow configured with S3 backend: {mlflow_uri}")
            
        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")
            # Fallback to local tracking
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("chinese-produce-forecasting")
            logger.warning("Using local MLflow tracking as fallback")

    def register_model(self, model_path: str, model_name: str, 
                      model_version: str, metadata: Dict) -> Dict[str, str]:
        """Register a trained model with comprehensive metadata"""
        logger.info(f"Registering model: {model_name} version {model_version}")
        
        try:
            with mlflow.start_run(run_name=f"{model_name}_{model_version}"):
                # Load model artifact
                model_artifact = joblib.load(model_path)
                
                # Extract model and preprocessor
                if isinstance(model_artifact, dict):
                    model = model_artifact.get('model')
                    scaler = model_artifact.get('scaler')
                else:
                    model = model_artifact
                    scaler = None
                
                # Log model parameters
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    mlflow.log_params(params)
                    logger.debug(f"Logged model parameters: {len(params)} params")
                
                # Log performance metrics
                if 'metrics' in metadata:
                    for metric_name, value in metadata['metrics'].items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(metric_name, value)
                    logger.info(f"Logged {len(metadata['metrics'])} metrics")
                
                # Log feature importance if available
                if 'feature_importance' in metadata and metadata['feature_importance'] is not None:
                    importance = metadata['feature_importance']
                    if isinstance(importance, np.ndarray):
                        # Create feature importance plot
                        import matplotlib.pyplot as plt
                        
                        # Get top 20 features
                        feature_names = metadata.get('feature_names', [f'feature_{i}' for i in range(len(importance))])
                        top_indices = np.argsort(importance)[-20:]
                        
                        plt.figure(figsize=(10, 8))
                        plt.barh(range(len(top_indices)), importance[top_indices])
                        plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
                        plt.xlabel('Feature Importance')
                        plt.title(f'Top 20 Feature Importance - {model_name}')
                        plt.tight_layout()
                        
                        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
                        mlflow.log_artifact('feature_importance.png')
                        plt.close()
                        
                        # Clean up
                        if os.path.exists('feature_importance.png'):
                            os.remove('feature_importance.png')
                
                # Log model artifacts
                if scaler is not None:
                    # Save complete artifact with model and scaler
                    artifact_path = f"{model_name}_complete_artifact.pkl"
                    joblib.dump(model_artifact, artifact_path)
                    mlflow.log_artifact(artifact_path)
                    
                    # Log sklearn model
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        registered_model_name=model_name
                    )
                    
                    # Clean up
                    if os.path.exists(artifact_path):
                        os.remove(artifact_path)
                else:
                    # Log sklearn model directly
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        registered_model_name=model_name
                    )
                
                # Log additional metadata
                mlflow.set_tags({
                    "model_version": model_version,
                    "training_timestamp": metadata.get('training_timestamp', datetime.now().isoformat()),
                    "model_type": type(model).__name__,
                    "framework": "scikit-learn",
                    "use_case": "chinese_produce_forecasting",
                    "data_version": metadata.get('data_version', 'unknown'),
                    "feature_count": metadata.get('feature_count', 'unknown'),
                    "has_scaler": str(scaler is not None)
                })
                
                # Get run info
                run = mlflow.active_run()
                run_id = run.info.run_id
                
                logger.info(f"Model registered successfully. Run ID: {run_id}")
                
                return {
                    'run_id': run_id,
                    'model_name': model_name,
                    'model_version': model_version,
                    'artifact_uri': run.info.artifact_uri,
                    'status': 'registered'
                }
                
        except Exception as e:
            logger.error(f"Error registering model {model_name}: {e}")
            logger.exception("Full traceback:")
            raise

    def promote_model_to_staging(self, model_name: str, version: str) -> bool:
        """Promote model to staging stage"""
        try:
            logger.info(f"Promoting {model_name} version {version} to Staging")
            
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging"
            )
            
            logger.info(f"Model {model_name} v{version} promoted to Staging")
            return True
            
        except Exception as e:
            logger.error(f"Error promoting model to staging: {e}")
            return False

    def promote_model_to_production(self, model_name: str, version: str, 
                                  performance_metrics: Dict[str, float]) -> bool:
        """Promote model to production with performance validation"""
        try:
            logger.info(f"Validating model for production: {model_name} v{version}")
            
            # Validate performance against thresholds
            if not self.validate_model_performance(performance_metrics):
                logger.warning("Model does not meet performance thresholds for production")
                return False
            
            # Archive current production model
            current_prod_versions = self.client.get_latest_versions(
                model_name, stages=["Production"]
            )
            
            for prod_version in current_prod_versions:
                logger.info(f"Archiving current production model: v{prod_version.version}")
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=prod_version.version,
                    stage="Archived"
                )
            
            # Promote new model to production
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            
            # Update model description
            self.client.update_model_version(
                name=model_name,
                version=version,
                description=f"Production model deployed on {datetime.now().isoformat()}"
            )
            
            logger.info(f"Model {model_name} v{version} promoted to Production")
            return True
            
        except Exception as e:
            logger.error(f"Error promoting model to production: {e}")
            return False

    def validate_model_performance(self, metrics: Dict[str, float]) -> bool:
        """Validate model performance against thresholds"""
        thresholds = self.performance_thresholds
        
        # Check MAPE threshold
        if 'test_mape' in metrics or 'val_mape' in metrics:
            mape = metrics.get('test_mape', metrics.get('val_mape'))
            if mape > thresholds['mape_threshold']:
                logger.warning(f"MAPE {mape:.3f}% exceeds threshold {thresholds['mape_threshold']}%")
                return False
        
        # Check RMSE threshold
        if 'test_rmse' in metrics or 'val_rmse' in metrics:
            rmse = metrics.get('test_rmse', metrics.get('val_rmse'))
            if rmse > thresholds['rmse_threshold']:
                logger.warning(f"RMSE {rmse:.3f} exceeds threshold {thresholds['rmse_threshold']}")
                return False
        
        # Check R² threshold
        if 'test_r2' in metrics or 'val_r2' in metrics:
            r2 = metrics.get('test_r2', metrics.get('val_r2'))
            if r2 < thresholds['r2_threshold']:
                logger.warning(f"R² {r2:.3f} below threshold {thresholds['r2_threshold']}")
                return False
        
        logger.info("Model performance meets all thresholds")
        return True

    def get_production_model(self, model_name: str) -> Optional[Dict]:
        """Get current production model information"""
        try:
            latest_versions = self.client.get_latest_versions(
                model_name, stages=["Production"]
            )
            
            if not latest_versions:
                logger.warning(f"No production model found for {model_name}")
                return None
            
            prod_version = latest_versions[0]
            
            # Get model details
            model_details = {
                'name': prod_version.name,
                'version': prod_version.version,
                'stage': prod_version.current_stage,
                'run_id': prod_version.run_id,
                'description': prod_version.description,
                'creation_timestamp': prod_version.creation_timestamp,
                'last_updated_timestamp': prod_version.last_updated_timestamp
            }
            
            # Get run metrics
            run = self.client.get_run(prod_version.run_id)
            model_details['metrics'] = run.data.metrics
            model_details['params'] = run.data.params
            model_details['tags'] = run.data.tags
            
            logger.info(f"Retrieved production model: {model_name} v{prod_version.version}")
            return model_details
            
        except Exception as e:
            logger.error(f"Error getting production model: {e}")
            return None

    def list_model_versions(self, model_name: str, stage: Optional[str] = None) -> List[Dict]:
        """List all versions of a model, optionally filtered by stage"""
        try:
            if stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
            else:
                versions = self.client.search_model_versions(f"name='{model_name}'")
            
            model_list = []
            for version in versions:
                version_info = {
                    'name': version.name,
                    'version': version.version,
                    'stage': version.current_stage,
                    'run_id': version.run_id,
                    'creation_timestamp': version.creation_timestamp,
                    'last_updated_timestamp': version.last_updated_timestamp
                }
                
                # Get performance metrics
                try:
                    run = self.client.get_run(version.run_id)
                    version_info['metrics'] = run.data.metrics
                    version_info['tags'] = run.data.tags
                except:
                    version_info['metrics'] = {}
                    version_info['tags'] = {}
                
                model_list.append(version_info)
            
            logger.info(f"Found {len(model_list)} versions for {model_name}")
            return model_list
            
        except Exception as e:
            logger.error(f"Error listing model versions: {e}")
            return []

    def compare_models(self, model_name: str, version1: str, version2: str) -> Dict:
        """Compare performance between two model versions"""
        try:
            logger.info(f"Comparing {model_name} v{version1} vs v{version2}")
            
            # Get model version details
            v1_info = self.client.get_model_version(model_name, version1)
            v2_info = self.client.get_model_version(model_name, version2)
            
            # Get metrics for both versions
            v1_run = self.client.get_run(v1_info.run_id)
            v2_run = self.client.get_run(v2_info.run_id)
            
            v1_metrics = v1_run.data.metrics
            v2_metrics = v2_run.data.metrics
            
            # Compare key metrics
            comparison = {
                'model_name': model_name,
                'version1': {
                    'version': version1,
                    'stage': v1_info.current_stage,
                    'metrics': v1_metrics
                },
                'version2': {
                    'version': version2,
                    'stage': v2_info.current_stage,
                    'metrics': v2_metrics
                },
                'comparison': {}
            }
            
            # Calculate differences for key metrics
            key_metrics = ['val_mape', 'test_mape', 'val_rmse', 'test_rmse', 'val_r2', 'test_r2']
            
            for metric in key_metrics:
                if metric in v1_metrics and metric in v2_metrics:
                    v1_val = v1_metrics[metric]
                    v2_val = v2_metrics[metric]
                    
                    # For MAPE and RMSE, lower is better
                    if 'mape' in metric or 'rmse' in metric:
                        improvement = ((v1_val - v2_val) / v1_val) * 100
                        better_version = version2 if v2_val < v1_val else version1
                    # For R², higher is better
                    else:
                        improvement = ((v2_val - v1_val) / v1_val) * 100
                        better_version = version2 if v2_val > v1_val else version1
                    
                    comparison['comparison'][metric] = {
                        'v1_value': v1_val,
                        'v2_value': v2_val,
                        'improvement_pct': improvement,
                        'better_version': better_version
                    }
            
            logger.info(f"Model comparison completed for {model_name}")
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {}

    def cleanup_old_models(self, model_name: str, keep_versions: int = 5) -> int:
        """Cleanup old model versions, keeping only the specified number"""
        try:
            logger.info(f"Cleaning up old versions of {model_name}, keeping {keep_versions}")
            
            # Get all versions sorted by creation time
            all_versions = self.client.search_model_versions(
                f"name='{model_name}'",
                order_by=["creation_timestamp DESC"]
            )
            
            # Never delete Production or Staging models
            protected_stages = ["Production", "Staging"]
            
            deleted_count = 0
            versions_to_keep = keep_versions
            
            for version in all_versions:
                if version.current_stage in protected_stages:
                    continue
                
                if versions_to_keep > 0:
                    versions_to_keep -= 1
                    continue
                
                # Delete old version
                self.client.delete_model_version(model_name, version.version)
                deleted_count += 1
                logger.info(f"Deleted {model_name} version {version.version}")
            
            logger.info(f"Cleanup completed. Deleted {deleted_count} old versions")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0

    def register_models_from_training(self, models_dir: str, evaluation_file: str) -> Dict[str, str]:
        """Register all models from a training run"""
        logger.info(f"Registering models from training run: {models_dir}")
        
        registration_results = {}
        
        try:
            # Load evaluation results
            with open(evaluation_file, 'r') as f:
                evaluation_data = json.load(f)
            
            # Load feature info if available
            feature_info_file = os.path.join(os.path.dirname(evaluation_file), 'feature_info.json')
            feature_info = {}
            if os.path.exists(feature_info_file):
                with open(feature_info_file, 'r') as f:
                    feature_info = json.load(f)
            
            # Register each model
            for model_name, metrics in evaluation_data.items():
                model_file = os.path.join(models_dir, f"{model_name}_model.pkl")
                
                if not os.path.exists(model_file):
                    logger.warning(f"Model file not found: {model_file}")
                    continue
                
                # Prepare metadata
                metadata = {
                    'metrics': metrics,
                    'training_timestamp': datetime.now().isoformat(),
                    'feature_count': feature_info.get('num_features', 'unknown'),
                    'feature_names': feature_info.get('feature_columns', []),
                    'data_version': feature_info.get('model_version', 'unknown')
                }
                
                # Add feature importance if available from feature info
                if 'feature_columns' in feature_info:
                    metadata['feature_names'] = feature_info['feature_columns']
                
                # Generate model version
                model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}"
                
                # Register model
                try:
                    result = self.register_model(
                        model_path=model_file,
                        model_name=f"chinese_produce_{model_name}",
                        model_version=model_version,
                        metadata=metadata
                    )
                    registration_results[model_name] = result
                    
                    # Auto-promote if performance is good enough
                    if self.validate_model_performance(metrics):
                        registered_name = result['model_name']
                        # Find the latest version number for this model
                        versions = self.client.search_model_versions(f"name='{registered_name}'")
                        if versions:
                            latest_version = max([v.version for v in versions])
                            self.promote_model_to_staging(registered_name, latest_version)
                        
                except Exception as e:
                    logger.error(f"Failed to register {model_name}: {e}")
                    registration_results[model_name] = {'status': 'failed', 'error': str(e)}
            
            logger.info(f"Registration completed. Registered {len(registration_results)} models")
            return registration_results
            
        except Exception as e:
            logger.error(f"Error registering models from training: {e}")
            raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Model Registry for Chinese Produce Forecasting')
    parser.add_argument('--config', required=True, help='Path to config.yaml')
    parser.add_argument('--action', required=True, 
                       choices=['register', 'promote', 'list', 'compare', 'cleanup'],
                       help='Action to perform')
    parser.add_argument('--models-dir', help='Directory containing trained models')
    parser.add_argument('--evaluation-file', help='Path to evaluation.json file')
    parser.add_argument('--model-name', help='Model name for operations')
    parser.add_argument('--version', help='Model version')
    parser.add_argument('--version2', help='Second model version for comparison')
    parser.add_argument('--stage', help='Model stage (Staging/Production)')
    
    args = parser.parse_args()
    
    try:
        # Initialize registry
        registry = ModelRegistry(args.config)
        
        if args.action == 'register':
            if not args.models_dir or not args.evaluation_file:
                print("Error: --models-dir and --evaluation-file required for register action")
                sys.exit(1)
            
            results = registry.register_models_from_training(args.models_dir, args.evaluation_file)
            
            print("\n" + "="*60)
            print("MODEL REGISTRATION RESULTS")
            print("="*60)
            for model_name, result in results.items():
                print(f"{model_name}: {result.get('status', 'unknown')}")
                if 'run_id' in result:
                    print(f"  Run ID: {result['run_id']}")
        
        elif args.action == 'promote':
            if not args.model_name or not args.version or not args.stage:
                print("Error: --model-name, --version, and --stage required for promote action")
                sys.exit(1)
            
            if args.stage.lower() == 'staging':
                success = registry.promote_model_to_staging(args.model_name, args.version)
            elif args.stage.lower() == 'production':
                # Need to get performance metrics
                model_versions = registry.list_model_versions(args.model_name)
                target_version = next((v for v in model_versions if v['version'] == args.version), None)
                
                if target_version:
                    success = registry.promote_model_to_production(
                        args.model_name, args.version, target_version['metrics']
                    )
                else:
                    print(f"Version {args.version} not found for model {args.model_name}")
                    success = False
            else:
                print("Error: --stage must be 'staging' or 'production'")
                sys.exit(1)
            
            print(f"Promotion {'successful' if success else 'failed'}")
        
        elif args.action == 'list':
            if not args.model_name:
                print("Error: --model-name required for list action")
                sys.exit(1)
            
            versions = registry.list_model_versions(args.model_name, args.stage)
            
            print(f"\n{args.model_name} Model Versions:")
            print("-" * 60)
            for version in versions:
                print(f"Version {version['version']} ({version['stage']})")
                if 'val_mape' in version['metrics']:
                    print(f"  MAPE: {version['metrics']['val_mape']:.3f}%")
                if 'val_r2' in version['metrics']:
                    print(f"  R²: {version['metrics']['val_r2']:.3f}")
        
        elif args.action == 'compare':
            if not args.model_name or not args.version or not args.version2:
                print("Error: --model-name, --version, and --version2 required for compare action")
                sys.exit(1)
            
            comparison = registry.compare_models(args.model_name, args.version, args.version2)
            
            print(f"\nModel Comparison: {args.model_name}")
            print("-" * 60)
            for metric, data in comparison.get('comparison', {}).items():
                print(f"{metric}:")
                print(f"  v{args.version}: {data['v1_value']:.3f}")
                print(f"  v{args.version2}: {data['v2_value']:.3f}")
                print(f"  Improvement: {data['improvement_pct']:.2f}%")
                print(f"  Better: v{data['better_version']}")
        
        elif args.action == 'cleanup':
            if not args.model_name:
                print("Error: --model-name required for cleanup action")
                sys.exit(1)
            
            deleted_count = registry.cleanup_old_models(args.model_name)
            print(f"Cleaned up {deleted_count} old model versions")
        
        logger.info("Model registry operation completed successfully")
        
    except Exception as e:
        logger.error(f"Model registry operation failed: {e}")
        print(f"\n❌ Operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()