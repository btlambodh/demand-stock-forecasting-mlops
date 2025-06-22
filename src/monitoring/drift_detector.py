#!/usr/bin/env python3
"""
Data and Model Drift Detection for Chinese Produce Market Forecasting
Monitors data distribution changes and model performance degradation

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import argparse
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings

import yaml
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Optional dependencies
try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    logging.warning("boto3 not available, AWS features disabled")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Plotting libraries not available")

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logging.warning("Evidently not available, using statistical methods only")

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DriftDetector:
    """FIXED production drift detection system for data and model monitoring"""
    
    def __init__(self, config_path: str, local_mode: bool = False):
        """Initialize drift detector with configuration"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            self.config = self._get_default_config()
        
        self.local_mode = local_mode
        self.aws_enabled = AWS_AVAILABLE and not local_mode
        
        # Initialize AWS clients only if available and not in local mode
        if self.aws_enabled:
            try:
                self.aws_config = self.config.get('aws', {})
                region = self.aws_config.get('region', 'us-east-1')
                self.s3_client = boto3.client('s3', region_name=region)
                self.cloudwatch = boto3.client('cloudwatch', region_name=region)
                self.sns = boto3.client('sns', region_name=region)
                logger.info("AWS integration enabled for drift detection")
            except Exception as e:
                logger.warning(f"AWS initialization failed: {e}, falling back to local mode")
                self.aws_enabled = False
        
        # Drift thresholds
        monitoring_config = self.config.get('monitoring', {})
        performance_config = monitoring_config.get('performance', {})
        self.drift_threshold = performance_config.get('drift_threshold', 0.25)
        self.performance_threshold = performance_config.get('performance_degradation_threshold', 0.15)
        
        # Reference data storage
        self.reference_data = None
        self.reference_stats = {}
        self.baseline_performance = {}
        
        # Create local storage directories
        self.reports_dir = "data/monitoring/reports"
        self.state_dir = "data/monitoring/state"
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        
        logger.info(f"DriftDetector initialized (local_mode: {local_mode}, aws_enabled: {self.aws_enabled})")

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'monitoring': {
                'performance': {
                    'drift_threshold': 0.25,
                    'performance_degradation_threshold': 0.15
                },
                'alerts': {
                    'enabled': True,
                    'cooldown_minutes': 30
                }
            },
            'aws': {
                'region': 'us-east-1',
                's3': {
                    'bucket_name': 'ml-monitoring-bucket'
                },
                'cloudwatch': {
                    'metrics_namespace': 'ChineseProduceForecast'
                }
            }
        }

    def load_reference_data(self, reference_data_path: str = None) -> bool:
        """Load reference data for drift detection"""
        logger.info("🔍 Loading reference data for drift detection")
        
        try:
            if reference_data_path and os.path.exists(reference_data_path):
                # Load from local file
                if reference_data_path.endswith('.parquet'):
                    self.reference_data = pd.read_parquet(reference_data_path)
                else:
                    self.reference_data = pd.read_csv(reference_data_path)
                logger.info(f"✅ Loaded reference data from local file: {self.reference_data.shape}")
            
            elif self.aws_enabled:
                # Try to load from S3
                bucket = self.aws_config.get('s3', {}).get('bucket_name', 'ml-monitoring-bucket')
                reference_key = "data/processed/train.parquet"
                
                try:
                    response = self.s3_client.get_object(Bucket=bucket, Key=reference_key)
                    self.reference_data = pd.read_parquet(response['Body'])
                    logger.info(f"✅ Loaded reference data from S3: {self.reference_data.shape}")
                
                except Exception as e:
                    logger.warning(f"Could not load reference data from S3: {e}")
                    return False
            
            else:
                # Try to find local training data
                potential_files = [
                    "data/processed/train.parquet",
                    "data/processed/train.csv",
                    "data/raw/annex2.csv"
                ]
                
                for file_path in potential_files:
                    if os.path.exists(file_path):
                        if file_path.endswith('.parquet'):
                            self.reference_data = pd.read_parquet(file_path)
                        else:
                            self.reference_data = pd.read_csv(file_path)
                        logger.info(f"✅ Loaded reference data from: {file_path} - {self.reference_data.shape}")
                        break
                
                if self.reference_data is None:
                    logger.warning("No reference data found")
                    return False
            
            # Calculate reference statistics
            self.calculate_reference_statistics()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading reference data: {e}")
            return False

    def calculate_reference_statistics(self):
        """Calculate statistical properties of reference data"""
        if self.reference_data is None:
            logger.warning("No reference data available for statistics calculation")
            return
        
        logger.info("📊 Calculating reference data statistics")
        
        # Select numeric columns only
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in self.reference_data.columns:
                col_data = self.reference_data[col].dropna()
                
                if len(col_data) > 0:
                    self.reference_stats[col] = {
                        'mean': col_data.mean(),
                        'std': col_data.std(),
                        'median': col_data.median(),
                        'q25': col_data.quantile(0.25),
                        'q75': col_data.quantile(0.75),
                        'min': col_data.min(),
                        'max': col_data.max(),
                        'skewness': stats.skew(col_data),
                        'kurtosis': stats.kurtosis(col_data),
                        'missing_rate': self.reference_data[col].isnull().mean()
                    }
        
        logger.info(f"✅ Calculated statistics for {len(self.reference_stats)} features")

    def detect_data_drift(self, current_data: pd.DataFrame, 
                         method: str = 'statistical') -> Dict[str, Any]:
        """Detect data drift using various statistical methods"""
        logger.info(f"🔍 Detecting data drift using {method} method")
        
        if self.reference_data is None:
            raise ValueError("Reference data not loaded. Call load_reference_data() first.")
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'overall_drift_detected': False,
            'drift_score': 0.0,
            'feature_drift': {},
            'summary': {},
            'reference_shape': self.reference_data.shape,
            'current_shape': current_data.shape
        }
        
        try:
            if method == 'statistical':
                drift_results.update(self._detect_statistical_drift(current_data))
            
            elif method == 'evidently':
                if EVIDENTLY_AVAILABLE:
                    drift_results.update(self._detect_evidently_drift(current_data))
                else:
                    logger.warning("Evidently not available, falling back to statistical method")
                    drift_results.update(self._detect_statistical_drift(current_data))
            
            elif method == 'ks_test':
                drift_results.update(self._detect_ks_drift(current_data))
            
            elif method == 'population_stability':
                drift_results.update(self._detect_psi_drift(current_data))
            
            else:
                raise ValueError(f"Unknown drift detection method: {method}")
            
            # Determine overall drift
            self._determine_overall_drift(drift_results)
            
            logger.info(f"✅ Drift detection completed. Overall drift: {drift_results['overall_drift_detected']}")
            
            return drift_results
            
        except Exception as e:
            logger.error(f"❌ Error in drift detection: {e}")
            drift_results['error'] = str(e)
            return drift_results

    def _detect_statistical_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift using statistical measures"""
        feature_drift = {}
        drift_scores = []
        
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        common_cols = set(numeric_cols) & set(self.reference_stats.keys())
        
        logger.info(f"📊 Analyzing {len(common_cols)} common numeric features")
        
        for col in common_cols:
            current_col = current_data[col].dropna()
            if len(current_col) == 0:
                continue
            
            ref_stats = self.reference_stats[col]
            
            # Calculate current statistics
            current_stats = {
                'mean': current_col.mean(),
                'std': current_col.std(),
                'median': current_col.median(),
                'skewness': stats.skew(current_col),
                'kurtosis': stats.kurtosis(current_col)
            }
            
            # Calculate drift scores for different statistics
            mean_drift = abs(current_stats['mean'] - ref_stats['mean']) / (ref_stats['std'] + 1e-8)
            std_drift = abs(current_stats['std'] - ref_stats['std']) / (ref_stats['std'] + 1e-8)
            median_drift = abs(current_stats['median'] - ref_stats['median']) / (ref_stats['std'] + 1e-8)
            
            # Aggregate drift score for this feature
            feature_drift_score = np.mean([mean_drift, std_drift, median_drift])
            drift_scores.append(feature_drift_score)
            
            feature_drift[col] = {
                'drift_score': feature_drift_score,
                'mean_drift': mean_drift,
                'std_drift': std_drift,
                'median_drift': median_drift,
                'drift_detected': feature_drift_score > self.drift_threshold,
                'current_stats': current_stats,
                'reference_stats': ref_stats
            }
        
        return {
            'feature_drift': feature_drift,
            'drift_score': np.mean(drift_scores) if drift_scores else 0.0,
            'method_details': {
                'features_analyzed': len(feature_drift),
                'mean_drift_score': np.mean(drift_scores) if drift_scores else 0.0,
                'max_drift_score': np.max(drift_scores) if drift_scores else 0.0,
                'min_drift_score': np.min(drift_scores) if drift_scores else 0.0
            }
        }

    def _detect_evidently_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift using Evidently library"""
        try:
            # Prepare data for Evidently
            reference_subset = self.reference_data.sample(
                n=min(len(self.reference_data), 10000), random_state=42
            )
            current_subset = current_data.sample(
                n=min(len(current_data), 10000), random_state=42
            )
            
            # Select common columns
            common_cols = list(set(reference_subset.columns) & set(current_subset.columns))
            numeric_cols = reference_subset[common_cols].select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                logger.warning("No common numeric columns found for Evidently analysis")
                return {'feature_drift': {}, 'drift_score': 0.0}
            
            ref_data = reference_subset[numeric_cols]
            curr_data = current_subset[numeric_cols]
            
            # Create Evidently report
            report = Report(metrics=[
                DataDriftPreset(),
                DataQualityPreset(),
                DatasetDriftMetric(),
                DatasetMissingValuesMetric()
            ])
            
            report.run(reference_data=ref_data, current_data=curr_data)
            
            # Extract results
            result_dict = report.as_dict()
            
            feature_drift = {}
            drift_scores = []
            
            # Parse Evidently results
            if 'metrics' in result_dict:
                for metric in result_dict['metrics']:
                    if metric.get('metric') == 'DatasetDriftMetric':
                        dataset_drift = metric.get('result', {})
                        
                        # Feature-level drift
                        if 'drift_by_columns' in dataset_drift:
                            for col, drift_info in dataset_drift['drift_by_columns'].items():
                                drift_score = drift_info.get('drift_score', 0)
                                drift_scores.append(drift_score)
                                
                                feature_drift[col] = {
                                    'drift_score': drift_score,
                                    'drift_detected': drift_info.get('drift_detected', False),
                                    'p_value': drift_info.get('stattest_threshold', 0),
                                    'method': drift_info.get('stattest_name', 'unknown')
                                }
            
            return {
                'feature_drift': feature_drift,
                'drift_score': np.mean(drift_scores) if drift_scores else 0.0,
                'method_details': {
                    'library': 'evidently',
                    'features_analyzed': len(feature_drift),
                    'reference_samples': len(ref_data),
                    'current_samples': len(curr_data)
                }
            }
            
        except Exception as e:
            logger.warning(f"Evidently drift detection failed: {e}")
            logger.info("Falling back to statistical drift detection")
            return self._detect_statistical_drift(current_data)

    def _detect_ks_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift using Kolmogorov-Smirnov test"""
        feature_drift = {}
        drift_scores = []
        
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        common_cols = set(numeric_cols) & set(self.reference_data.columns)
        
        logger.info(f"🔍 Running KS test on {len(common_cols)} features")
        
        for col in common_cols:
            ref_col = self.reference_data[col].dropna()
            curr_col = current_data[col].dropna()
            
            if len(ref_col) == 0 or len(curr_col) == 0:
                continue
            
            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(ref_col, curr_col)
            
            # Use KS statistic as drift score
            drift_scores.append(ks_statistic)
            
            feature_drift[col] = {
                'drift_score': ks_statistic,
                'p_value': p_value,
                'drift_detected': p_value < 0.05,  # Standard significance level
                'test': 'kolmogorov_smirnov',
                'interpretation': self._interpret_ks_test(ks_statistic, p_value)
            }
        
        return {
            'feature_drift': feature_drift,
            'drift_score': np.mean(drift_scores) if drift_scores else 0.0,
            'method_details': {
                'test': 'kolmogorov_smirnov',
                'significance_level': 0.05,
                'features_analyzed': len(feature_drift),
                'significant_features': sum(1 for fd in feature_drift.values() if fd['drift_detected'])
            }
        }

    def _interpret_ks_test(self, ks_statistic: float, p_value: float) -> str:
        """Interpret KS test results"""
        if p_value < 0.001:
            return f"Highly significant drift (KS={ks_statistic:.3f}, p<0.001)"
        elif p_value < 0.01:
            return f"Very significant drift (KS={ks_statistic:.3f}, p<0.01)"
        elif p_value < 0.05:
            return f"Significant drift (KS={ks_statistic:.3f}, p<0.05)"
        else:
            return f"No significant drift (KS={ks_statistic:.3f}, p={p_value:.3f})"

    def _detect_psi_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift using Population Stability Index (PSI)"""
        feature_drift = {}
        drift_scores = []
        
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        common_cols = set(numeric_cols) & set(self.reference_data.columns)
        
        logger.info(f"📊 Calculating PSI for {len(common_cols)} features")
        
        for col in common_cols:
            ref_col = self.reference_data[col].dropna()
            curr_col = current_data[col].dropna()
            
            if len(ref_col) == 0 or len(curr_col) == 0:
                continue
            
            # Calculate PSI
            psi_score = self._calculate_psi(ref_col, curr_col)
            drift_scores.append(psi_score)
            
            # PSI interpretation:
            # < 0.1: No significant change
            # 0.1-0.25: Some change
            # > 0.25: Significant change
            drift_detected = psi_score > 0.25
            
            feature_drift[col] = {
                'drift_score': psi_score,
                'drift_detected': drift_detected,
                'interpretation': self._interpret_psi(psi_score),
                'test': 'population_stability_index'
            }
        
        return {
            'feature_drift': feature_drift,
            'drift_score': np.mean(drift_scores) if drift_scores else 0.0,
            'method_details': {
                'test': 'population_stability_index',
                'features_analyzed': len(feature_drift),
                'high_drift_features': sum(1 for fd in feature_drift.values() if fd['drift_score'] > 0.25),
                'moderate_drift_features': sum(1 for fd in feature_drift.values() if 0.1 < fd['drift_score'] <= 0.25)
            }
        }

    def _calculate_psi(self, reference: pd.Series, current: pd.Series, 
                      bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins based on reference data
            bin_edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
            bin_edges = np.unique(bin_edges)  # Remove duplicates
            
            if len(bin_edges) < 2:
                return 0.0
            
            # Calculate distributions
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            curr_counts, _ = np.histogram(current, bins=bin_edges)
            
            # Convert to percentages
            ref_pct = ref_counts / len(reference)
            curr_pct = curr_counts / len(current)
            
            # Avoid division by zero
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)
            
            # Calculate PSI
            psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
            
            return psi
            
        except Exception as e:
            logger.warning(f"Error calculating PSI: {e}")
            return 0.0

    def _interpret_psi(self, psi_value: float) -> str:
        """Interpret PSI value"""
        if psi_value < 0.1:
            return f"No significant change (PSI: {psi_value:.3f})"
        elif psi_value < 0.25:
            return f"Some change detected (PSI: {psi_value:.3f})"
        else:
            return f"Significant change detected (PSI: {psi_value:.3f})"

    def _determine_overall_drift(self, drift_results: Dict[str, Any]):
        """Determine overall drift status"""
        feature_drift = drift_results.get('feature_drift', {})
        
        if not feature_drift:
            return
        
        # Count features with drift
        drifted_features = sum(1 for fd in feature_drift.values() if fd.get('drift_detected', False))
        total_features = len(feature_drift)
        
        # Overall drift if more than 30% of features show drift OR drift score exceeds threshold
        drift_ratio = drifted_features / total_features if total_features > 0 else 0
        overall_drift = drift_ratio > 0.3 or drift_results.get('drift_score', 0) > self.drift_threshold
        
        drift_results['overall_drift_detected'] = overall_drift
        drift_results['summary'] = {
            'total_features': total_features,
            'drifted_features': drifted_features,
            'drift_ratio': drift_ratio,
            'overall_drift_score': drift_results.get('drift_score', 0),
            'drift_threshold': self.drift_threshold
        }

    def detect_model_performance_drift(self, predictions: List[float], 
                                     actual_values: List[float],
                                     model_name: str = "unknown") -> Dict[str, Any]:
        """Detect model performance drift"""
        logger.info(f"🔍 Detecting performance drift for model: {model_name}")
        
        if len(predictions) != len(actual_values):
            raise ValueError("Predictions and actual values must have same length")
        
        # Calculate current performance metrics
        current_mae = mean_absolute_error(actual_values, predictions)
        current_mse = mean_squared_error(actual_values, predictions)
        
        # Calculate MAPE safely
        actual_array = np.array(actual_values)
        pred_array = np.array(predictions)
        non_zero_mask = actual_array != 0
        
        if np.sum(non_zero_mask) > 0:
            current_mape = np.mean(np.abs((actual_array[non_zero_mask] - pred_array[non_zero_mask]) / actual_array[non_zero_mask])) * 100
        else:
            current_mape = 0.0
        
        current_performance = {
            'mae': current_mae,
            'mse': current_mse,
            'mape': current_mape,
            'rmse': np.sqrt(current_mse),
            'sample_size': len(predictions)
        }
        
        # Compare with baseline if available
        baseline_key = f"{model_name}_baseline"
        performance_drift = False
        degradation_pct = 0.0
        
        if baseline_key in self.baseline_performance:
            baseline = self.baseline_performance[baseline_key]
            baseline_mape = baseline.get('mape', current_mape)
            
            # Calculate performance degradation
            if baseline_mape > 0:
                degradation_pct = (current_mape - baseline_mape) / baseline_mape * 100
                performance_drift = degradation_pct > (self.performance_threshold * 100)
        
        else:
            # Set current as baseline if no baseline exists
            self.baseline_performance[baseline_key] = current_performance
            logger.info(f"✅ Set baseline performance for {model_name}")
        
        drift_result = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'performance_drift_detected': performance_drift,
            'current_performance': current_performance,
            'baseline_performance': self.baseline_performance.get(baseline_key, {}),
            'degradation_percentage': degradation_pct,
            'threshold_percentage': self.performance_threshold * 100,
            'sample_size': len(predictions)
        }
        
        logger.info(f"✅ Performance drift analysis completed. Drift detected: {performance_drift}")
        
        return drift_result

    def generate_drift_report(self, drift_results: Dict[str, Any], 
                            output_path: str = None) -> str:
        """Generate comprehensive drift report"""
        logger.info("📄 Generating drift report")
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.reports_dir, f"drift_report_{timestamp}.html")
        
        try:
            # Create HTML report
            html_content = self._create_html_report(drift_results)
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"✅ Drift report saved to: {output_path}")
            
            # Upload to S3 if configured and AWS enabled
            if self.aws_enabled:
                try:
                    bucket = self.aws_config.get('s3', {}).get('bucket_name', 'ml-monitoring-bucket')
                    s3_key = f"monitoring/drift-reports/{os.path.basename(output_path)}"
                    
                    self.s3_client.upload_file(output_path, bucket, s3_key)
                    logger.info(f"☁️  Drift report uploaded to S3: s3://{bucket}/{s3_key}")
                    
                except Exception as e:
                    logger.warning(f"Could not upload report to S3: {e}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error generating drift report: {e}")
            return ""

    def _create_html_report(self, drift_results: Dict[str, Any]) -> str:
        """Create HTML drift report"""
        feature_drift = drift_results.get('feature_drift', {})
        summary = drift_results.get('summary', {})
        method = drift_results.get('method', 'unknown')
        
        # Count drift by severity
        high_drift = sum(1 for fd in feature_drift.values() 
                        if fd.get('drift_score', 0) > 0.5)
        medium_drift = sum(1 for fd in feature_drift.values() 
                          if 0.2 < fd.get('drift_score', 0) <= 0.5)
        low_drift = sum(1 for fd in feature_drift.values() 
                       if 0.1 < fd.get('drift_score', 0) <= 0.2)
        no_drift = len(feature_drift) - high_drift - medium_drift - low_drift
        
        # Status color
        overall_drift = drift_results.get('overall_drift_detected', False)
        status_color = '#e74c3c' if overall_drift else '#27ae60'
        status_text = 'DRIFT DETECTED' if overall_drift else 'NO SIGNIFICANT DRIFT'
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Drift Detection Report - Chinese Produce Forecasting</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f8f9fa;
                }}
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background-color: white; 
                    border-radius: 10px; 
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    overflow: hidden;
                }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    padding: 30px; 
                    text-align: center;
                }}
                .header h1 {{ margin: 0; font-size: 2.5em; }}
                .header p {{ margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9; }}
                .status {{ 
                    background-color: {status_color}; 
                    color: white; 
                    padding: 20px; 
                    text-align: center; 
                    font-size: 1.5em; 
                    font-weight: bold;
                }}
                .content {{ padding: 30px; }}
                .summary {{ 
                    background-color: #f8f9fa; 
                    padding: 25px; 
                    margin: 20px 0; 
                    border-radius: 8px; 
                    border-left: 5px solid #667eea;
                }}
                .summary h2 {{ margin-top: 0; color: #2c3e50; }}
                .metrics-grid {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 20px; 
                    margin: 20px 0; 
                }}
                .metric {{ 
                    background-color: white; 
                    padding: 20px; 
                    border-radius: 8px; 
                    text-align: center; 
                    border: 1px solid #e9ecef;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{ 
                    font-size: 2em; 
                    font-weight: bold; 
                    margin-bottom: 5px; 
                }}
                .metric-label {{ 
                    color: #6c757d; 
                    font-size: 0.9em; 
                }}
                .drift-high {{ color: #dc3545; }}
                .drift-medium {{ color: #fd7e14; }}
                .drift-low {{ color: #ffc107; }}
                .drift-none {{ color: #28a745; }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 20px 0; 
                    background-color: white;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                th, td {{ 
                    padding: 12px 15px; 
                    text-align: left; 
                    border-bottom: 1px solid #e9ecef;
                }}
                th {{ 
                    background-color: #495057; 
                    color: white; 
                    font-weight: 600;
                }}
                tr:hover {{ background-color: #f8f9fa; }}
                .footer {{
                    background-color: #495057;
                    color: white;
                    text-align: center;
                    padding: 20px;
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔍 Data Drift Detection Report</h1>
                    <p>Chinese Produce Market Forecasting System</p>
                </div>
                
                <div class="status">
                    {status_text}
                </div>
                
                <div class="content">
                    <div class="summary">
                        <h2>📊 Executive Summary</h2>
                        <p><strong>Analysis Timestamp:</strong> {drift_results.get('timestamp', 'Unknown')}</p>
                        <p><strong>Detection Method:</strong> {method.title()}</p>
                        <p><strong>Reference Data:</strong> {drift_results.get('reference_shape', 'Unknown')} rows × columns</p>
                        <p><strong>Current Data:</strong> {drift_results.get('current_shape', 'Unknown')} rows × columns</p>
                        <p><strong>Overall Drift Score:</strong> {summary.get('overall_drift_score', 0):.3f}</p>
                        <p><strong>Drift Threshold:</strong> {summary.get('drift_threshold', 0.25)}</p>
                    </div>
                    
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-value">{summary.get('total_features', 0)}</div>
                            <div class="metric-label">Total Features Analyzed</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value drift-high">{high_drift}</div>
                            <div class="metric-label">High Drift (>0.5)</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value drift-medium">{medium_drift}</div>
                            <div class="metric-label">Medium Drift (0.2-0.5)</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value drift-low">{low_drift}</div>
                            <div class="metric-label">Low Drift (0.1-0.2)</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value drift-none">{no_drift}</div>
                            <div class="metric-label">No Drift (<0.1)</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{summary.get('drift_ratio', 0)*100:.1f}%</div>
                            <div class="metric-label">Features with Drift</div>
                        </div>
                    </div>
                    
                    <h2>📈 Feature-Level Drift Analysis</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Feature Name</th>
                                <th>Drift Score</th>
                                <th>Drift Detected</th>
                                <th>Risk Level</th>
                                <th>Interpretation</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        # Sort features by drift score (highest first)
        sorted_features = sorted(feature_drift.items(), 
                               key=lambda x: x[1].get('drift_score', 0), 
                               reverse=True)
        
        # Add feature rows
        for feature, drift_info in sorted_features:
            drift_score = drift_info.get('drift_score', 0)
            drift_detected = drift_info.get('drift_detected', False)
            interpretation = drift_info.get('interpretation', '')
            
            if drift_score > 0.5:
                status_class = "drift-high"
                risk_level = "🔴 High Risk"
            elif drift_score > 0.2:
                status_class = "drift-medium"
                risk_level = "🟠 Medium Risk"
            elif drift_score > 0.1:
                status_class = "drift-low"
                risk_level = "🟡 Low Risk"
            else:
                status_class = "drift-none"
                risk_level = "🟢 No Risk"
            
            html_content += f"""
                <tr>
                    <td><strong>{feature}</strong></td>
                    <td class="{status_class}"><strong>{drift_score:.4f}</strong></td>
                    <td>{'Yes' if drift_detected else 'No'}</td>
                    <td class="{status_class}">{risk_level}</td>
                    <td>{interpretation}</td>
                </tr>
            """
        
        # Add method details if available
        method_details = drift_results.get('method_details', {})
        if method_details:
            html_content += f"""
                        </tbody>
                    </table>
                    
                    <div class="summary">
                        <h2>🔬 Technical Details</h2>
                        <p><strong>Method:</strong> {method_details.get('test', method_details.get('library', method))}</p>
                        <p><strong>Features Analyzed:</strong> {method_details.get('features_analyzed', 0)}</p>
            """
            
            if 'mean_drift_score' in method_details:
                html_content += f"<p><strong>Mean Drift Score:</strong> {method_details['mean_drift_score']:.4f}</p>"
            if 'max_drift_score' in method_details:
                html_content += f"<p><strong>Max Drift Score:</strong> {method_details['max_drift_score']:.4f}</p>"
            if 'significant_features' in method_details:
                html_content += f"<p><strong>Statistically Significant Features:</strong> {method_details['significant_features']}</p>"
            
            html_content += "</div>"
        
        html_content += f"""
                    <div class="summary">
                        <h2>📋 Recommendations</h2>
        """
        
        # Add recommendations based on drift level
        if overall_drift:
            if summary.get('drift_ratio', 0) > 0.5:
                html_content += """
                        <p>🚨 <strong>Critical Action Required:</strong></p>
                        <ul>
                            <li>Immediate investigation of data pipeline changes</li>
                            <li>Consider retraining all models with recent data</li>
                            <li>Review data collection and preprocessing steps</li>
                            <li>Implement emergency monitoring protocols</li>
                        </ul>
                """
            else:
                html_content += """
                        <p>⚠️ <strong>Moderate Action Required:</strong></p>
                        <ul>
                            <li>Investigate specific features showing high drift</li>
                            <li>Schedule model retraining within next maintenance window</li>
                            <li>Monitor model performance closely</li>
                            <li>Review recent changes to data sources</li>
                        </ul>
                """
        else:
            html_content += """
                        <p>✅ <strong>No Immediate Action Required:</strong></p>
                        <ul>
                            <li>Continue regular monitoring schedule</li>
                            <li>Maintain current model deployment</li>
                            <li>Schedule next drift analysis as planned</li>
                        </ul>
            """
        
        html_content += f"""
                    </div>
                </div>
                
                <div class="footer">
                    <p>Generated by Chinese Produce Forecasting ML Monitoring System</p>
                    <p>Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content

    def send_drift_alert(self, drift_results: Dict[str, Any], 
                        alert_type: str = "data_drift"):
        """Send drift detection alerts"""
        if not drift_results.get('overall_drift_detected', False):
            logger.info("No drift detected, skipping alert")
            return
        
        logger.info(f"🚨 Sending {alert_type} alert")
        
        try:
            # Prepare alert message
            summary = drift_results.get('summary', {})
            timestamp = drift_results.get('timestamp', datetime.now().isoformat())
            
            message = f"""
            🚨 DRIFT ALERT - Chinese Produce Forecasting System
            
            Alert Type: {alert_type.upper()}
            Timestamp: {timestamp}
            
            Summary:
            - Overall Drift Detected: {drift_results.get('overall_drift_detected')}
            - Drift Score: {drift_results.get('drift_score', 0):.3f}
            - Features Analyzed: {summary.get('total_features', 0)}
            - Drifted Features: {summary.get('drifted_features', 0)}
            - Drift Ratio: {summary.get('drift_ratio', 0):.2%}
            - Method: {drift_results.get('method', 'unknown')}
            
            Action Required: Please review the model performance and consider retraining.
            """
            
            # Save alert locally
            alert_file = os.path.join(self.reports_dir, f"alert_{alert_type}_{int(time.time())}.json")
            alert_data = {
                'timestamp': timestamp,
                'alert_type': alert_type,
                'drift_results': drift_results,
                'message': message
            }
            
            with open(alert_file, 'w') as f:
                json.dump(alert_data, f, indent=2)
            
            logger.info(f"📁 Alert saved locally: {alert_file}")
            
            # Send to AWS services (only if enabled)
            if self.aws_enabled:
                # Send CloudWatch metric
                try:
                    self.cloudwatch.put_metric_data(
                        Namespace='ChineseProduceForecast/DriftDetection',
                        MetricData=[
                            {
                                'MetricName': 'DriftDetected',
                                'Value': 1.0 if drift_results.get('overall_drift_detected') else 0.0,
                                'Unit': 'Count',
                                'Dimensions': [
                                    {
                                        'Name': 'AlertType',
                                        'Value': alert_type
                                    }
                                ]
                            },
                            {
                                'MetricName': 'DriftScore',
                                'Value': drift_results.get('drift_score', 0),
                                'Unit': 'None'
                            }
                        ]
                    )
                    logger.info("☁️  Alert metrics sent to CloudWatch")
                except Exception as e:
                    logger.warning(f"Could not send metrics to CloudWatch: {e}")
                
                # Send SNS notification if configured
                alerts_config = self.config.get('monitoring', {}).get('alerts', {})
                sns_topic = alerts_config.get('sns_topic')
                
                if sns_topic:
                    try:
                        self.sns.publish(
                            TopicArn=sns_topic,
                            Message=message,
                            Subject=f"Drift Alert - {alert_type}"
                        )
                        logger.info("📧 SNS alert sent successfully")
                    except Exception as e:
                        logger.warning(f"Could not send SNS alert: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error sending drift alert: {e}")

    def save_drift_state(self, output_path: str = None):
        """Save current drift detection state"""
        if output_path is None:
            output_path = os.path.join(self.state_dir, "drift_state.json")
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        state = {
            'reference_stats': convert_numpy_types(self.reference_stats),
            'baseline_performance': convert_numpy_types(self.baseline_performance),
            'drift_threshold': float(self.drift_threshold),
            'performance_threshold': float(self.performance_threshold),
            'last_updated': datetime.now().isoformat(),
            'local_mode': self.local_mode,
            'aws_enabled': self.aws_enabled
        }
        
        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"📁 Drift state saved to: {output_path}")

    def load_drift_state(self, state_path: str = None) -> bool:
        """Load drift detection state"""
        if state_path is None:
            state_path = os.path.join(self.state_dir, "drift_state.json")
        
        try:
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                self.reference_stats = state.get('reference_stats', {})
                self.baseline_performance = state.get('baseline_performance', {})
                self.drift_threshold = state.get('drift_threshold', self.drift_threshold)
                self.performance_threshold = state.get('performance_threshold', self.performance_threshold)
                
                logger.info(f"✅ Drift state loaded from: {state_path}")
                return True
            
        except Exception as e:
            logger.error(f"❌ Error loading drift state: {e}")
        
        return False

    def export_drift_state(self, output_path: str = None) -> str:
        """Export drift state for analysis"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"drift_export_{timestamp}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'reference_stats': self.reference_stats,
            'baseline_performance': self.baseline_performance,
            'configuration': {
                'drift_threshold': self.drift_threshold,
                'performance_threshold': self.performance_threshold,
                'local_mode': self.local_mode,
                'aws_enabled': self.aws_enabled
            },
            'reference_data_shape': self.reference_data.shape if self.reference_data is not None else None
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"📁 Drift state exported to: {output_path}")
        return output_path


def main():
    """Main function for testing drift detection"""
    parser = argparse.ArgumentParser(description='FIXED Drift Detection for Chinese Produce Forecasting')
    parser.add_argument('--config', default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--action', required=True,
                       choices=['detect', 'monitor', 'alert-test', 'export-state'],
                       help='Action to perform')
    parser.add_argument('--reference-data', help='Path to reference data file')
    parser.add_argument('--current-data', help='Path to current data file')
    parser.add_argument('--method', default='statistical',
                       choices=['statistical', 'evidently', 'ks_test', 'population_stability'],
                       help='Drift detection method')
    parser.add_argument('--local-mode', action='store_true', help='Run in local mode (no AWS)')
    parser.add_argument('--generate-report', action='store_true', help='Generate HTML report')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    try:
        # Initialize drift detector
        detector = DriftDetector(args.config, local_mode=args.local_mode)
        
        if args.action == 'detect':
            print(f"🔍 Data Drift Detection Starting")
            print(f"   Method: {args.method}")
            print(f"   Local mode: {args.local_mode}")
            
            if not args.current_data:
                print("❌ Error: --current-data required for detect action")
                return
            
            # Load reference data
            reference_loaded = detector.load_reference_data(args.reference_data)
            if not reference_loaded:
                print("❌ Error: Could not load reference data")
                return
            
            # Load current data
            try:
                if args.current_data.endswith('.parquet'):
                    current_data = pd.read_parquet(args.current_data)
                else:
                    current_data = pd.read_csv(args.current_data)
                
                print(f"✅ Current data loaded: {current_data.shape}")
            except Exception as e:
                print(f"❌ Error loading current data: {e}")
                return
            
            # Detect drift
            results = detector.detect_data_drift(current_data, args.method)
            
            # Generate report if requested
            if args.generate_report or results.get('overall_drift_detected', False):
                report_path = detector.generate_drift_report(results)
                print(f"📄 Report generated: {report_path}")
            
            # Display results
            print(f"\n📊 Drift Detection Results:")
            print(f"=" * 50)
            print(f"Overall Drift Detected: {results['overall_drift_detected']}")
            print(f"Drift Score: {results['drift_score']:.4f}")
            print(f"Method: {results['method']}")
            print(f"Reference Data: {results.get('reference_shape', 'Unknown')}")
            print(f"Current Data: {results.get('current_shape', 'Unknown')}")
            
            summary = results.get('summary', {})
            if summary:
                print(f"\n📈 Summary:")
                print(f"Features Analyzed: {summary.get('total_features', 0)}")
                print(f"Drifted Features: {summary.get('drifted_features', 0)}")
                print(f"Drift Ratio: {summary.get('drift_ratio', 0)*100:.1f}%")
                print(f"Threshold: {summary.get('drift_threshold', 0.25)}")
            
            # Show top drifted features
            feature_drift = results.get('feature_drift', {})
            if feature_drift:
                print(f"\n🎯 Top Drifted Features:")
                sorted_features = sorted(feature_drift.items(), 
                                       key=lambda x: x[1].get('drift_score', 0), 
                                       reverse=True)
                
                for i, (feature, drift_info) in enumerate(sorted_features[:5]):
                    drift_score = drift_info.get('drift_score', 0)
                    status = "🔴" if drift_score > 0.5 else "🟠" if drift_score > 0.2 else "🟡" if drift_score > 0.1 else "🟢"
                    print(f"  {status} {feature}: {drift_score:.4f}")
            
            # Send alert if drift detected
            if results['overall_drift_detected']:
                detector.send_drift_alert(results)
                print(f"🚨 Drift alert sent")
        
        elif args.action == 'monitor':
            print("🔄 Starting continuous drift monitoring...")
            print("   (In production, this would run as a scheduled job)")
            print("   Use cron or systemd for continuous monitoring")
        
        elif args.action == 'alert-test':
            print("🧪 Testing drift alert system...")
            
            # Create test drift results
            test_results = {
                'timestamp': datetime.now().isoformat(),
                'method': 'test',
                'overall_drift_detected': True,
                'drift_score': 0.75,
                'summary': {
                    'total_features': 20,
                    'drifted_features': 8,
                    'drift_ratio': 0.4,
                    'drift_threshold': 0.25
                },
                'feature_drift': {
                    'test_feature_1': {'drift_score': 0.8, 'drift_detected': True},
                    'test_feature_2': {'drift_score': 0.6, 'drift_detected': True}
                }
            }
            
            detector.send_drift_alert(test_results, "test_alert")
            print("✅ Test alert processed")
            
            if detector.aws_enabled:
                print("☁️  Test alert sent to AWS services")
            else:
                print("🏠 Running in local mode - alerts saved locally")
        
        elif args.action == 'export-state':
            output_file = detector.export_drift_state(args.output)
            print(f"📁 Drift state exported to: {output_file}")
        
        print(f"\n✅ Drift detection operation completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Drift detection operation failed: {e}")
        print(f"\n❌ Operation failed: {e}")


if __name__ == "__main__":
    main()