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
import boto3
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric
import joblib

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DriftDetector:
    """Production drift detection system for data and model monitoring"""
    
    def __init__(self, config_path: str):
        """Initialize drift detector with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.aws_config = self.config['aws']
        self.monitoring_config = self.config['monitoring']
        
        # Initialize AWS clients
        self.s3_client = boto3.client('s3', region_name=self.aws_config['region'])
        self.cloudwatch = boto3.client('cloudwatch', region_name=self.aws_config['region'])
        self.sns = boto3.client('sns', region_name=self.aws_config['region'])
        
        # Drift thresholds
        self.drift_threshold = self.monitoring_config['performance']['drift_threshold']
        self.performance_threshold = self.monitoring_config['performance']['performance_degradation_threshold']
        
        # Reference data storage
        self.reference_data = None
        self.reference_stats = {}
        self.baseline_performance = {}
        
        logger.info("DriftDetector initialized successfully")

    def load_reference_data(self, reference_data_path: str = None) -> bool:
        """Load reference data for drift detection"""
        logger.info("Loading reference data for drift detection")
        
        try:
            if reference_data_path and os.path.exists(reference_data_path):
                # Load from local file
                if reference_data_path.endswith('.parquet'):
                    self.reference_data = pd.read_parquet(reference_data_path)
                else:
                    self.reference_data = pd.read_csv(reference_data_path)
                logger.info(f"Loaded reference data from local file: {self.reference_data.shape}")
            
            else:
                # Try to load from S3
                bucket = self.aws_config['s3']['bucket_name']
                reference_key = "data/processed/train.parquet"
                
                try:
                    response = self.s3_client.get_object(Bucket=bucket, Key=reference_key)
                    self.reference_data = pd.read_parquet(response['Body'])
                    logger.info(f"Loaded reference data from S3: {self.reference_data.shape}")
                
                except Exception as e:
                    logger.warning(f"Could not load reference data from S3: {e}")
                    return False
            
            # Calculate reference statistics
            self.calculate_reference_statistics()
            return True
            
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            return False

    def calculate_reference_statistics(self):
        """Calculate statistical properties of reference data"""
        if self.reference_data is None:
            logger.warning("No reference data available for statistics calculation")
            return
        
        logger.info("Calculating reference data statistics")
        
        # Select numeric columns only
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in self.reference_data.columns:
                col_data = self.reference_data[col].dropna()
                
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
        
        logger.info(f"Calculated statistics for {len(self.reference_stats)} features")

    def detect_data_drift(self, current_data: pd.DataFrame, 
                         method: str = 'statistical') -> Dict[str, Any]:
        """Detect data drift using various statistical methods"""
        logger.info(f"Detecting data drift using {method} method")
        
        if self.reference_data is None:
            raise ValueError("Reference data not loaded. Call load_reference_data() first.")
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'overall_drift_detected': False,
            'drift_score': 0.0,
            'feature_drift': {},
            'summary': {}
        }
        
        try:
            if method == 'statistical':
                drift_results.update(self._detect_statistical_drift(current_data))
            
            elif method == 'evidently':
                drift_results.update(self._detect_evidently_drift(current_data))
            
            elif method == 'ks_test':
                drift_results.update(self._detect_ks_drift(current_data))
            
            elif method == 'population_stability':
                drift_results.update(self._detect_psi_drift(current_data))
            
            else:
                raise ValueError(f"Unknown drift detection method: {method}")
            
            # Determine overall drift
            self._determine_overall_drift(drift_results)
            
            logger.info(f"Drift detection completed. Overall drift: {drift_results['overall_drift_detected']}")
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            drift_results['error'] = str(e)
            return drift_results

    def _detect_statistical_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift using statistical measures"""
        feature_drift = {}
        drift_scores = []
        
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in self.reference_stats:
                continue
            
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
                'max_drift_score': np.max(drift_scores) if drift_scores else 0.0
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
                    'features_analyzed': len(feature_drift)
                }
            }
            
        except Exception as e:
            logger.warning(f"Evidently drift detection failed: {e}")
            return self._detect_statistical_drift(current_data)

    def _detect_ks_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift using Kolmogorov-Smirnov test"""
        feature_drift = {}
        drift_scores = []
        
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in self.reference_data.columns:
                continue
            
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
                'test': 'kolmogorov_smirnov'
            }
        
        return {
            'feature_drift': feature_drift,
            'drift_score': np.mean(drift_scores) if drift_scores else 0.0,
            'method_details': {
                'test': 'kolmogorov_smirnov',
                'significance_level': 0.05,
                'features_analyzed': len(feature_drift)
            }
        }

    def _detect_psi_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift using Population Stability Index (PSI)"""
        feature_drift = {}
        drift_scores = []
        
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in self.reference_data.columns:
                continue
            
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
                'features_analyzed': len(feature_drift)
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
            return "No significant change"
        elif psi_value < 0.25:
            return "Some change detected"
        else:
            return "Significant change detected"

    def _determine_overall_drift(self, drift_results: Dict[str, Any]):
        """Determine overall drift status"""
        feature_drift = drift_results.get('feature_drift', {})
        
        if not feature_drift:
            return
        
        # Count features with drift
        drifted_features = sum(1 for fd in feature_drift.values() if fd.get('drift_detected', False))
        total_features = len(feature_drift)
        
        # Overall drift if more than 30% of features show drift
        drift_ratio = drifted_features / total_features if total_features > 0 else 0
        overall_drift = drift_ratio > 0.3 or drift_results.get('drift_score', 0) > self.drift_threshold
        
        drift_results['overall_drift_detected'] = overall_drift
        drift_results['summary'] = {
            'total_features': total_features,
            'drifted_features': drifted_features,
            'drift_ratio': drift_ratio,
            'overall_drift_score': drift_results.get('drift_score', 0)
        }

    def detect_model_performance_drift(self, predictions: List[float], 
                                     actual_values: List[float],
                                     model_name: str = "unknown") -> Dict[str, Any]:
        """Detect model performance drift"""
        logger.info(f"Detecting performance drift for model: {model_name}")
        
        if len(predictions) != len(actual_values):
            raise ValueError("Predictions and actual values must have same length")
        
        # Calculate current performance metrics
        current_mae = mean_absolute_error(actual_values, predictions)
        current_mse = mean_squared_error(actual_values, predictions)
        current_mape = np.mean(np.abs((np.array(actual_values) - np.array(predictions)) / np.array(actual_values))) * 100
        
        current_performance = {
            'mae': current_mae,
            'mse': current_mse,
            'mape': current_mape,
            'rmse': np.sqrt(current_mse)
        }
        
        # Compare with baseline if available
        baseline_key = f"{model_name}_baseline"
        performance_drift = False
        degradation_pct = 0.0
        
        if baseline_key in self.baseline_performance:
            baseline = self.baseline_performance[baseline_key]
            baseline_mape = baseline.get('mape', current_mape)
            
            # Calculate performance degradation
            degradation_pct = (current_mape - baseline_mape) / baseline_mape * 100
            performance_drift = degradation_pct > (self.performance_threshold * 100)
        
        else:
            # Set current as baseline if no baseline exists
            self.baseline_performance[baseline_key] = current_performance
            logger.info(f"Set baseline performance for {model_name}")
        
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
        
        logger.info(f"Performance drift analysis completed. Drift detected: {performance_drift}")
        
        return drift_result

    def generate_drift_report(self, drift_results: Dict[str, Any], 
                            output_path: str = None) -> str:
        """Generate comprehensive drift report"""
        logger.info("Generating drift report")
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"drift_report_{timestamp}.html"
        
        try:
            # Create HTML report
            html_content = self._create_html_report(drift_results)
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Drift report saved to: {output_path}")
            
            # Upload to S3 if configured
            try:
                bucket = self.aws_config['s3']['bucket_name']
                s3_key = f"monitoring/drift-reports/{os.path.basename(output_path)}"
                
                self.s3_client.upload_file(output_path, bucket, s3_key)
                logger.info(f"Drift report uploaded to S3: s3://{bucket}/{s3_key}")
                
            except Exception as e:
                logger.warning(f"Could not upload report to S3: {e}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating drift report: {e}")
            return ""

    def _create_html_report(self, drift_results: Dict[str, Any]) -> str:
        """Create HTML drift report"""
        feature_drift = drift_results.get('feature_drift', {})
        summary = drift_results.get('summary', {})
        
        # Count drift by severity
        high_drift = sum(1 for fd in feature_drift.values() 
                        if fd.get('drift_score', 0) > 0.5)
        medium_drift = sum(1 for fd in feature_drift.values() 
                          if 0.2 < fd.get('drift_score', 0) <= 0.5)
        low_drift = sum(1 for fd in feature_drift.values() 
                       if 0.1 < fd.get('drift_score', 0) <= 0.2)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Drift Report - Chinese Produce Forecasting</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .summary {{ background-color: #e9ecef; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .drift-high {{ color: #dc3545; font-weight: bold; }}
                .drift-medium {{ color: #fd7e14; font-weight: bold; }}
                .drift-low {{ color: #ffc107; }}
                .drift-none {{ color: #28a745; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Drift Detection Report</h1>
                <p><strong>Generated:</strong> {drift_results.get('timestamp', 'Unknown')}</p>
                <p><strong>Method:</strong> {drift_results.get('method', 'Unknown')}</p>
                <p><strong>Overall Drift Detected:</strong> 
                   <span class="{'drift-high' if drift_results.get('overall_drift_detected') else 'drift-none'}">
                   {drift_results.get('overall_drift_detected', False)}
                   </span>
                </p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <div class="metric">
                    <strong>Total Features:</strong> {summary.get('total_features', 0)}
                </div>
                <div class="metric">
                    <strong>Drifted Features:</strong> {summary.get('drifted_features', 0)}
                </div>
                <div class="metric">
                    <strong>Overall Drift Score:</strong> {summary.get('overall_drift_score', 0):.3f}
                </div>
                <div class="metric">
                    <strong>High Drift:</strong> <span class="drift-high">{high_drift}</span>
                </div>
                <div class="metric">
                    <strong>Medium Drift:</strong> <span class="drift-medium">{medium_drift}</span>
                </div>
                <div class="metric">
                    <strong>Low Drift:</strong> <span class="drift-low">{low_drift}</span>
                </div>
            </div>
            
            <h2>Feature-Level Drift Analysis</h2>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Drift Score</th>
                    <th>Drift Detected</th>
                    <th>Status</th>
                </tr>
        """
        
        # Add feature rows
        for feature, drift_info in feature_drift.items():
            drift_score = drift_info.get('drift_score', 0)
            drift_detected = drift_info.get('drift_detected', False)
            
            if drift_score > 0.5:
                status_class = "drift-high"
                status = "High Drift"
            elif drift_score > 0.2:
                status_class = "drift-medium"
                status = "Medium Drift"
            elif drift_score > 0.1:
                status_class = "drift-low"
                status = "Low Drift"
            else:
                status_class = "drift-none"
                status = "No Drift"
            
            html_content += f"""
                <tr>
                    <td>{feature}</td>
                    <td>{drift_score:.3f}</td>
                    <td>{drift_detected}</td>
                    <td class="{status_class}">{status}</td>
                </tr>
            """
        
        html_content += """
            </table>
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
        
        logger.info(f"Sending {alert_type} alert")
        
        try:
            # Prepare alert message
            summary = drift_results.get('summary', {})
            message = f"""
            🚨 DRIFT ALERT - Chinese Produce Forecasting System
            
            Alert Type: {alert_type.upper()}
            Timestamp: {drift_results.get('timestamp')}
            
            Summary:
            - Overall Drift Detected: {drift_results.get('overall_drift_detected')}
            - Drift Score: {drift_results.get('drift_score', 0):.3f}
            - Features Analyzed: {summary.get('total_features', 0)}
            - Drifted Features: {summary.get('drifted_features', 0)}
            - Drift Ratio: {summary.get('drift_ratio', 0):.2%}
            
            Action Required: Please review the model performance and consider retraining.
            """
            
            # Send CloudWatch metric
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
            
            # Send SNS notification if configured
            alerts_config = self.monitoring_config.get('alerts', {})
            sns_topic = alerts_config.get('sns_topic')
            
            if sns_topic:
                self.sns.publish(
                    TopicArn=sns_topic,
                    Message=message,
                    Subject=f"Drift Alert - {alert_type}"
                )
                logger.info("SNS alert sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending drift alert: {e}")

    def save_drift_state(self, output_path: str = "drift_state.json"):
        """Save current drift detection state"""
        state = {
            'reference_stats': self.reference_stats,
            'baseline_performance': self.baseline_performance,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Drift state saved to: {output_path}")

    def load_drift_state(self, state_path: str = "drift_state.json") -> bool:
        """Load drift detection state"""
        try:
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                self.reference_stats = state.get('reference_stats', {})
                self.baseline_performance = state.get('baseline_performance', {})
                
                logger.info(f"Drift state loaded from: {state_path}")
                return True
            
        except Exception as e:
            logger.error(f"Error loading drift state: {e}")
        
        return False


def main():
    """Main function for testing drift detection"""
    parser = argparse.ArgumentParser(description='Drift Detection for Chinese Produce Forecasting')
    parser.add_argument('--config', required=True, help='Path to config.yaml')
    parser.add_argument('--action', required=True,
                       choices=['detect', 'monitor', 'alert-test'],
                       help='Action to perform')
    parser.add_argument('--reference-data', help='Path to reference data file')
    parser.add_argument('--current-data', help='Path to current data file')
    parser.add_argument('--method', default='statistical',
                       choices=['statistical', 'evidently', 'ks_test', 'population_stability'],
                       help='Drift detection method')
    
    args = parser.parse_args()
    
    try:
        # Initialize drift detector
        detector = DriftDetector(args.config)
        
        if args.action == 'detect':
            if not args.reference_data or not args.current_data:
                print("Error: --reference-data and --current-data required for detect action")
                return
            
            # Load reference data
            detector.load_reference_data(args.reference_data)
            
            # Load current data
            if args.current_data.endswith('.parquet'):
                current_data = pd.read_parquet(args.current_data)
            else:
                current_data = pd.read_csv(args.current_data)
            
            # Detect drift
            results = detector.detect_data_drift(current_data, args.method)
            
            # Generate report
            report_path = detector.generate_drift_report(results)
            
            print(f"\nDrift Detection Results:")
            print(f"Overall Drift Detected: {results['overall_drift_detected']}")
            print(f"Drift Score: {results['drift_score']:.3f}")
            print(f"Method: {results['method']}")
            print(f"Report saved to: {report_path}")
            
            # Send alert if drift detected
            if results['overall_drift_detected']:
                detector.send_drift_alert(results)
        
        elif args.action == 'monitor':
            print("Starting continuous drift monitoring...")
            # This would typically run as a scheduled job
            print("Monitor mode - implement scheduling logic here")
        
        elif args.action == 'alert-test':
            # Test alert system
            test_results = {
                'timestamp': datetime.now().isoformat(),
                'overall_drift_detected': True,
                'drift_score': 0.75,
                'summary': {
                    'total_features': 50,
                    'drifted_features': 15,
                    'drift_ratio': 0.3
                }
            }
            detector.send_drift_alert(test_results, "test_alert")
            print("Test alert sent")
        
        logger.info("Drift detection operation completed successfully")
        
    except Exception as e:
        logger.error(f"Drift detection operation failed: {e}")
        print(f"\n❌ Operation failed: {e}")


if __name__ == "__main__":
    main()