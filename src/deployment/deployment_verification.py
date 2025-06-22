#!/usr/bin/env python3
"""
Deployment Verification Script for Chinese Produce Market Forecasting
Comprehensive verification of SageMaker deployment

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import os
import sys
import json
import time
import subprocess
import yaml
from datetime import datetime
from typing import Dict, List, Tuple
import boto3
import pandas as pd


class DeploymentVerifier:
    """Comprehensive deployment verification"""
    
    def __init__(self, config_path: str):
        """Initialize verifier"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.aws_config = self.config['aws']
        self.region = self.aws_config['region']
        self.s3_client = boto3.client('s3', region_name=self.region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.region)
        self.iam_client = boto3.client('iam', region_name=self.region)
        
        self.verification_results = []
    
    def log_result(self, test_name: str, status: str, message: str, details: Dict = None):
        """Log verification result"""
        result = {
            'test_name': test_name,
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.verification_results.append(result)
        
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {message}")
    
    def verify_aws_credentials(self) -> bool:
        """Verify AWS credentials and permissions"""
        try:
            # Check AWS credentials
            sts_client = boto3.client('sts', region_name=self.region)
            identity = sts_client.get_caller_identity()
            
            self.log_result(
                "AWS Credentials",
                "PASS",
                f"AWS credentials valid. Account: {identity['Account']}",
                {"account_id": identity['Account'], "user_arn": identity['Arn']}
            )
            
            # Test S3 access
            try:
                self.s3_client.list_objects_v2(Bucket=self.aws_config['s3']['bucket_name'], MaxKeys=1)
                self.log_result("S3 Access", "PASS", "S3 bucket accessible")
            except Exception as e:
                self.log_result("S3 Access", "FAIL", f"S3 access error: {str(e)}")
                return False
            
            # Test SageMaker permissions
            try:
                self.sagemaker_client.list_endpoints(MaxResults=1)
                self.log_result("SageMaker Access", "PASS", "SageMaker permissions valid")
            except Exception as e:
                self.log_result("SageMaker Access", "FAIL", f"SageMaker access error: {str(e)}")
                return False
            
            return True
            
        except Exception as e:
            self.log_result("AWS Credentials", "FAIL", f"AWS credential error: {str(e)}")
            return False
    
    def verify_iam_role(self) -> bool:
        """Verify SageMaker execution role"""
        try:
            role_arn = self.aws_config['sagemaker']['execution_role']
            role_name = role_arn.split('/')[-1]
            
            # Check if role exists
            try:
                role = self.iam_client.get_role(RoleName=role_name)
                self.log_result(
                    "IAM Role Exists",
                    "PASS",
                    f"SageMaker role exists: {role_name}",
                    {"role_arn": role_arn}
                )
            except Exception as e:
                self.log_result("IAM Role Exists", "FAIL", f"Role not found: {str(e)}")
                return False
            
            # Check role policies
            try:
                attached_policies = self.iam_client.list_attached_role_policies(RoleName=role_name)
                required_policies = ['AmazonSageMakerFullAccess', 'AmazonS3FullAccess']
                
                policy_arns = [p['PolicyArn'] for p in attached_policies['AttachedPolicies']]
                missing_policies = [p for p in required_policies if not any(p in arn for arn in policy_arns)]
                
                if missing_policies:
                    self.log_result(
                        "IAM Role Policies",
                        "WARN",
                        f"Missing policies: {missing_policies}",
                        {"attached_policies": policy_arns}
                    )
                else:
                    self.log_result("IAM Role Policies", "PASS", "Required policies attached")
                
            except Exception as e:
                self.log_result("IAM Role Policies", "WARN", f"Could not verify policies: {str(e)}")
            
            return True
            
        except Exception as e:
            self.log_result("IAM Role", "FAIL", f"IAM role verification failed: {str(e)}")
            return False
    
    def verify_model_files(self) -> bool:
        """Verify required model files exist"""
        try:
            models_dir = "models"
            
            if not os.path.exists(models_dir):
                self.log_result("Model Directory", "FAIL", "models/ directory not found")
                return False
            
            # Look for model files
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            
            if not model_files:
                self.log_result("Model Files", "FAIL", "No .pkl model files found in models/")
                return False
            
            # Check for best_model.pkl specifically
            if 'best_model.pkl' in model_files:
                model_path = os.path.join(models_dir, 'best_model.pkl')
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                
                self.log_result(
                    "Best Model File",
                    "PASS",
                    f"best_model.pkl found ({size_mb:.2f} MB)",
                    {"model_files": model_files, "size_mb": size_mb}
                )
                return True
            else:
                # Use any available model
                model_file = model_files[0]
                model_path = os.path.join(models_dir, model_file)
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                
                self.log_result(
                    "Model File",
                    "PASS",
                    f"Model file found: {model_file} ({size_mb:.2f} MB)",
                    {"model_files": model_files, "selected_model": model_file}
                )
                return True
                
        except Exception as e:
            self.log_result("Model Files", "FAIL", f"Model file verification failed: {str(e)}")
            return False
    
    def verify_config_file(self) -> bool:
        """Verify configuration file"""
        try:
            required_sections = ['aws', 'deployment']
            missing_sections = [s for s in required_sections if s not in self.config]
            
            if missing_sections:
                self.log_result(
                    "Config Structure",
                    "FAIL",
                    f"Missing config sections: {missing_sections}"
                )
                return False
            
            # Check AWS config
            aws_required = ['region', 'sagemaker', 's3']
            aws_missing = [k for k in aws_required if k not in self.aws_config]
            
            if aws_missing:
                self.log_result(
                    "AWS Config",
                    "FAIL",
                    f"Missing AWS config: {aws_missing}"
                )
                return False
            
            self.log_result("Config File", "PASS", "Configuration file is valid")
            return True
            
        except Exception as e:
            self.log_result("Config File", "FAIL", f"Config verification failed: {str(e)}")
            return False
    
    def verify_dependencies(self) -> bool:
        """Verify Python dependencies"""
        try:
            required_packages = [
                'boto3', 'sagemaker', 'pandas', 'numpy', 
                'scikit-learn', 'joblib', 'yaml'
            ]
            
            missing_packages = []
            
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                self.log_result(
                    "Python Dependencies",
                    "FAIL",
                    f"Missing packages: {missing_packages}"
                )
                return False
            
            self.log_result("Python Dependencies", "PASS", "All required packages installed")
            return True
            
        except Exception as e:
            self.log_result("Python Dependencies", "FAIL", f"Dependency check failed: {str(e)}")
            return False
    
    def run_deployment_test(self, model_path: str, endpoint_name: str) -> bool:
        """Run actual deployment test"""
        try:
            print(f"\n🚀 Running deployment test...")
            
            # Use the fixed deployment script
            cmd = [
                sys.executable, 'fixed_sagemaker_deploy.py',
                '--config', 'config.yaml',
                '--action', 'deploy',
                '--model-path', model_path,
                '--model-name', 'test-chinese-produce-forecaster',
                '--endpoint-name', endpoint_name,
                '--environment', 'staging',
                '--instance-type', 'ml.t2.medium'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            if result.returncode == 0:
                self.log_result(
                    "Deployment Test",
                    "PASS",
                    "Deployment completed successfully",
                    {"stdout": result.stdout[-500:], "stderr": result.stderr[-500:]}  # Last 500 chars
                )
                return True
            else:
                self.log_result(
                    "Deployment Test",
                    "FAIL",
                    f"Deployment failed (exit code: {result.returncode})",
                    {"stdout": result.stdout, "stderr": result.stderr}
                )
                return False
                
        except subprocess.TimeoutExpired:
            self.log_result("Deployment Test", "FAIL", "Deployment timed out after 30 minutes")
            return False
        except Exception as e:
            self.log_result("Deployment Test", "FAIL", f"Deployment test failed: {str(e)}")
            return False
    
    def run_endpoint_test(self, endpoint_name: str) -> bool:
        """Test deployed endpoint"""
        try:
            print(f"\n🧪 Testing endpoint...")
            
            cmd = [
                sys.executable, 'fixed_sagemaker_deploy.py',
                '--config', 'config.yaml',
                '--action', 'test',
                '--endpoint-name', endpoint_name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                self.log_result(
                    "Endpoint Test",
                    "PASS",
                    "Endpoint test successful",
                    {"stdout": result.stdout[-500:]}
                )
                return True
            else:
                self.log_result(
                    "Endpoint Test",
                    "FAIL",
                    f"Endpoint test failed (exit code: {result.returncode})",
                    {"stdout": result.stdout, "stderr": result.stderr}
                )
                return False
                
        except subprocess.TimeoutExpired:
            self.log_result("Endpoint Test", "FAIL", "Endpoint test timed out")
            return False
        except Exception as e:
            self.log_result("Endpoint Test", "FAIL", f"Endpoint test failed: {str(e)}")
            return False
    
    def cleanup_test_resources(self, endpoint_name: str) -> bool:
        """Clean up test resources"""
        try:
            print(f"\n🧹 Cleaning up test resources...")
            
            cmd = [
                sys.executable, 'fixed_sagemaker_deploy.py',
                '--config', 'config.yaml',
                '--action', 'delete',
                '--endpoint-name', endpoint_name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.log_result("Cleanup", "PASS", "Test resources cleaned up")
                return True
            else:
                self.log_result("Cleanup", "WARN", "Cleanup may have failed")
                return False
                
        except Exception as e:
            self.log_result("Cleanup", "WARN", f"Cleanup failed: {str(e)}")
            return False
    
    def generate_verification_report(self) -> Dict:
        """Generate comprehensive verification report"""
        total_tests = len(self.verification_results)
        passed_tests = len([r for r in self.verification_results if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.verification_results if r['status'] == 'FAIL'])
        warned_tests = len([r for r in self.verification_results if r['status'] == 'WARN'])
        
        report = {
            'verification_summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'warnings': warned_tests,
                'success_rate': f"{(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%",
                'overall_status': 'PASS' if failed_tests == 0 else 'FAIL',
                'verification_time': datetime.now().isoformat()
            },
            'test_results': self.verification_results,
            'recommendations': self.get_recommendations()
        }
        
        return report
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations based on verification results"""
        recommendations = []
        
        failed_tests = [r for r in self.verification_results if r['status'] == 'FAIL']
        
        for test in failed_tests:
            if 'AWS Credentials' in test['test_name']:
                recommendations.append("Run 'aws configure' to set up AWS credentials")
            elif 'S3 Access' in test['test_name']:
                recommendations.append("Check S3 bucket permissions and ensure bucket exists")
            elif 'IAM Role' in test['test_name']:
                recommendations.append("Run the IAM setup script: python sagemaker_iam_setup.py")
            elif 'Model Files' in test['test_name']:
                recommendations.append("Train models first: python train_model.py --config config.yaml")
            elif 'Deployment' in test['test_name']:
                recommendations.append("Check deployment logs and fix reported issues")
        
        if not recommendations:
            recommendations.append("All verifications passed! Ready for production deployment.")
        
        return recommendations
    
    def run_full_verification(self, run_deployment_test: bool = True) -> Dict:
        """Run complete verification process"""
        print("="*60)
        print("CHINESE PRODUCE FORECASTING - M3 DEPLOYMENT VERIFICATION")
        print("="*60)
        
        # Pre-deployment checks
        print("\n📋 Pre-deployment Verification:")
        self.verify_config_file()
        self.verify_dependencies()
        self.verify_aws_credentials()
        self.verify_iam_role()
        self.verify_model_files()
        
        # Find best model file
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            if 'best_model.pkl' in model_files:
                model_path = os.path.join(models_dir, 'best_model.pkl')
            elif model_files:
                model_path = os.path.join(models_dir, model_files[0])
            else:
                model_path = None
        else:
            model_path = None
        
        # Deployment test (optional)
        if run_deployment_test and model_path:
            test_endpoint_name = f"test-endpoint-{int(time.time())}"
            
            deployment_success = self.run_deployment_test(model_path, test_endpoint_name)
            
            if deployment_success:
                # Test the endpoint
                endpoint_success = self.run_endpoint_test(test_endpoint_name)
                
                # Cleanup
                self.cleanup_test_resources(test_endpoint_name)
        
        # Generate report
        report = self.generate_verification_report()
        
        # Print summary
        print("\n" + "="*60)
        print("VERIFICATION SUMMARY")
        print("="*60)
        print(f"Total Tests: {report['verification_summary']['total_tests']}")
        print(f"Passed: {report['verification_summary']['passed']}")
        print(f"Failed: {report['verification_summary']['failed']}")
        print(f"Warnings: {report['verification_summary']['warnings']}")
        print(f"Success Rate: {report['verification_summary']['success_rate']}")
        print(f"Overall Status: {report['verification_summary']['overall_status']}")
        
        if report['recommendations']:
            print(f"\n📝 Recommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        # Save report
        with open('verification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: verification_report.json")
        
        return report


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify SageMaker deployment readiness')
    parser.add_argument('--config', default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--skip-deployment', action='store_true', 
                       help='Skip actual deployment test (faster verification)')
    parser.add_argument('--report-only', action='store_true',
                       help='Only generate report from previous verification')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    verifier = DeploymentVerifier(args.config)
    
    if args.report_only:
        # Just generate a basic report
        report = verifier.generate_verification_report()
        print("📄 Verification report generated")
    else:
        # Run full verification
        run_deployment = not args.skip_deployment
        report = verifier.run_full_verification(run_deployment_test=run_deployment)
    
    # Exit with appropriate code
    overall_status = report['verification_summary']['overall_status']
    sys.exit(0 if overall_status == 'PASS' else 1)


if __name__ == "__main__":
    main()