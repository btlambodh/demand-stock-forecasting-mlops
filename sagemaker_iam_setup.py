#!/usr/bin/env python3
"""
SageMaker IAM Role Setup & Deployment Fix
Automatically creates necessary IAM roles and fixes deployment issues

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import boto3
import json
import time
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SageMakerIAMSetup:
    """Setup and manage SageMaker IAM roles and permissions"""
    
    def __init__(self, region: str = 'us-east-1', account_id: str = '346761359662'):
        self.region = region
        self.account_id = account_id
        
        # Initialize AWS clients
        self.iam_client = boto3.client('iam', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        self.sts_client = boto3.client('sts', region_name=region)
        
        # Verify account ID
        try:
            actual_account = self.sts_client.get_caller_identity()['Account']
            if actual_account != account_id:
                logger.warning(f"Account ID mismatch! Expected: {account_id}, Actual: {actual_account}")
                self.account_id = actual_account
        except Exception as e:
            logger.error(f"Failed to verify account ID: {e}")
    
    def check_role_exists(self, role_name: str) -> bool:
        """Check if IAM role exists"""
        try:
            self.iam_client.get_role(RoleName=role_name)
            logger.info(f" Role {role_name} exists")
            return True
        except self.iam_client.exceptions.NoSuchEntityException:
            logger.info(f" Role {role_name} does not exist")
            return False
        except Exception as e:
            logger.error(f"Error checking role {role_name}: {e}")
            return False
    
    def create_sagemaker_execution_role(self) -> str:
        """Create SageMaker execution role with proper permissions"""
        role_name = "SageMakerExecutionRole"
        
        # Trust policy for SageMaker
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": [
                            "sagemaker.amazonaws.com",
                            "glue.amazonaws.com"
                        ]
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        # Custom policy for Chinese Produce Forecasting
        custom_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        # SageMaker permissions
                        "sagemaker:*",
                        
                        # S3 permissions
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject",
                        "s3:ListBucket",
                        "s3:GetBucketLocation",
                        "s3:ListAllMyBuckets",
                        "s3:CreateBucket",
                        "s3:GetBucketCors",
                        "s3:PutBucketCors",
                        
                        # CloudWatch permissions
                        "cloudwatch:PutMetricData",
                        "cloudwatch:GetMetricStatistics",
                        "cloudwatch:ListMetrics",
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:DescribeLogStreams",
                        "logs:PutLogEvents",
                        "logs:GetLogEvents",
                        
                        # ECR permissions (for custom containers)
                        "ecr:GetAuthorizationToken",
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:BatchGetImage",
                        
                        # EC2 permissions (for VPC)
                        "ec2:CreateNetworkInterface",
                        "ec2:CreateNetworkInterfacePermission",
                        "ec2:DeleteNetworkInterface",
                        "ec2:DeleteNetworkInterfacePermission",
                        "ec2:DescribeNetworkInterfaces",
                        "ec2:DescribeVpcs",
                        "ec2:DescribeDhcpOptions",
                        "ec2:DescribeSubnets",
                        "ec2:DescribeSecurityGroups",
                        
                        # Auto Scaling permissions
                        "application-autoscaling:*",
                        
                        # SNS permissions (for alerts)
                        "sns:Publish"
                    ],
                    "Resource": "*"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "s3:*"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::chinese-produce-forecast-{self.account_id}",
                        f"arn:aws:s3:::chinese-produce-forecast-{self.account_id}/*",
                        f"arn:aws:s3:::sagemaker-{self.region}-{self.account_id}",
                        f"arn:aws:s3:::sagemaker-{self.region}-{self.account_id}/*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "iam:PassRole"
                    ],
                    "Resource": f"arn:aws:iam::{self.account_id}:role/*SageMaker*"
                }
            ]
        }
        
        try:
            if self.check_role_exists(role_name):
                logger.info(f"Role {role_name} already exists, updating policies...")
                role_arn = f"arn:aws:iam::{self.account_id}:role/{role_name}"
            else:
                # Create the role
                logger.info(f"Creating role {role_name}...")
                response = self.iam_client.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(trust_policy),
                    Description="SageMaker execution role for Chinese Produce Forecasting",
                    Tags=[
                        {'Key': 'Project', 'Value': 'chinese-produce-forecast'},
                        {'Key': 'Purpose', 'Value': 'sagemaker-execution'},
                        {'Key': 'Environment', 'Value': 'production'}
                    ]
                )
                role_arn = response['Role']['Arn']
                logger.info(f" Created role: {role_arn}")
            
            # Attach AWS managed policies
            managed_policies = [
                'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
                'arn:aws:iam::aws:policy/AmazonS3FullAccess',
                'arn:aws:iam::aws:policy/CloudWatchFullAccess'
            ]
            
            for policy_arn in managed_policies:
                try:
                    self.iam_client.attach_role_policy(
                        RoleName=role_name,
                        PolicyArn=policy_arn
                    )
                    logger.info(f" Attached managed policy: {policy_arn}")
                except Exception as e:
                    if "already attached" in str(e).lower():
                        logger.info(f"Policy already attached: {policy_arn}")
                    else:
                        logger.warning(f"Failed to attach policy {policy_arn}: {e}")
            
            # Create and attach custom policy
            custom_policy_name = f"{role_name}CustomPolicy"
            try:
                self.iam_client.put_role_policy(
                    RoleName=role_name,
                    PolicyName=custom_policy_name,
                    PolicyDocument=json.dumps(custom_policy)
                )
                logger.info(f" Attached custom policy: {custom_policy_name}")
            except Exception as e:
                logger.warning(f"Failed to attach custom policy: {e}")
            
            # Wait for role to propagate
            logger.info("Waiting for role to propagate...")
            time.sleep(10)
            
            return role_arn
            
        except Exception as e:
            logger.error(f"Failed to create/update role: {e}")
            raise
    
    def setup_s3_buckets(self) -> Dict[str, str]:
        """Setup required S3 buckets with proper permissions"""
        buckets = {
            'project_bucket': f'chinese-produce-forecast-{self.account_id}',
            'sagemaker_bucket': f'sagemaker-{self.region}-{self.account_id}'
        }
        
        bucket_info = {}
        
        for bucket_type, bucket_name in buckets.items():
            try:
                # Check if bucket exists
                try:
                    self.s3_client.head_bucket(Bucket=bucket_name)
                    logger.info(f" Bucket {bucket_name} already exists")
                    bucket_info[bucket_type] = bucket_name
                    continue
                except:
                    pass
                
                # Create bucket
                if self.region == 'us-east-1':
                    self.s3_client.create_bucket(Bucket=bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': self.region}
                    )
                
                logger.info(f" Created bucket: {bucket_name}")
                
                # Set bucket policy for SageMaker access
                bucket_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Sid": "SageMakerAccess",
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "sagemaker.amazonaws.com"
                            },
                            "Action": [
                                "s3:GetObject",
                                "s3:PutObject",
                                "s3:DeleteObject",
                                "s3:ListBucket"
                            ],
                            "Resource": [
                                f"arn:aws:s3:::{bucket_name}",
                                f"arn:aws:s3:::{bucket_name}/*"
                            ]
                        },
                        {
                            "Sid": "SageMakerExecutionRoleAccess",
                            "Effect": "Allow",
                            "Principal": {
                                "AWS": f"arn:aws:iam::{self.account_id}:role/SageMakerExecutionRole"
                            },
                            "Action": "s3:*",
                            "Resource": [
                                f"arn:aws:s3:::{bucket_name}",
                                f"arn:aws:s3:::{bucket_name}/*"
                            ]
                        }
                    ]
                }
                
                self.s3_client.put_bucket_policy(
                    Bucket=bucket_name,
                    Policy=json.dumps(bucket_policy)
                )
                
                # Enable versioning
                self.s3_client.put_bucket_versioning(
                    Bucket=bucket_name,
                    VersioningConfiguration={'Status': 'Enabled'}
                )
                
                bucket_info[bucket_type] = bucket_name
                logger.info(f" Configured bucket: {bucket_name}")
                
            except Exception as e:
                logger.error(f"Failed to setup bucket {bucket_name}: {e}")
                # For SageMaker default bucket, try to get it instead
                if bucket_type == 'sagemaker_bucket':
                    try:
                        import sagemaker
                        session = sagemaker.Session()
                        default_bucket = session.default_bucket()
                        bucket_info[bucket_type] = default_bucket
                        logger.info(f" Using SageMaker default bucket: {default_bucket}")
                    except Exception as e2:
                        logger.error(f"Failed to get SageMaker default bucket: {e2}")
        
        return bucket_info
    
    def test_role_permissions(self, role_arn: str) -> bool:
        """Test if the role has necessary permissions"""
        logger.info("Testing role permissions...")
        
        try:
            # Test assume role
            sts_client = boto3.client('sts', region_name=self.region)
            response = sts_client.assume_role(
                RoleArn=role_arn,
                RoleSessionName="test-session"
            )
            
            # Test SageMaker permissions with assumed role
            credentials = response['Credentials']
            sagemaker_client = boto3.client(
                'sagemaker',
                region_name=self.region,
                aws_access_key_id=credentials['AccessKeyId'],
                aws_secret_access_key=credentials['SecretAccessKey'],
                aws_session_token=credentials['SessionToken']
            )
            
            # Test list endpoints (basic SageMaker permission)
            sagemaker_client.list_endpoints()
            logger.info(" Role can access SageMaker")
            
            # Test S3 permissions
            s3_client = boto3.client(
                's3',
                region_name=self.region,
                aws_access_key_id=credentials['AccessKeyId'],
                aws_secret_access_key=credentials['SecretAccessKey'],
                aws_session_token=credentials['SessionToken']
            )
            
            s3_client.list_buckets()
            logger.info(" Role can access S3")
            
            return True
            
        except Exception as e:
            logger.error(f"Role permission test failed: {e}")
            return False
    
    def fix_deployment_script_issues(self) -> str:
        """Generate fixes for common deployment script issues"""
        
        fixes = """
# FIXES FOR SAGEMAKER DEPLOYMENT ISSUES

## 1. Update config.yaml with correct role ARN
execution_role: "arn:aws:iam::{account_id}:role/SageMakerExecutionRole"

## 2. Ensure model files are properly packaged
# Create model.tar.gz from your trained model:
import tarfile
import os

def create_model_tar(model_path, output_path):
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(model_path, arcname="model.pkl")
    print(f"Created {output_path}")

# Usage:
create_model_tar("models/random_forest_model.pkl", "models/model.tar.gz")

## 3. Upload model to S3 properly
import boto3

def upload_model_to_s3(local_path, bucket, key):
    s3_client = boto3.client('s3')
    s3_client.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"

# Usage:
model_s3_uri = upload_model_to_s3(
    "models/model.tar.gz",
    "chinese-produce-forecast-{account_id}",
    "models/random_forest/model.tar.gz"
)

## 4. Fix endpoint naming (no underscores)
def clean_endpoint_name(name):
    return name.replace('_', '-').replace(' ', '-').lower()[:63]

## 5. Use correct instance types for your region
# For staging: ml.t2.medium or ml.m5.large
# For production: ml.m5.xlarge or ml.c5.xlarge

## 6. Enable proper logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## 7. Test with sample data first
test_data = {{
    "features": {{
        "Total_Quantity": 100.0,
        "Avg_Price": 15.5,
        "Transaction_Count": 10,
        "Month": 6,
        "DayOfWeek": 2,
        "IsWeekend": 0
    }}
}}
        """.format(account_id=self.account_id)
        
        return fixes
    
    def run_complete_setup(self) -> Dict[str, str]:
        """Run complete IAM and S3 setup"""
        logger.info("🚀 Starting complete SageMaker setup...")
        
        results = {
            'status': 'failed',
            'role_arn': None,
            'buckets': {},
            'fixes': '',
            'issues': []
        }
        
        try:
            # 1. Create/verify SageMaker execution role
            logger.info("Step 1: Setting up SageMaker execution role...")
            role_arn = self.create_sagemaker_execution_role()
            results['role_arn'] = role_arn
            
            # 2. Setup S3 buckets
            logger.info("Step 2: Setting up S3 buckets...")
            buckets = self.setup_s3_buckets()
            results['buckets'] = buckets
            
            # 3. Test permissions
            logger.info("Step 3: Testing role permissions...")
            if self.test_role_permissions(role_arn):
                logger.info(" Role permissions test passed")
            else:
                results['issues'].append("Role permissions test failed")
            
            # 4. Generate fixes
            logger.info("Step 4: Generating deployment fixes...")
            results['fixes'] = self.fix_deployment_script_issues()
            
            results['status'] = 'success'
            logger.info(" Complete setup finished successfully!")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            results['issues'].append(str(e))
        
        return results


def main():
    """Main function to run the setup"""
    print("🔧 SageMaker IAM Role & S3 Setup")
    print("=" * 50)
    
    # Initialize setup
    setup = SageMakerIAMSetup()
    
    # Run complete setup
    results = setup.run_complete_setup()
    
    # Print results
    print("\n📋 SETUP RESULTS")
    print("=" * 50)
    print(f"Status: {results['status'].upper()}")
    
    if results['role_arn']:
        print(f" SageMaker Role ARN: {results['role_arn']}")
    
    if results['buckets']:
        print(" S3 Buckets:")
        for bucket_type, bucket_name in results['buckets'].items():
            print(f"   {bucket_type}: {bucket_name}")
    
    if results['issues']:
        print(" Issues found:")
        for issue in results['issues']:
            print(f"   - {issue}")
    
    print("\n NEXT STEPS:")
    print("1. Update your config.yaml with the new role ARN")
    print("2. Ensure your model files are in the correct S3 bucket")
    print("3. Run the deployment command again")
    
    if results['status'] == 'success':
        print("\n Setup completed! You can now run your SageMaker deployment.")
    else:
        print("\n Setup had issues. Please check the logs above.")
    
    return results


if __name__ == "__main__":
    main()
