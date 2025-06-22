#!/usr/bin/env python3
"""
AWS Setup Diagnostics for SageMaker Deployment
Checks IAM roles, S3 permissions, and provides setup guidance
"""

import boto3
import json
import yaml
from botocore.exceptions import ClientError, NoCredentialsError

def check_aws_credentials():
    """Check if AWS credentials are configured"""
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print("✅ AWS Credentials configured")
        print(f"   Account ID: {identity['Account']}")
        print(f"   User/Role: {identity['Arn']}")
        return True
    except NoCredentialsError:
        print("❌ AWS credentials not configured")
        print("   Run: aws configure")
        return False
    except Exception as e:
        print(f"❌ Error checking credentials: {e}")
        return False

def check_sagemaker_role(role_arn):
    """Check if SageMaker execution role exists and has proper permissions"""
    try:
        iam = boto3.client('iam')
        
        # Extract role name from ARN
        role_name = role_arn.split('/')[-1]
        
        # Check if role exists
        try:
            role = iam.get_role(RoleName=role_name)
            print(f"✅ SageMaker role exists: {role_name}")
            
            # Check trust policy
            trust_policy = role['Role']['AssumeRolePolicyDocument']
            sagemaker_trusted = False
            
            for statement in trust_policy.get('Statement', []):
                principal = statement.get('Principal', {})
                if isinstance(principal, dict):
                    service = principal.get('Service', [])
                    if isinstance(service, str):
                        service = [service]
                    if 'sagemaker.amazonaws.com' in service:
                        sagemaker_trusted = True
                        break
            
            if sagemaker_trusted:
                print("   ✅ Trust policy allows SageMaker")
            else:
                print("   ❌ Trust policy missing SageMaker")
                return False
            
            # Check attached policies
            policies = iam.list_attached_role_policies(RoleName=role_name)
            policy_arns = [p['PolicyArn'] for p in policies['AttachedPolicies']]
            
            has_sagemaker_access = any('SageMaker' in arn for arn in policy_arns)
            if has_sagemaker_access:
                print("   ✅ Has SageMaker permissions")
            else:
                print("   ⚠️  No SageMaker permissions found")
            
            print(f"   Attached policies: {len(policy_arns)}")
            for arn in policy_arns:
                print(f"     - {arn.split('/')[-1]}")
            
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchEntity':
                print(f"❌ SageMaker role does not exist: {role_name}")
                return False
            else:
                raise
                
    except Exception as e:
        print(f"❌ Error checking SageMaker role: {e}")
        return False

def check_s3_bucket(bucket_name, region):
    """Check if S3 bucket exists and is accessible"""
    try:
        s3 = boto3.client('s3', region_name=region)
        
        # Check if bucket exists
        try:
            s3.head_bucket(Bucket=bucket_name)
            print(f"✅ S3 bucket exists: {bucket_name}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                print(f"❌ S3 bucket does not exist: {bucket_name}")
                return False
            elif error_code == '403':
                print(f"❌ No access to S3 bucket: {bucket_name}")
                return False
            else:
                raise
        
        # Check bucket region
        try:
            bucket_region = s3.get_bucket_location(Bucket=bucket_name)['LocationConstraint']
            if bucket_region is None:
                bucket_region = 'us-east-1'  # Default for us-east-1
            
            if bucket_region == region:
                print(f"   ✅ Bucket in correct region: {region}")
            else:
                print(f"   ⚠️  Bucket in different region: {bucket_region} (expected: {region})")
        except:
            print("   ⚠️  Could not determine bucket region")
        
        # Test write permissions
        try:
            test_key = "sagemaker-test/test-file.txt"
            s3.put_object(Bucket=bucket_name, Key=test_key, Body=b"test")
            s3.delete_object(Bucket=bucket_name, Key=test_key)
            print("   ✅ Write permissions confirmed")
        except Exception as e:
            print(f"   ❌ No write permissions: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking S3 bucket: {e}")
        return False

def create_sagemaker_role_instructions(account_id, bucket_name):
    """Provide instructions for creating SageMaker role"""
    print("\n🔧 To create the SageMaker execution role:")
    print("="*60)
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    s3_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:GetObjectVersion", 
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                    "s3:ListBucketVersions"
                ],
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*"
                ]
            }
        ]
    }
    
    print("\n1. Save trust policy to sagemaker-trust-policy.json:")
    print(json.dumps(trust_policy, indent=2))
    
    print("\n2. Save S3 policy to sagemaker-s3-policy.json:")
    print(json.dumps(s3_policy, indent=2))
    
    print("\n3. Run these AWS CLI commands:")
    print(f"""
# Create the role
aws iam create-role \\
    --role-name SageMakerExecutionRole \\
    --assume-role-policy-document file://sagemaker-trust-policy.json

# Attach AWS managed policy
aws iam attach-role-policy \\
    --role-name SageMakerExecutionRole \\
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

# Create custom S3 policy
aws iam create-policy \\
    --policy-name SageMakerS3AccessPolicy \\
    --policy-document file://sagemaker-s3-policy.json

# Attach S3 policy
aws iam attach-role-policy \\
    --role-name SageMakerExecutionRole \\
    --policy-arn arn:aws:iam::{account_id}:policy/SageMakerS3AccessPolicy
""")

def create_s3_bucket_instructions(bucket_name, region):
    """Provide instructions for creating S3 bucket"""
    print(f"\n🪣 To create the S3 bucket:")
    print("="*60)
    print(f"""
# Create S3 bucket
aws s3 mb s3://{bucket_name} --region {region}

# Verify bucket creation
aws s3 ls s3://{bucket_name}
""")

def main():
    """Main diagnostic function"""
    print("🔍 AWS Setup Diagnostics for SageMaker")
    print("="*60)
    
    # Load config
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        aws_config = config['aws']
        account_id = aws_config['account_id']
        region = aws_config['region']
        bucket_name = aws_config['s3']['bucket_name']
        role_arn = aws_config['sagemaker']['execution_role']
        
        print(f"Account ID: {account_id}")
        print(f"Region: {region}")
        print(f"S3 Bucket: {bucket_name}")
        print(f"IAM Role: {role_arn}")
        print()
        
    except Exception as e:
        print(f"❌ Error loading config.yaml: {e}")
        return
    
    # Run diagnostics
    issues_found = []
    
    print("1. Checking AWS credentials...")
    if not check_aws_credentials():
        issues_found.append("AWS credentials")
    print()
    
    print("2. Checking SageMaker IAM role...")
    if not check_sagemaker_role(role_arn):
        issues_found.append("SageMaker IAM role")
    print()
    
    print("3. Checking S3 bucket...")
    if not check_s3_bucket(bucket_name, region):
        issues_found.append("S3 bucket")
    print()
    
    # Provide solutions
    if issues_found:
        print("❌ Issues found:", ", ".join(issues_found))
        print()
        
        if "SageMaker IAM role" in issues_found:
            create_sagemaker_role_instructions(account_id, bucket_name)
        
        if "S3 bucket" in issues_found:
            create_s3_bucket_instructions(bucket_name, region)
        
        print("\n🔄 After fixing issues, run this again to verify:")
        print("python aws_diagnostics.py")
        
    else:
        print("🎉 All AWS setup looks good!")
        print("You can proceed with SageMaker deployment.")

if __name__ == "__main__":
    main()