#!/usr/bin/env python3
"""
AWS Role & Permission Checker
Check current AWS permissions and find existing SageMaker roles

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import boto3
import json
from botocore.exceptions import ClientError, NoCredentialsError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWSRoleChecker:
    """Check AWS permissions and find suitable roles"""
    
    def __init__(self):
        try:
            self.sts_client = boto3.client('sts')
            self.iam_client = boto3.client('iam')
            self.sagemaker_client = boto3.client('sagemaker')
            self.identity = self.sts_client.get_caller_identity()
            logger.info(" AWS clients initialized successfully")
        except NoCredentialsError:
            logger.error(" No AWS credentials found. Please configure AWS CLI.")
            raise
        except Exception as e:
            logger.error(f" Failed to initialize AWS clients: {e}")
            raise
    
    def check_current_identity(self):
        """Check current AWS identity and permissions"""
        print(" CURRENT AWS IDENTITY")
        print("=" * 50)
        
        try:
            user_id = self.identity.get('UserId', 'Unknown')
            account = self.identity.get('Account', 'Unknown')
            arn = self.identity.get('Arn', 'Unknown')
            
            print(f"User ID: {user_id}")
            print(f"Account: {account}")
            print(f"ARN: {arn}")
            
            # Check if running from SageMaker
            if 'sagemaker' in arn.lower() or 'SageMaker' in arn:
                print("  WARNING: You're running from SageMaker environment")
                print("   This has limited IAM permissions for security")
                print("   Consider running from local machine or AWS CloudShell")
            
            return self.identity
            
        except Exception as e:
            print(f" Error checking identity: {e}")
            return None
    
    def check_iam_permissions(self):
        """Check if current user can create/manage IAM roles"""
        print("\n IAM PERMISSIONS CHECK")
        print("=" * 50)
        
        permissions_to_test = [
            ('iam:CreateRole', 'Create IAM roles'),
            ('iam:ListRoles', 'List IAM roles'),
            ('iam:GetRole', 'Get IAM role details'),
            ('iam:AttachRolePolicy', 'Attach policies to roles'),
            ('iam:PassRole', 'Pass roles to services')
        ]
        
        results = {}
        
        for action, description in permissions_to_test:
            try:
                # Try a test action that would use this permission
                if action == 'iam:ListRoles':
                    self.iam_client.list_roles(MaxItems=1)
                    results[action] = True
                    print(f" {description}: ALLOWED")
                elif action == 'iam:CreateRole':
                    # We can't actually test this without trying to create a role
                    # So we'll assume it's not allowed if we're in SageMaker
                    arn = self.identity.get('Arn', '')
                    if 'sagemaker' in arn.lower():
                        results[action] = False
                        print(f" {description}: DENIED (SageMaker limitation)")
                    else:
                        results[action] = True
                        print(f" {description}: LIKELY ALLOWED")
                else:
                    # For other permissions, assume they're available if list works
                    results[action] = results.get('iam:ListRoles', False)
                    status = " LIKELY ALLOWED" if results[action] else " LIKELY DENIED"
                    print(f"{status} {description}")
                    
            except ClientError as e:
                results[action] = False
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                print(f" {description}: DENIED ({error_code})")
            except Exception as e:
                results[action] = False
                print(f" {description}: ERROR ({str(e)[:50]}...)")
        
        return results
    
    def find_existing_sagemaker_roles(self):
        """Find existing SageMaker execution roles"""
        print("\n EXISTING SAGEMAKER ROLES")
        print("=" * 50)
        
        try:
            response = self.iam_client.list_roles()
            sagemaker_roles = []
            
            for role in response['Roles']:
                role_name = role['RoleName']
                role_arn = role['Arn']
                
                # Check if it's a SageMaker role
                is_sagemaker_role = False
                trust_policy = role.get('AssumeRolePolicyDocument', {})
                
                # Check trust policy for SageMaker service
                if isinstance(trust_policy, dict):
                    statements = trust_policy.get('Statement', [])
                    for statement in statements:
                        principals = statement.get('Principal', {})
                        if isinstance(principals, dict):
                            services = principals.get('Service', [])
                            if isinstance(services, str):
                                services = [services]
                            if 'sagemaker.amazonaws.com' in services:
                                is_sagemaker_role = True
                                break
                
                # Also check role name patterns
                if ('sagemaker' in role_name.lower() or 
                    'SageMaker' in role_name or
                    role_name.startswith('AmazonSageMaker')):
                    is_sagemaker_role = True
                
                if is_sagemaker_role:
                    sagemaker_roles.append({
                        'name': role_name,
                        'arn': role_arn,
                        'created': role['CreateDate'].strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            if sagemaker_roles:
                print(f"Found {len(sagemaker_roles)} SageMaker roles:")
                for i, role in enumerate(sagemaker_roles, 1):
                    print(f"\n{i}. {role['name']}")
                    print(f"   ARN: {role['arn']}")
                    print(f"   Created: {role['created']}")
                    
                    # Check if role has required policies
                    self.check_role_policies(role['name'])
                
                # Recommend best role
                print(f"\n RECOMMENDATION:")
                best_role = sagemaker_roles[0]  # Use first one found
                print(f"Use this role ARN in your config.yaml:")
                print(f'execution_role: "{best_role["arn"]}"')
                
                return sagemaker_roles
            else:
                print(" No SageMaker roles found")
                print("You need to create one using the manual setup guide")
                return []
                
        except ClientError as e:
            print(f" Error listing roles: {e}")
            return []
    
    def check_role_policies(self, role_name):
        """Check what policies are attached to a role"""
        try:
            # Check attached managed policies
            managed_policies = self.iam_client.list_attached_role_policies(RoleName=role_name)
            
            required_policies = [
                'AmazonSageMakerFullAccess',
                'AmazonS3FullAccess',
                'CloudWatchFullAccess'
            ]
            
            attached_policy_names = [p['PolicyName'] for p in managed_policies['AttachedPolicies']]
            
            has_sagemaker_access = any('SageMaker' in policy for policy in attached_policy_names)
            has_s3_access = any('S3' in policy for policy in attached_policy_names)
            
            if has_sagemaker_access and has_s3_access:
                print(f"    Has required permissions")
            elif has_sagemaker_access:
                print(f"     Has SageMaker access, may need S3 permissions")
            else:
                print(f"    May need additional permissions")
                
        except Exception as e:
            print(f"   ❓ Could not check policies: {str(e)[:50]}...")
    
    def test_sagemaker_access(self):
        """Test SageMaker service access"""
        print("\n🧪 SAGEMAKER ACCESS TEST")
        print("=" * 50)
        
        try:
            # Try to list endpoints
            response = self.sagemaker_client.list_endpoints()
            endpoints = response.get('Endpoints', [])
            print(f" SageMaker access: OK ({len(endpoints)} endpoints found)")
            
            if endpoints:
                print("Existing endpoints:")
                for endpoint in endpoints[:3]:  # Show first 3
                    print(f"  - {endpoint['EndpointName']} ({endpoint['EndpointStatus']})")
            
            return True
            
        except ClientError as e:
            print(f" SageMaker access: DENIED ({e.response.get('Error', {}).get('Code', 'Unknown')})")
            return False
        except Exception as e:
            print(f" SageMaker access: ERROR ({str(e)[:50]}...)")
            return False
    
    def generate_config_update(self, roles):
        """Generate config.yaml update based on findings"""
        print("\n CONFIG.YAML UPDATE")
        print("=" * 50)
        
        if roles:
            best_role = roles[0]
            print("Add this to your config.yaml:")
            print()
            print("aws:")
            print("  sagemaker:")
            print(f'    execution_role: "{best_role["arn"]}"')
            print()
            return best_role["arn"]
        else:
            print("No suitable roles found. You need to:")
            print("1. Create a role manually using AWS Console")
            print("2. Or ask your AWS administrator to create one")
            print("3. Or use AWS CLI from your local machine")
            return None
    
    def run_complete_check(self):
        """Run all checks and provide recommendations"""
        print(" AWS ROLE & PERMISSION CHECKER")
        print("=" * 60)
        
        # Check current identity
        identity = self.check_current_identity()
        
        # Check IAM permissions
        iam_perms = self.check_iam_permissions()
        
        # Find existing roles
        roles = self.find_existing_sagemaker_roles()
        
        # Test SageMaker access
        sagemaker_ok = self.test_sagemaker_access()
        
        # Generate recommendations
        role_arn = self.generate_config_update(roles)
        
        print("\n SUMMARY & NEXT STEPS")
        print("=" * 60)
        
        if roles and sagemaker_ok:
            print(" GOOD NEWS: You can deploy immediately!")
            print(f"1. Update config.yaml with: {role_arn}")
            print("2. Run: python simple_sagemaker_deploy.py --config config.yaml --action deploy [options]")
        elif not iam_perms.get('iam:CreateRole', False):
            print("  LIMITED PERMISSIONS: You can't create IAM roles")
            if roles:
                print("But you have existing roles to use!")
                print(f"1. Update config.yaml with: {roles[0]['arn']}")
                print("2. Try deployment with existing role")
            else:
                print("You need to:")
                print("1. Use AWS Console to create SageMaker execution role")
                print("2. Or ask AWS administrator for help")
                print("3. Or run from local machine with admin permissions")
        else:
            print(" You have permissions to create roles!")
            print("1. Run the IAM setup script from local machine")
            print("2. Or create role manually using AWS Console")
        
        if not sagemaker_ok:
            print(" SageMaker access issue - check your permissions")
        
        return {
            'identity': identity,
            'iam_permissions': iam_perms,
            'existing_roles': roles,
            'sagemaker_access': sagemaker_ok,
            'recommended_role_arn': role_arn
        }


def main():
    """Main function"""
    try:
        checker = AWSRoleChecker()
        results = checker.run_complete_check()
        
        # Save results to file for reference
        with open('aws_check_results.json', 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            import json
            json.dump({
                'identity': results['identity'],
                'iam_permissions': results['iam_permissions'],
                'sagemaker_access': results['sagemaker_access'],
                'recommended_role_arn': results['recommended_role_arn'],
                'role_count': len(results['existing_roles'])
            }, f, indent=2, default=str)
        
        print(f"\n Results saved to: aws_check_results.json")
        
    except Exception as e:
        print(f" Checker failed: {e}")
        print("Please check your AWS credentials and try again")


if __name__ == "__main__":
    main()
