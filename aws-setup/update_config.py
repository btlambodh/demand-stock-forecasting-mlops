#!/usr/bin/env python3
"""
Update config.yaml to use existing SageMaker execution role
"""

import yaml
import boto3

def get_current_sagemaker_role():
    """Get the current SageMaker execution role ARN"""
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        
        # Extract role ARN from assumed role ARN
        assumed_role_arn = identity['Arn']
        print(f"Current identity: {assumed_role_arn}")
        
        # Parse the role name from assumed-role ARN
        # Format: arn:aws:sts::ACCOUNT:assumed-role/ROLE_NAME/SESSION_NAME
        if 'assumed-role' in assumed_role_arn:
            parts = assumed_role_arn.split('/')
            role_name = parts[1]  # Get role name
            account_id = identity['Account']
            
            # Construct the role ARN
            role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
            
            print(f"Extracted role name: {role_name}")
            print(f"Role ARN: {role_arn}")
            
            return role_arn, role_name
        else:
            print("Not running under an assumed role")
            return None, None
            
    except Exception as e:
        print(f"Error getting current role: {e}")
        return None, None

def check_role_permissions(role_name):
    """Check if the role has necessary permissions"""
    try:
        iam = boto3.client('iam')
        
        print(f"\n🔍 Checking permissions for role: {role_name}")
        
        # Get attached policies
        attached_policies = iam.list_attached_role_policies(RoleName=role_name)
        
        print("Attached managed policies:")
        has_sagemaker_access = False
        for policy in attached_policies['AttachedPolicies']:
            policy_name = policy['PolicyArn'].split('/')[-1]
            print(f"  ✅ {policy_name}")
            if 'SageMaker' in policy_name:
                has_sagemaker_access = True
        
        # Check inline policies
        inline_policies = iam.list_role_policies(RoleName=role_name)
        if inline_policies['PolicyNames']:
            print("Inline policies:")
            for policy_name in inline_policies['PolicyNames']:
                print(f"  ✅ {policy_name}")
        
        return has_sagemaker_access
        
    except Exception as e:
        print(f"Error checking role permissions: {e}")
        return False

def update_config_file():
    """Update config.yaml with the current SageMaker role"""
    try:
        # Get current role
        role_arn, role_name = get_current_sagemaker_role()
        
        if not role_arn:
            print("❌ Could not determine current SageMaker role")
            return False
        
        # Check permissions
        has_permissions = check_role_permissions(role_name)
        
        # Load current config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Update the role ARN
        old_role = config['aws']['sagemaker']['execution_role']
        config['aws']['sagemaker']['execution_role'] = role_arn
        
        # Save updated config
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"\n✅ Updated config.yaml:")
        print(f"  Old role: {old_role}")
        print(f"  New role: {role_arn}")
        
        if has_permissions:
            print("✅ Role appears to have SageMaker permissions")
        else:
            print("⚠️  Role may need additional SageMaker permissions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False

def main():
    print("🔧 Updating config.yaml to use existing SageMaker role...")
    print("=" * 60)
    
    success = update_config_file()
    
    if success:
        print("\n🎉 Config updated successfully!")
        print("\nNext steps:")
        print("1. Run diagnostics: python aws_diagnostics.py")
        print("2. Try deployment: python src/deployment/sagemaker_deploy.py ...")
    else:
        print("\n❌ Failed to update config")

if __name__ == "__main__":
    main()