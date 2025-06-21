#!/usr/bin/env python3
"""
Setup script to create the complete MLOps project structure
Run this script to set up all directories and files for the Chinese Produce Forecasting project

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import os
import json

def create_directory_structure():
    """Create all necessary directories"""
    directories = [
        'src',
        'src/data_processing',
        'src/training',
        'src/models',
        'src/deployment',
        'src/inference',
        'src/monitoring',
        'src/utils',
        'src/evaluation',
        'notebooks',
        'data',
        'data/raw',
        'data/processed',
        'data/validation',
        'data/predictions',
        'models',
        'tests',
        'tests/unit',
        'tests/integration',
        'tests/deployment',
        'infrastructure',
        'infrastructure/cloudformation',
        'infrastructure/docker',
        '.github',
        '.github/workflows',
        'reports',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_init_files():
    """Create __init__.py files for Python packages"""
    init_locations = [
        'src/__init__.py',
        'src/data_processing/__init__.py',
        'src/training/__init__.py',
        'src/models/__init__.py',
        'src/deployment/__init__.py',
        'src/inference/__init__.py',
        'src/monitoring/__init__.py',
        'src/utils/__init__.py',
        'src/evaluation/__init__.py',
        'tests/__init__.py',
        'tests/unit/__init__.py',
        'tests/integration/__init__.py',
        'tests/deployment/__init__.py'
    ]
    
    for init_file in init_locations:
        with open(init_file, 'w') as f:
            f.write('"""Package initialization file"""\n')
        print(f"✅ Created: {init_file}")

def create_placeholder_data_files():
    """Create placeholder files to show expected data structure"""
    
    # Create a data structure info file
    data_info = {
        "expected_files": {
            "annex1.csv": {
                "description": "Item master data",
                "columns": ["Item Code", "Item Name", "Category Code", "Category Name"],
                "expected_rows": 241
            },
            "annex2.csv": {
                "description": "Sales transactions",
                "columns": ["Date", "Time", "Item Code", "Quantity Sold (kilo)", 
                           "Unit Selling Price (RMB/kg)", "Sale or Return", "Discount (Yes/No)"],
                "expected_rows": 497990
            },
            "annex3.csv": {
                "description": "Wholesale prices",
                "columns": ["Date", "Item Code", "Wholesale Price (RMB/kg)"],
                "expected_rows": 55956
            },
            "annex4.csv": {
                "description": "Loss rates",
                "columns": ["Item Code", "Item Name", "Loss Rate (%)"],
                "expected_rows": 241
            }
        },
        "instructions": "Place your actual CSV files in this directory (data/raw/) with the exact names listed above."
    }
    
    with open('data/raw/README.json', 'w') as f:
        json.dump(data_info, f, indent=2)
    
    print("✅ Created data structure info in data/raw/README.json")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# Environment variables
.env
.env.local
.env.production

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
models/*.pkl
models/*.h5
models/*.joblib
data/processed/*.parquet
data/validation/*.html
logs/*.log
mlflow.db
mlruns/

# AWS
.aws/

# Data files (uncomment if you don't want to track data)
# data/raw/*.csv
# data/processed/*
# data/predictions/*

# Model artifacts
*.model
*.weights

# Temporary files
*.tmp
*.temp
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✅ Created .gitignore")

def main():
    """Main setup function"""
    print("🚀 Setting up Chinese Produce Forecasting MLOps Project")
    print("=" * 60)
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    create_directory_structure()
    
    # Create __init__.py files
    print("\n🐍 Creating Python package files...")
    create_init_files()
    
    # Create data info
    print("\n📊 Creating data structure info...")
    create_placeholder_data_files()
    
    # Create .gitignore
    print("\n📝 Creating .gitignore...")
    create_gitignore()
    
    print("\n" + "=" * 60)
    print("✅ PROJECT SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Copy your CSV files to data/raw/ directory:")
    print("   - annex1.csv")
    print("   - annex2.csv") 
    print("   - annex3.csv")
    print("   - annex4.csv")
    print("\n2. Save the provided Python scripts to their respective directories")
    print("3. Run the validation script to test everything")
    print("\n📂 Project structure is ready for MLOps pipeline!")

if __name__ == "__main__":
    main()