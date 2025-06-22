#!/usr/bin/env python3
"""
FIXED Monitoring System Test Script
Tests both performance monitoring and drift detection

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import os
import sys
import time
import json
import logging
import tempfile
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """Create test data for monitoring"""
    logger.info("🔧 Creating test data...")
    
    # Create directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/monitoring/metrics", exist_ok=True)
    os.makedirs("data/monitoring/reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Generate synthetic reference data (training data)
    np.random.seed(42)
    n_samples = 1000
    
    reference_data = pd.DataFrame({
        'Total_Quantity': np.random.normal(150, 30, n_samples),
        'Avg_Price': np.random.normal(18, 3, n_samples),
        'Transaction_Count': np.random.poisson(25, n_samples),
        'Price_Volatility': np.random.gamma(2, 0.5, n_samples),
        'Revenue': np.random.normal(2700, 500, n_samples),
        'Wholesale_Price': np.random.normal(14, 2, n_samples),
        'Loss_Rate': np.random.beta(2, 20, n_samples) * 20,  # 0-20% loss rate
        'Month': np.random.randint(1, 13, n_samples),
        'DayOfWeek': np.random.randint(0, 7, n_samples),
        'IsWeekend': np.random.binomial(1, 0.2, n_samples),
        'Category_Code': np.random.randint(1, 5, n_samples)
    })
    
    # Add some correlation
    reference_data['Revenue'] = reference_data['Total_Quantity'] * reference_data['Avg_Price'] * np.random.normal(1, 0.1, n_samples)
    reference_data['Wholesale_Price'] = reference_data['Avg_Price'] * 0.75 * np.random.normal(1, 0.05, n_samples)
    
    # Save reference data
    reference_data.to_parquet("data/processed/train.parquet")
    logger.info(f"✅ Reference data created: {reference_data.shape}")
    
    # Generate current data with some drift
    current_data = reference_data.copy()
    
    # Introduce drift in some features
    current_data['Avg_Price'] *= np.random.normal(1.1, 0.1, n_samples)  # Price inflation
    current_data['Price_Volatility'] *= np.random.normal(1.3, 0.15, n_samples)  # Higher volatility
    current_data['Loss_Rate'] += np.random.normal(2, 1, n_samples)  # Higher loss rates
    current_data['Loss_Rate'] = np.clip(current_data['Loss_Rate'], 0, 25)
    
    # Add some noise to break exact correlations
    current_data += np.random.normal(0, 0.1, current_data.shape)
    
    # Save current data
    current_data.to_parquet("data/processed/validation.parquet")
    logger.info(f"✅ Current data created with drift: {current_data.shape}")
    
    # Create model evaluation data
    evaluation_data = {
        'best_model': {
            'val_mape': 12.5,
            'val_rmse': 2.8,
            'val_r2': 0.85,
            'test_mape': 13.2,
            'test_rmse': 3.1,
            'test_r2': 0.82
        },
        'linear_regression': {
            'val_mape': 15.8,
            'val_rmse': 3.2,
            'val_r2': 0.78,
            'test_mape': 16.1,
            'test_rmse': 3.4,
            'test_r2': 0.76
        },
        'ridge': {
            'val_mape': 14.2,
            'val_rmse': 3.0,
            'val_r2': 0.81,
            'test_mape': 14.8,
            'test_rmse': 3.2,
            'test_r2': 0.79
        }
    }
    
    with open("models/evaluation.json", 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    logger.info("✅ Model evaluation data created")
    
    return reference_data, current_data

def create_config_file():
    """Create configuration file"""
    logger.info("⚙️  Creating configuration file...")
    
    config = {
        'project': {
            'name': 'chinese-produce-forecasting',
            'version': '1.2.0'
        },
        'monitoring': {
            'performance': {
                'drift_threshold': 0.25,
                'performance_degradation_threshold': 0.15,
                'cpu_threshold': 80,
                'memory_threshold': 85,
                'disk_threshold': 90
            },
            'alerts': {
                'enabled': True,
                'cooldown_minutes': 30,
                'local_mode': True
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
    
    import yaml
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("✅ Configuration file created")

def test_performance_monitor():
    """Test performance monitoring"""
    logger.info("🧪 Testing performance monitor...")
    
    try:
        # Import the fixed performance monitor
        sys.path.append('src/monitoring')
        from performance_monitor import PerformanceMonitor
        
        # Initialize in local mode
        monitor = PerformanceMonitor('config.yaml', local_mode=True)
        
        # Test health summary
        health = monitor.get_health_summary()
        
        print(f"\n📊 Performance Monitor Test Results:")
        print(f"✅ Monitor initialized successfully")
        print(f"✅ Health summary generated: {health['overall_status']}")
        print(f"✅ Local mode: {health['local_mode']}")
        print(f"✅ AWS enabled: {health['aws_enabled']}")
        
        # Test metrics collection
        monitor.collect_system_metrics()
        monitor.collect_model_metrics()
        monitor.collect_api_metrics()
        monitor.collect_data_quality_metrics()
        
        print(f"✅ All metrics collection methods working")
        
        # Test alert processing
        test_alert = {
            'type': 'test',
            'severity': 'info',
            'message': 'Test alert from monitoring test',
            'timestamp': datetime.now().isoformat()
        }
        monitor._process_alert(test_alert)
        print(f"✅ Alert processing working")
        
        # Test export
        export_file = monitor.export_metrics()
        print(f"✅ Metrics export working: {export_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance monitor test failed: {e}")
        return False

def test_drift_detector():
    """Test drift detection"""
    logger.info("🧪 Testing drift detector...")
    
    try:
        # Import the fixed drift detector
        sys.path.append('src/monitoring')
        from drift_detector import DriftDetector
        
        # Initialize in local mode
        detector = DriftDetector('config.yaml', local_mode=True)
        
        # Load test data
        reference_loaded = detector.load_reference_data("data/processed/train.parquet")
        if not reference_loaded:
            logger.error("❌ Could not load reference data")
            return False
        
        current_data = pd.read_parquet("data/processed/validation.parquet")
        
        print(f"\n🔍 Drift Detector Test Results:")
        print(f"✅ Detector initialized successfully")
        print(f"✅ Reference data loaded: {detector.reference_data.shape}")
        print(f"✅ Current data loaded: {current_data.shape}")
        
        # Test statistical drift detection
        results = detector.detect_data_drift(current_data, method='statistical')
        
        print(f"✅ Statistical drift detection: {results['overall_drift_detected']}")
        print(f"✅ Drift score: {results['drift_score']:.4f}")
        print(f"✅ Features analyzed: {len(results['feature_drift'])}")
        
        # Test KS test
        ks_results = detector.detect_data_drift(current_data, method='ks_test')
        print(f"✅ KS test drift detection: {ks_results['overall_drift_detected']}")
        
        # Test PSI
        psi_results = detector.detect_data_drift(current_data, method='population_stability')
        print(f"✅ PSI drift detection: {psi_results['overall_drift_detected']}")
        
        # Test report generation
        report_path = detector.generate_drift_report(results)
        print(f"✅ Report generated: {report_path}")
        
        # Test state save/load
        detector.save_drift_state()
        state_loaded = detector.load_drift_state()
        print(f"✅ State save/load: {state_loaded}")
        
        # Test export
        export_file = detector.export_drift_state()
        print(f"✅ State export: {export_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Drift detector test failed: {e}")
        return False

def test_api_integration():
    """Test integration with API"""
    logger.info("🧪 Testing API integration...")
    
    try:
        import requests
        
        # Test API health
        try:
            response = requests.get('http://localhost:8000/health', timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"\n🌐 API Integration Test Results:")
                print(f"✅ API is responding")
                print(f"✅ Status: {health_data.get('status')}")
                print(f"✅ Models loaded: {health_data.get('models_loaded')}")
                api_available = True
            else:
                print(f"\n🌐 API Integration Test Results:")
                print(f"⚠️  API responding but with status: {response.status_code}")
                api_available = False
        except:
            print(f"\n🌐 API Integration Test Results:")
            print(f"⚠️  API not available (this is OK if not running)")
            api_available = False
        
        # Test metrics endpoint
        if api_available:
            try:
                metrics_response = requests.get('http://localhost:8000/metrics', timeout=5)
                if metrics_response.status_code == 200:
                    print(f"✅ Metrics endpoint accessible")
                else:
                    print(f"⚠️  Metrics endpoint status: {metrics_response.status_code}")
            except:
                print(f"⚠️  Metrics endpoint not accessible")
        
        return api_available
        
    except Exception as e:
        logger.error(f"❌ API integration test failed: {e}")
        return False

def check_dependencies():
    """Check required dependencies"""
    logger.info("🔍 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scipy', 'sklearn', 'pydantic',
        'fastapi', 'uvicorn', 'requests', 'yaml', 'psutil'
    ]
    
    optional_packages = [
        'boto3', 'plotly', 'dash', 'evidently', 'matplotlib', 'seaborn'
    ]
    
    missing_required = []
    missing_optional = []
    
    print(f"\n📦 Dependency Check:")
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (REQUIRED)")
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package} (optional)")
        except ImportError:
            print(f"⚠️  {package} (optional - some features disabled)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print(f"Install with: pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print(f"Install with: pip install {' '.join(missing_optional)}")
    
    print(f"\n✅ All required dependencies available")
    return True

def main():
    """Run complete monitoring system test"""
    print("🚀 FIXED Monitoring System Test Suite")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        print("❌ Please install missing required dependencies first")
        sys.exit(1)
    
    # Create test data and config
    print(f"\n📁 Setting up test environment...")
    create_config_file()
    reference_data, current_data = create_test_data()
    
    # Run tests
    tests = [
        ("Performance Monitor", test_performance_monitor),
        ("Drift Detector", test_drift_detector),
        ("API Integration", test_api_integration)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
    
    # Final results
    print(f"\n" + "="*60)
    print(f"🏁 Test Results Summary")
    print(f"="*60)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print(f"🎉 ALL TESTS PASSED! Monitoring system is ready!")
        print(f"\n📝 Next Steps:")
        print(f"1. Start API: python -c \"import sys; sys.path.append('src/inference'); from api import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)\"")
        print(f"2. Start monitoring: python src/monitoring/performance_monitor_fixed.py --config config.yaml --action start --local-mode")
        print(f"3. Run drift detection: python src/monitoring/drift_detector.py --config config.yaml --action detect --current-data data/processed/validation.parquet --local-mode")
        print(f"4. Open dashboard: python src/monitoring/performance_monitor_fixed.py --config config.yaml --action dashboard --local-mode")
    elif passed_tests >= total_tests * 0.7:
        print(f"⚠️  Most tests passed - system mostly functional")
        print(f"Review failed tests above for issues")
    else:
        print(f"❌ Multiple test failures - please review setup")
    
    # Show created files
    print(f"\n📁 Files Created:")
    created_files = [
        "config.yaml",
        "data/processed/train.parquet",
        "data/processed/validation.parquet",
        "models/evaluation.json"
    ]
    
    for file_path in created_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} (missing)")
    
    print(f"\n📊 You can now run the monitoring commands!")

if __name__ == "__main__":
    main()