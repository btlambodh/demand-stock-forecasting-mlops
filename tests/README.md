# MLOps Testing Guide

**Project:** Demand Stock Forecasting MLOps  
**Author:** Bhupal Lambodhar  
**Email:** btiduwarlambodhar@sandiego.edu  
**Repository:** https://github.com/btlambodh/demand-stock-forecasting-mlops

---

## Table of Contents

- [Overview](#overview)
- [Test Architecture](#test-architecture)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Test Configuration](#test-configuration)
- [Test Data & Mocks](#test-data--mocks)
- [Running Tests](#running-tests)
- [Make Targets](#make-targets)
- [Testing Libraries](#testing-libraries)
- [CI/CD Integration](#cicd-integration)
- [Coverage Analysis](#coverage-analysis)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Overview

This MLOps project implements a comprehensive testing framework covering the entire machine learning pipeline from data processing to model deployment and monitoring. The test suite ensures reliability, maintainability, and quality across all components.

### Test Statistics

| Metric | Value |
|--------|-------|
| **Total Tests** | 130 tests |
| **Test Categories** | Unit (95), Integration (25), End-to-End (10) |
| **Coverage Target** | 70%+ (enforced) |
| **Components Tested** | 8 major components |
| **Execution Time** | ~49 seconds (full suite) |

### Components Tested

- **Data Processing** - Validation, feature engineering, feature store
- **Model Training** - Training, evaluation, registry
- **Deployment** - SageMaker deployment, API endpoints
- **Monitoring** - Performance monitoring, drift detection
- **Infrastructure** - AWS services, configurations
- **Workflows** - End-to-end pipeline integration

---

## Test Architecture

### Directory Structure

```
tests/
├── unit/                          # Fast, isolated unit tests
│   ├── test_data_processing.py    # Data validation & feature engineering
│   ├── test_model_training.py     # Model training & evaluation
│   ├── test_deployment.py         # Deployment & API components
│   └── test_monitoring.py         # Performance & drift monitoring
├── integration/                   # Component interaction tests
│   ├── test_data_pipeline.py      # Data workflow integration
│   ├── test_training_pipeline.py  # Training workflow integration
│   ├── test_deployment_pipeline.py # Deployment workflow integration
│   └── test_end_to_end.py         # Complete workflow tests
├── config/                        # Test configurations
│   └── test_config.yaml          # Test-specific settings
└── data/                          # Test datasets
    ├── raw/                       # Raw test data files
    └── processed/                 # Processed test datasets
```

### Test Categories & Markers

```python
# Test Categories
@pytest.mark.unit          # Fast, isolated tests (<5 seconds)
@pytest.mark.integration   # Component interaction tests
@pytest.mark.e2e          # End-to-end workflow tests

# Component Markers
@pytest.mark.data          # Data processing tests
@pytest.mark.training      # Model training tests
@pytest.mark.deployment    # Deployment tests
@pytest.mark.monitoring    # Monitoring tests
@pytest.mark.api          # API endpoint tests

# Infrastructure Markers
@pytest.mark.aws          # AWS service tests (mocked)
@pytest.mark.sagemaker    # SageMaker-specific tests
@pytest.mark.s3           # S3 storage tests
@pytest.mark.athena       # Athena analytics tests

# Performance Markers
@pytest.mark.fast         # Tests completing <5 seconds
@pytest.mark.slow         # Tests taking >30 seconds
@pytest.mark.performance  # Performance benchmarks
```

---

## Quick Start

### One-Command Setup

```bash
# Complete test environment setup and run tests
make setup-test-env && make test-unit
```

### Development Workflow

```bash
# Setup development environment
make setup-dev

# Run fast tests during development
make test-fast

# Run with coverage
make test-coverage

# Quality checks
make quality-check
```

### CI/CD Simulation

```bash
# Complete CI/CD pipeline simulation
make ci-full
```

---

## Environment Setup

### Automated Setup (Recommended)

```bash
# Complete test environment setup
make setup-test-env
```

This command:
- Creates test directories and configuration
- Generates realistic test datasets
- Creates mock models for API testing
- Sets up proper directory structure
- Validates environment

### Manual Setup (Advanced)

```bash
# 1. Create directories
mkdir -p tests/config tests/data/{raw,processed} models

# 2. Install dependencies
pip install pytest pytest-cov pytest-mock black flake8 mypy

# 3. Generate test data
python scripts/setup_test_env.py

# 4. Verify setup
make health-check
```

### Environment Variables

```bash
# Test configuration
export TEST_CONFIG=tests/config/test_config.yaml

# AWS configuration (for integration tests)
export AWS_DEFAULT_REGION=us-east-1
export AWS_PROFILE=default
```

---

## Test Configuration

### pytest.ini Configuration

```ini
[tool:pytest]
# Test discovery
testpaths = tests
python_files = test_*.py *_test.py

# Enhanced organization markers
markers =
    unit: Unit tests for individual components
    integration: Integration tests for component interactions
    e2e: End-to-end workflow tests
    data: Data processing tests
    training: Model training tests
    deployment: Deployment tests
    monitoring: Monitoring tests
    api: API endpoint tests
    aws: AWS service tests
    slow: Tests taking >30 seconds
    fast: Tests completing <5 seconds

# Coverage and reporting
addopts = 
    --cov=src
    --cov-report=html:reports/htmlcov
    --cov-report=xml:reports/coverage.xml
    --cov-fail-under=70
    --tb=short
    --maxfail=3

# Performance settings
timeout = 300
```

### Test-Specific Configuration

```yaml
# tests/config/test_config.yaml
project:
  name: demand-stock-forecasting-mlops
  version: 1.0.0

aws:
  region: us-east-1
  s3:
    bucket_name: test-bucket
  sagemaker:
    execution_role: arn:aws:iam::123456789:role/test-role

evaluation:
  thresholds:
    mape_threshold: 25.0  # Relaxed for testing
    rmse_threshold: 10.0
    r2_threshold: 0.5

models:
  default_model: random_forest
  model_types: [linear_regression, ridge, random_forest, gradient_boosting]
```

---

## Test Data & Mocks

### Generated Test Data

The test environment automatically creates comprehensive datasets:

| Dataset | Records | Description |
|---------|---------|-------------|
| **Item Master** | 5 items | Product catalog with categories |
| **Sales Transactions** | 200+ records | Realistic sales data with seasonality |
| **Wholesale Prices** | Daily data | Price trends and variations |
| **Loss Rates** | Item-specific | Product-specific loss percentages |
| **Processed Features** | 100+ features | Engineered features for ML |

### Data Splits

```
Train: 70% (70 records)     # Training dataset
Validation: 15% (15 records) # Validation dataset  
Test: 15% (15 records)      # Test dataset
```

### Mock Implementations

#### AWS Service Mocks

```python
@patch('boto3.client')
@patch('boto3.Session')
def test_aws_functionality(mock_session, mock_client):
    """Test AWS interactions without actual AWS calls"""
    mock_client.return_value = MagicMock()
    # Test implementation without AWS costs
```

#### Model Mocks

```python
def create_mock_model():
    """Creates trained RandomForest model for testing"""
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    # Pre-fitted with consistent predictions
    return {'model': model, 'scaler': StandardScaler()}
```

#### API Testing

```python
from fastapi.testclient import TestClient

client = TestClient(app)
response = client.get("/health")
assert response.status_code == 200
```

### Test Fixtures

```python
@pytest.fixture
def config_file():
    """Test configuration file path"""
    return "tests/config/test_config.yaml"

@pytest.fixture
def sample_processed_data():
    """Sample processed data for testing"""
    return pd.read_parquet("tests/data/processed/features_complete.parquet")

@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp
```

---

## Running Tests

### Basic Test Execution

```bash
# Run all tests
make test

# Unit tests only (fast)
make test-unit

# Integration tests only
make test-integration

# Fast tests (exclude slow ones)
make test-fast

# With coverage analysis
make test-coverage
```

### Selective Testing

```bash
# Filter by markers
pytest -m "unit and not slow"              # Fast unit tests
pytest -m "api or deployment"              # API/deployment tests
pytest -m "not (slow or aws)"              # Skip slow/AWS tests

# Filter by keywords
pytest -k "test_data"                      # Tests with 'data' in name
pytest -k "not test_aws"                   # Exclude AWS tests

# Specific test files
pytest tests/unit/test_data_processing.py -v
pytest tests/integration/test_end_to_end.py -v
```

### Parallel Execution

```bash
# Speed up with parallel execution (if pytest-xdist installed)
pytest -n auto                            # Automatic parallelization
pytest -n 4                               # Use 4 workers
```

### Debug Mode

```bash
# Verbose output with debugging
pytest -v -s tests/unit/test_specific.py

# Drop into debugger on failure
pytest --pdb tests/unit/test_specific.py::test_function

# Show test discovery
pytest --collect-only tests/
```

---

## Make Targets

### Environment Setup

```bash
make setup-test-env         # Complete test environment setup
make setup-dev              # Development environment
make clean-test-env         # Clean test environment
make health-check           # System health verification
```

### Test Execution

```bash
make test                   # All tests (unit + integration)
make test-unit              # Unit tests with environment setup
make test-integration       # Integration tests only
make test-fast              # Fast tests (exclude slow)
make test-coverage          # Tests with coverage analysis
make test-with-config       # Tests with specific configuration
```

### Quality Assurance

```bash
make format                 # Code formatting (black, isort)
make lint                   # Linting (flake8, mypy)
make security-check         # Security vulnerability scan
make quality-check          # Combined quality checks
make test-full              # Complete testing pipeline
```

### CI/CD Simulation

```bash
make ci-install             # Simulate CI dependency installation
make ci-test                # Simulate CI testing phase
make ci-quality             # Simulate CI quality gates
make ci-full                # Complete CI/CD simulation
```

### Maintenance

```bash
make clean                  # Clean temporary files
make clean-data             # Clean processed data (caution)
make clean-test-env         # Clean test environment
```

---

## Testing Libraries

### Core Testing Framework

```python
pytest==7.4.0              # Primary testing framework
pytest-cov==4.1.0          # Coverage analysis
pytest-mock==3.11.1        # Enhanced mocking capabilities
moto==4.2.5                # AWS service mocking
```

### Code Quality Tools

```python
black==23.7.0              # Code formatting
flake8==6.0.0              # Code linting
isort==5.12.0              # Import sorting
mypy==1.5.1                # Static type checking
```

### Security Testing

```python
bandit==3.11.1             # Security vulnerability scanner
safety==2.3.4              # Dependency vulnerability checker
```

### Specialized Testing

```python
# API Testing
fastapi.testclient          # FastAPI endpoint testing
requests-mock               # HTTP request mocking

# Data Testing
pandas.testing              # DataFrame comparison
numpy.testing               # Array comparison

# ML Testing
scikit-learn.utils.testing  # ML model testing utilities
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: MLOps Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: make ci-install
    
    - name: Run quality checks
      run: make ci-quality
    
    - name: Run tests
      run: make ci-test
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
      with:
        file: reports/coverage.xml
```

### Generated Reports

```bash
reports/
├── htmlcov/               # HTML coverage report
├── coverage.xml           # XML coverage for CI
├── test-results.xml       # JUnit test results
└── security_report.json   # Security scan results
```

---

## Coverage Analysis

### Coverage Targets

| Component | Target Coverage |
|-----------|----------------|
| **Overall** | 70%+ (enforced) |
| **Data Processing** | 85%+ |
| **Model Training** | 80%+ |
| **Deployment** | 75%+ |
| **API** | 90%+ |
| **Monitoring** | 70%+ |

### Coverage Commands

```bash
# Generate coverage reports
make test-coverage

# View HTML report
open reports/htmlcov/index.html

# Terminal summary with missing lines
pytest --cov=src --cov-report=term-missing

# Coverage for specific component
pytest --cov=src/data_processing tests/unit/test_data_processing.py
```

### Coverage Configuration

```ini
[coverage:run]
source = src
omit = 
    */tests/*
    */test_*
    */__pycache__/*
    setup.py

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:

precision = 2
show_missing = true
```

---

## Troubleshooting

### Common Issues & Solutions

#### Import Errors
```bash
# Fix Python path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Install in development mode
pip install -e .
```

#### AWS Mocking Issues
```bash
# Ensure moto is properly installed
pip install moto[all]

# Verify mock setup in tests
@patch('boto3.client')
def test_function(mock_client):
    mock_client.return_value = MagicMock()
```

#### Test Data Issues
```bash
# Regenerate test data
make clean-test-env
make setup-test-env

# Verify test data exists
ls -la tests/data/processed/
```

### Debug Commands

```bash
# Verbose test output with print statements
pytest -v -s tests/unit/test_specific.py

# Debug specific test with breakpoint
pytest --pdb tests/unit/test_specific.py::test_function

# Show test discovery and collection
pytest --collect-only tests/

# Identify slow tests
pytest --durations=10
```

### Performance Issues

```bash
# Identify slowest tests
pytest --durations=0

# Run without coverage for speed
pytest tests/ --no-cov

# Parallel execution (if available)
pytest -n auto
```

### Environment Reset

```bash
# Complete environment reset
make clean-test-env
make setup-test-env
make health-check

# Verify configuration
cat tests/config/test_config.yaml

# Check dependencies
make health-check
```

---

## Best Practices

### Test Design Principles

- **Isolation**: Each test runs independently
- **Repeatability**: Consistent results across runs
- **Speed**: Unit tests complete in <5 seconds
- **Clarity**: Descriptive test names and documentation
- **Coverage**: Meaningful coverage over percentage targets

### Mock Strategy

- **Mock External Services**: AWS, APIs, file systems
- **Keep Logic Pure**: Mock dependencies, test business logic
- **Use Fixtures**: Reusable test data and configurations
- **Patch Correctly**: Mock at interface boundaries

### Test Data Management

- **Generate Programmatically**: Avoid large committed datasets
- **Use Realistic Data**: Resembles production patterns
- **Isolate Test Data**: Each test has independent data
- **Clean Up**: Remove artifacts after execution

### Continuous Improvement

- **Monitor Coverage**: Track and improve coverage over time
- **Optimize Performance**: Keep test suite fast and responsive
- **Regular Updates**: Keep dependencies and practices current
- **Documentation**: Maintain clear testing documentation

---

## Quick Reference

### Essential Commands

```bash
# Setup
make setup-test-env         # Complete setup

# Testing
make test-fast              # Quick development tests
make test-coverage          # Full tests with coverage
make test-integration       # Integration tests

# Quality
make quality-check          # Code quality checks
make ci-full               # Complete CI simulation

# Maintenance
make clean-test-env         # Reset environment
make health-check          # Verify setup
```

### Test Markers

```bash
# By category
pytest -m unit             # Unit tests
pytest -m integration      # Integration tests
pytest -m "not slow"       # Exclude slow tests

# By component
pytest -m data             # Data processing
pytest -m training         # Model training
pytest -m api              # API tests
```

### Coverage

```bash
# Generate reports
make test-coverage

# View results
open reports/htmlcov/index.html
```

---

**Support**: For questions or issues, contact [btiduwarlambodhar@sandiego.edu](mailto:btiduwarlambodhar@sandiego.edu)

**Repository**: [github.com/btlambodh/demand-stock-forecasting-mlops](https://github.com/btlambodh/demand-stock-forecasting-mlops)