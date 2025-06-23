# Integration Tests for MLOps Pipeline

This directory contains integration tests that verify the interactions between different components of the Demand Stock Forecasting MLOps pipeline.

## Test Structure

```
tests/integration/
├── __init__.py                    # Package initialization
├── README.md                     # This file
├── test_data_pipeline.py         # Data processing integration tests
├── test_training_pipeline.py     # Model training integration tests
├── test_deployment_pipeline.py   # Deployment integration tests
└── test_end_to_end.py            # End-to-end workflow tests
```

## Test Categories

### 1. Data Pipeline Integration (`test_data_pipeline.py`)
Tests the flow and integration between:
- Data validation → Feature engineering
- Feature engineering → Feature store integration
- Data quality checks across components
- Error handling in data pipeline

**Key Tests:**
- `test_validation_to_feature_engineering_flow()` - Validates data flows correctly
- `test_feature_engineering_to_feature_store_flow()` - Tests Feature Store integration
- `test_end_to_end_data_pipeline()` - Complete data pipeline
- `test_data_pipeline_error_handling()` - Error scenarios

### 2. Training Pipeline Integration (`test_training_pipeline.py`)
Tests the flow and integration between:
- Model training → Evaluation
- Training → Model registry
- Model loading → Prediction
- Performance validation

**Key Tests:**
- `test_training_to_evaluation_flow()` - Training produces valid evaluation
- `test_training_to_model_loading_flow()` - Models can be loaded and used
- `test_training_to_registry_integration()` - Model registry integration
- `test_model_predictor_integration()` - Predictor component integration

### 3. Deployment Pipeline Integration (`test_deployment_pipeline.py`)
Tests the flow and integration between:
- Model deployment → SageMaker endpoints
- API initialization → Model loading
- Prediction API → Inference
- Configuration → Deployment

**Key Tests:**
- `test_sagemaker_deployer_initialization()` - SageMaker deployer setup
- `test_api_model_loading_integration()` - API loads models correctly
- `test_api_prediction_integration()` - API prediction functionality
- `test_deployment_configuration_validation()` - Config validation

### 4. End-to-End Integration (`test_end_to_end.py`)
Tests complete MLOps workflows:
- Raw data → Trained model → Deployed API
- Full pipeline with monitoring
- Error recovery scenarios
- Performance and scalability

**Key Tests:**
- `test_complete_mlops_workflow()` - Full pipeline execution
- `test_model_to_api_pipeline()` - Model deployment to API
- `test_monitoring_integration_pipeline()` - Monitoring integration
- `test_error_recovery_workflow()` - Error handling

## Running the Tests

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov

# Ensure you're in the project root directory
cd /path/to/demand-stock-forecasting-mlops
```

### Basic Test Execution

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific test file
pytest tests/integration/test_data_pipeline.py -v

# Run specific test method
pytest tests/integration/test_data_pipeline.py::TestDataPipelineIntegration::test_validation_to_feature_engineering_flow -v
```

### Advanced Test Options

```bash
# Run with coverage report
pytest tests/integration/ --cov=src --cov-report=html --cov-report=term-missing

# Run only fast tests (exclude slow end-to-end tests)
pytest tests/integration/ -m "not slow" -v

# Run only data-related tests
pytest tests/integration/ -m "data" -v

# Run with detailed output and timing
pytest tests/integration/ -v --durations=10 --tb=long

# Run in parallel (if pytest-xdist installed)
pytest tests/integration/ -n auto

# Run with timeout (useful for CI)
pytest tests/integration/ --timeout=300
```

### Using Make Commands

```bash
# From project root, use the Makefile
make test-integration          # Run integration tests
make test-full                # Run all tests with coverage
make ci-test                  # Simulate CI testing
```

## Test Configuration

The tests use temporary directories and mock AWS services to avoid:
- Requiring actual AWS credentials during testing
- Creating real AWS resources
- Interfering with production systems

### Key Configuration Features:
- **Temporary workspaces**: All file I/O uses temporary directories
- **Mocked AWS services**: Boto3 clients are mocked to prevent real AWS calls
- **Realistic test data**: Synthetic data mimics real market data patterns
- **Configurable timeouts**: Tests have reasonable time limits
- **Error simulation**: Tests include error scenarios and edge cases

## Test Markers

Tests are marked for easy filtering:

```bash
# Available markers
pytest --markers

# Filter by marker
pytest tests/integration/ -m "data"        # Data processing tests
pytest tests/integration/ -m "training"    # Training pipeline tests  
pytest tests/integration/ -m "deployment"  # Deployment tests
pytest tests/integration/ -m "e2e"         # End-to-end tests
pytest tests/integration/ -m "aws"         # AWS integration tests
pytest tests/integration/ -m "slow"        # Long-running tests
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure src is in Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **Missing Dependencies**
   ```bash
   # Install all requirements
   pip install -r requirements.txt
   ```

3. **Temporary Directory Issues**
   ```bash
   # Clean up any leftover temp files
   find /tmp -name "*demand*" -type d -exec rm -rf {} +
   ```

4. **Test Timeouts**
   ```bash
   # Increase timeout for slow systems
   pytest tests/integration/ --timeout=600
   ```

### Debug Mode

```bash
# Run with maximum verbosity and debugging
pytest tests/integration/ -vvs --tb=long --capture=no

# Run single test with debugging
pytest tests/integration/test_data_pipeline.py::TestDataPipelineIntegration::test_validation_to_feature_engineering_flow -vvs --capture=no
```

## Continuous Integration

These tests are designed to run in CI environments:

```yaml
# Example GitHub Actions configuration
- name: Run Integration Tests
  run: |
    pytest tests/integration/ \
      --cov=src \
      --cov-report=xml \
      --timeout=300 \
      -m "not slow"
```

## Contributing

When adding new integration tests:

1. **Follow naming conventions**: `test_*.py` files, `Test*` classes, `test_*` methods
2. **Use appropriate markers**: Mark tests with relevant categories
3. **Mock external services**: Use `@patch` for AWS and external API calls
4. **Clean up resources**: Use fixtures for temporary directories
5. **Test error scenarios**: Include negative test cases
6. **Document test purpose**: Clear docstrings explaining what's being tested

## Performance Guidelines

- Integration tests should complete in under 5 minutes total
- Individual tests should finish within 30 seconds
- Use `@pytest.mark.slow` for tests taking longer than 30 seconds
- Mock expensive operations (AWS calls, large data processing)
- Use small, realistic datasets for testing

## Support

For questions or issues with integration tests:
- Check the main project documentation
- Review test logs and error messages
- Use debug mode for detailed output
- Contact: btiduwarlambodhar@sandiego.edu