# Test Data for Chinese Produce Market Forecasting

Generated on: 2025-06-22 01:32:18

## Available Test Files

### Single Prediction Tests
- `basic_test.json` - Standard test case
- `high_volume_test.json` - High volume scenario (450+ units)
- `low_volume_test.json` - Low volume scenario (35 units)
- `high_volatility_test.json` - Price volatility scenario
- `holiday_peak_test.json` - Holiday demand peak
- `spring_fresh_test.json` - Spring season scenario
- `summer_peak_test.json` - Summer peak scenario
- `weekend_shopping_test.json` - Weekend pattern
- `random_test_*.json` - Random scenarios for robustness testing

### Batch Prediction Tests
- `batch_test.json` - Multiple instances for batch testing

## Quick Test Commands

```bash
# Test basic scenario
python fixed_sagemaker_deploy.py --config config.yaml --action test --endpoint-name YOUR_ENDPOINT --test-data data/example/basic_test.json

# Test batch predictions
python fixed_sagemaker_deploy.py --config config.yaml --action test --endpoint-name YOUR_ENDPOINT --test-data data/example/batch_test.json

# Test high volatility scenario
python fixed_sagemaker_deploy.py --config config.yaml --action test --endpoint-name YOUR_ENDPOINT --test-data data/example/high_volatility_test.json
```

## Data Format

All test files use the format expected by SageMaker endpoints:
- Single: `{"features": {"Total_Quantity": 150.0, ...}}`
- Batch: `{"instances": [{"Total_Quantity": 150.0, ...}, ...]}`

The inference script automatically expands these 19 basic features to the 90 features required by the model.
