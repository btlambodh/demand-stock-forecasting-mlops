# Chinese Produce Market RMB Forecasting - MLOps Project

A comprehensive MLOps solution for forecasting produce prices in the Chinese market using AWS SageMaker, built with CI/CD automation and production-grade deployment.

## Project Overview

This project implements an end-to-end machine learning pipeline for forecasting produce prices in the Chinese market using historical sales data, wholesale prices, and market indicators. The solution leverages AWS SageMaker for model training and deployment with full MLOps automation.

## Business Problem

Agricultural price forecasting in China involves multiple complex factors:
- **Seasonal patterns** with predictable yearly recurrence
- **Exchange rate fluctuations** (RMB/USD) impacting market volatility
- **Supply chain disruptions** and loss rates affecting availability
- **Import/export volumes** influencing domestic pricing
- **Regional demand variations** across different produce categories

## Architecture

```
├── Data Ingestion (S3) → Feature Engineering → Model Training (SageMaker)
├── Model Evaluation → Model Registry → Automated Deployment
├── Real-time Inference → Batch Predictions → Monitoring & Alerts
└── CI/CD Pipeline (GitHub Actions) → Infrastructure as Code (CloudFormation)
```

## Data Structure

Our dataset consists of four key annexes:

- **annex1.csv** (241 rows): Item master data with categories
- **annex2.csv** (497,990 rows): Sales transactions with timestamps
- **annex3.csv** (55,956 rows): Historical wholesale prices
- **annex4.csv** (241 rows): Product loss rates by item

## Quick Start

### Prerequisites
- AWS Account with SageMaker access
- Python 3.8+
- Git

### Setup Instructions

1. **Clone Repository**
```bash
git clone https://github.com/btlambodh/demand-stock-forecasting-mlops.git
cd demand-stock-forecasting-mlops
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure AWS Credentials**
```bash
aws configure
# Region: us-east-1
# Account: 346761359662
```

4. **Upload Data to S3**
```bash
aws s3 sync data/ s3://chinese-produce-forecast-346761359662/data/
```

5. **Run End-to-End Pipeline**
```bash
jupyter notebook run_end_to_end_pipeline.ipynb
```

## 🛠️ Project Structure

```
├── data/
│   ├── raw/                    # Raw CSV files
│   ├── processed/              # Feature-engineered datasets
│   └── predictions/            # Model outputs
├── src/
│   ├── data_processing/        # ETL and feature engineering
│   ├── models/                 # ML model definitions
│   ├── training/               # Training scripts
│   ├── inference/              # Prediction scripts
│   └── utils/                  # Helper utilities
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── feature_engineering.ipynb
│   └── run_end_to_end_pipeline.ipynb
├── infrastructure/
│   ├── cloudformation/         # AWS infrastructure templates
│   └── docker/                 # Container definitions
├── tests/                      # Unit and integration tests
├── .github/workflows/          # CI/CD automation
├── config.yaml                 # Project configuration
├── requirements.txt            # Python dependencies
└── organized-notebook.md       # Project documentation
```

## Configuration

Key configuration parameters in `config.yaml`:
- **AWS Region**: us-east-1
- **S3 Bucket**: chinese-produce-forecast-346761359662
- **SageMaker Role ARN**: arn:aws:iam::346761359662:role/SageMakerExecutionRole
- **Model Parameters**: Network architecture, hyperparameters
- **Data Processing**: Feature engineering settings

## Machine Learning Pipeline

### 1. Data Processing
- **Temporal Feature Engineering**: Seasonal patterns, trend decomposition
- **Price Relationship Modeling**: Wholesale vs retail price dynamics
- **Loss Rate Integration**: Supply chain efficiency metrics
- **Exchange Rate Features**: RMB/USD impact analysis

### 2. Model Architecture
- **Neural Networks**: RBF, BP, NARX for time series forecasting
- **Ensemble Methods**: Multiple model combination for robustness
- **Feature Importance**: Automated feature selection
- **Hyperparameter Tuning**: Automated optimization via SageMaker

### 3. Deployment Strategy
- **Real-time Endpoints**: Low-latency price predictions
- **Batch Transform**: Bulk forecasting for inventory planning
- **A/B Testing**: Model performance comparison
- **Auto Scaling**: Dynamic resource allocation

## Key Features

-  **Automated ETL Pipeline** with error handling and validation
-  **Feature Engineering** for seasonal and exchange rate patterns
-  **Multi-model Training** with hyperparameter optimization
-  **Model Registry** with versioning and lineage tracking
-  **CI/CD Integration** with automated testing and deployment
-  **Monitoring & Alerting** for model drift and performance
-  **Infrastructure as Code** for reproducible deployments

## Expected Outcomes

- **Price Forecast Accuracy**: MAPE < 15% for 30-day predictions
- **Real-time Performance**: < 100ms inference latency
- **Cost Optimization**: 40% reduction in compute costs via auto-scaling
- **Operational Efficiency**: 80% reduction in manual intervention

## Team

- **Author**: Bhupal Lambodhar
- **Email**: btiduwarlambodhar@sandiego.edu
- **Repository**: https://github.com/btlambodh/demand-stock-forecasting-mlops.git

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Support

For questions and support, please contact btiduwarlambodhar@sandiego.edu or open an issue in the GitHub repository.