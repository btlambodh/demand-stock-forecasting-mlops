# =============================================================================
# Demand Stock Forecasting MLOps Project - Organized Makefile
# =============================================================================
# Author: Bhupal Lambodhar
# Email: btiduwarlambodhar@sandiego.edu
# Repository: https://github.com/btlambodh/demand-stock-forecasting-mlops

.PHONY: help install test lint format clean build deploy docs setup-dev
.DEFAULT_GOAL := help

# =============================================================================
# Project Variables
# =============================================================================
PYTHON := python3
PIP := pip3
PROJECT_NAME := demand-stock-forecasting-mlops
VENV_NAME := dsfenv
TEST_PATH := tests/
SRC_PATH := src/
REPORTS_PATH := reports/
DATA_PATH := data/

# Configuration variables
CONFIG_FILE := config.yaml
TEST_CONFIG_FILE := tests/config/test_config.yaml

# Model variables
MODEL_VERSION = v$(shell date +%Y%m%d_%H%M%S)
FORECAST_HORIZON = 1
PROCESSED_DATA_PATH = data/processed/
MODEL_OUTPUT_PATH = models/
MODEL_NAME = chinese_produce_forecaster
MODEL_PATH = models/best_model.pkl

# Stage variables
SOURCE_STAGE = dev
TARGET_STAGE = staging

# Endpoint variables
ENDPOINT_BASE_NAME = produce-forecast
ENDPOINT_NAME = $(ENDPOINT_BASE_NAME)-dev

# API variables
API_PORT = 8000
API_HOST = 0.0.0.0

# Monitoring variables
MONITORING_INTERVAL = 60
DASHBOARD_PORT = 8050
REFERENCE_DATA = data/processed/train.parquet
CURRENT_DATA = data/processed/validation.parquet

# =============================================================================
# Terminal Colors
# =============================================================================
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
RESET := \033[0m

# =============================================================================
# 1. PROJECT INFORMATION & HELP
# =============================================================================

help: ## Show this help message
	@echo "$(CYAN)Demand Stock Forecasting MLOps Project$(RESET)"
	@echo "$(YELLOW)Author: Bhupal Lambodhar$(RESET)"
	@echo "$(YELLOW)Email: btiduwarlambodhar@sandiego.edu$(RESET)"
	@echo "$(YELLOW)Repository: https://github.com/btlambodh/demand-stock-forecasting-mlops$(RESET)"
	@echo ""
	@echo "$(GREEN)Available targets:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(CYAN)%-25s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Quick Start Workflows:$(RESET)"
	@echo "  $(CYAN)make pipeline-full$(RESET)           Complete MLOps pipeline"
	@echo "  $(CYAN)make workflow-dev$(RESET)            Development workflow"
	@echo "  $(CYAN)make workflow-staging$(RESET)        Staging deployment"
	@echo "  $(CYAN)make workflow-prod$(RESET)           Production deployment"

info: ## Show detailed project information
	@echo "$(CYAN)Demand Stock Forecasting MLOps Project$(RESET)"
	@echo "$(YELLOW)Author:$(RESET) Bhupal Lambodhar"
	@echo "$(YELLOW)Email:$(RESET) btiduwarlambodhar@sandiego.edu"
	@echo "$(YELLOW)Repository:$(RESET) https://github.com/btlambodh/demand-stock-forecasting-mlops"
	@echo "$(YELLOW)AWS Account:$(RESET) 346761359662"
	@echo "$(YELLOW)Region:$(RESET) us-east-1"
	@echo "$(YELLOW)S3 Bucket:$(RESET) sagemaker-us-east-1-346761359662"
	@echo "$(YELLOW)Project Name:$(RESET) $(PROJECT_NAME)"
	@echo ""
	@echo "$(GREEN)Current Milestone:$(RESET) Testing & CI/CD Implementation"
	@echo "$(GREEN)Next Steps:$(RESET) Production Deployment & Advanced Features"

status: ## Show current project status
	@echo "$(CYAN)Project Status:$(RESET)"
	@echo "$(YELLOW)Python Version:$(RESET) $(shell $(PYTHON) --version)"
	@echo "$(YELLOW)Virtual Environment:$(RESET) $(CONDA_DEFAULT_ENV)"
	@echo "$(YELLOW)Git Branch:$(RESET) $(shell git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "$(YELLOW)Git Status:$(RESET)"
	@git status --porcelain 2>/dev/null || echo "  Not a git repository"
	@echo "$(YELLOW)Data Files:$(RESET)"
	@ls -la $(DATA_PATH)/ 2>/dev/null || echo "  No data directory"
	@echo "$(YELLOW)Model Files:$(RESET)"
	@ls -la models/ 2>/dev/null || echo "  No models directory"

help-workflows: ## Show workflow examples
	@echo "$(CYAN)MLOps Workflow Examples:$(RESET)"
	@echo ""
	@echo "$(GREEN)Development Workflow:$(RESET)"
	@echo "  make workflow-dev                     # Complete dev workflow"
	@echo "  make pipeline-data-full               # Data pipeline with feature store"
	@echo "  make train-models                     # Train and validate models"
	@echo "  make register-models                  # Register to model registry"
	@echo ""
	@echo "$(GREEN)Testing & Quality:$(RESET)"
	@echo "  make test-full                        # Complete test suite"
	@echo "  make quality-check                    # Code quality checks"
	@echo "  make security-check                   # Security scan"
	@echo ""
	@echo "$(GREEN)Deployment Workflows:$(RESET)"
	@echo "  make workflow-staging                 # Deploy to staging"
	@echo "  make workflow-prod                    # Deploy to production"
	@echo "  make api-local                        # Start local API"
	@echo ""
	@echo "$(GREEN)Monitoring & Operations:$(RESET)"
	@echo "  make monitoring-start                 # Start all monitoring"
	@echo "  make monitoring-status                # Check system health"
	@echo "  make pipeline-bi                      # Business intelligence"

# =============================================================================
# 2. ENVIRONMENT SETUP
# =============================================================================

install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(RESET)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Dependencies installed successfully!$(RESET)"

setup-dev: ## Setup development environment
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 mypy bandit safety
	pre-commit install || echo "$(YELLOW)pre-commit not available, skipping...$(RESET)"
	@echo "$(GREEN)Development environment setup complete!$(RESET)"

setup-test-env: ## Setup test environment with test data and configuration
	@echo "$(BLUE)Setting up test environment...$(RESET)"
	@mkdir -p tests/config tests/data/processed tests/data/raw models
	@if [ ! -f $(TEST_CONFIG_FILE) ]; then \
		echo "$(YELLOW)Creating test configuration...$(RESET)"; \
		cp $(CONFIG_FILE) $(TEST_CONFIG_FILE) 2>/dev/null || \
		echo "$(YELLOW)Main config not found, using minimal test config$(RESET)"; \
	fi
	$(PYTHON) scripts/setup_test_env.py
	@echo "$(GREEN)Test environment setup complete!$(RESET)"

test: ## Run all tests (unit + integration)
	@echo "$(BLUE)Running comprehensive test suite...$(RESET)"
	$(PYTHON) -m pytest $(TEST_PATH) -v --tb=short
	@echo "$(GREEN) All tests completed!$(RESET)"

test-unit: setup-test-env ## Run unit tests with proper environment setup
	@echo "$(BLUE)Running unit tests with test environment...$(RESET)"
	@export TEST_CONFIG=$(TEST_CONFIG_FILE) && \
	$(PYTHON) -m pytest $(TEST_PATH) -v -m "unit or not integration" --tb=short
	@echo "$(GREEN) Unit tests completed!$(RESET)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(PYTHON) -m pytest $(TEST_PATH) -v -m "integration" --tb=short
	@echo "$(GREEN) Integration tests completed!$(RESET)"

test-coverage: setup-test-env ## Run tests with coverage analysis
	@echo "$(BLUE)Running tests with coverage analysis...$(RESET)"
	@export TEST_CONFIG=$(TEST_CONFIG_FILE) && \
	$(PYTHON) -m pytest $(TEST_PATH) \
		--cov=$(SRC_PATH) \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-report=xml
	@echo "$(GREEN) Coverage report generated in htmlcov/$(RESET)"

test-fast: setup-test-env ## Run fast tests only (exclude slow tests)
	@echo "$(BLUE)Running fast test suite...$(RESET)"
	@export TEST_CONFIG=$(TEST_CONFIG_FILE) && \
	$(PYTHON) -m pytest $(TEST_PATH) -v -m "not slow" --tb=short
	@echo "$(GREEN) Fast tests completed!$(RESET)"

test-with-config: setup-test-env ## Run tests with test configuration
	@echo "$(BLUE)Running tests with test configuration...$(RESET)"
	@export TEST_CONFIG=$(TEST_CONFIG_FILE) && \
	$(PYTHON) -m pytest $(TEST_PATH) -v --tb=short \
		--config-file=$(TEST_CONFIG_FILE)
	@echo "$(GREEN) Tests with configuration completed!$(RESET)"

health-check: ## Check system health and dependencies
	@echo "$(BLUE)Running system health checks...$(RESET)"
	@echo "$(YELLOW)Python Dependencies:$(RESET)"
	$(PYTHON) -c "import sys; print('✓ Python OK'); import pandas; print('✓ Pandas OK'); import numpy; print('✓ NumPy OK'); import sklearn; print('✓ Scikit-learn OK'); import boto3; print('✓ Boto3 OK')"
	@echo "$(YELLOW)System Resources:$(RESET)"
	$(PYTHON) -c "import psutil; print(f'✓ CPU: {psutil.cpu_percent()}%'); print(f'✓ Memory: {psutil.virtual_memory().percent}%'); print(f'✓ Disk: {psutil.disk_usage(\"/\").percent}%')"
	@echo "$(GREEN)Health check completed!$(RESET)"

test-aws-connection: ## Test AWS connectivity and permissions
	@echo "$(BLUE)Testing AWS connections...$(RESET)"
	$(PYTHON) -c "\
import boto3, yaml; \
try: \
    config = yaml.safe_load(open('config.yaml')); \
    region = config['aws']['region']; \
    bucket = config['aws']['s3']['bucket_name']; \
    \
    s3 = boto3.client('s3', region_name=region); \
    s3.list_objects_v2(Bucket=bucket, MaxKeys=1); \
    print('✓ S3 access confirmed'); \
    \
    sagemaker = boto3.client('sagemaker', region_name=region); \
    sagemaker.list_endpoints(MaxResults=1); \
    print('✓ SageMaker access confirmed'); \
    \
    athena = boto3.client('athena', region_name=region); \
    athena.list_work_groups(); \
    print('✓ Athena access confirmed'); \
    \
    print(' All AWS connections successful!'); \
except Exception as e: \
    print(f' AWS connection test failed: {e}'); \
    exit(1)"
	@echo "$(GREEN)AWS connection test completed!$(RESET)"

# =============================================================================
# 3. DATA PIPELINE
# =============================================================================

validate-data: ## Run comprehensive data validation
	@echo "$(BLUE)Running data validation...$(RESET)"
	@mkdir -p data/validation
	$(PYTHON) $(SRC_PATH)/data_processing/data_validation.py \
		--config config.yaml \
		--data-path data/raw/ \
		--output-path data/validation/
	@echo "$(GREEN)Data validation completed!$(RESET)"

process-features: ## Execute feature engineering pipeline
	@echo "$(BLUE)Running feature engineering...$(RESET)"
	@mkdir -p data/processed
	$(PYTHON) $(SRC_PATH)/data_processing/feature_engineering.py \
		--config config.yaml \
		--data-path data/raw/ \
		--output-path data/processed/
	@echo "$(GREEN)Feature engineering completed!$(RESET)"

setup-feature-store: ## Setup SageMaker Feature Store integration
	@echo "$(BLUE)Setting up Feature Store integration...$(RESET)"
	$(PYTHON) $(SRC_PATH)/data_processing/feature_store_integration.py \
		--config config.yaml \
		--data-path data/processed/ \
		--feature-store
	@echo "$(GREEN)Feature Store setup completed!$(RESET)"

setup-athena: ## Setup Athena tables for analytics
	@echo "$(BLUE)Setting up Athena tables...$(RESET)"
	$(PYTHON) $(SRC_PATH)/data_processing/feature_store_integration.py \
		--config config.yaml \
		--data-path data/processed/ \
		--athena
	@echo "$(GREEN)Athena tables setup completed!$(RESET)"

setup-data-infrastructure: ## Setup complete data infrastructure (Feature Store + Athena)
	@echo "$(BLUE)Setting up complete data infrastructure...$(RESET)"
	$(PYTHON) $(SRC_PATH)/data_processing/feature_store_integration.py \
		--config config.yaml \
		--data-path data/processed/ \
		--all
	@echo "$(GREEN)Complete data infrastructure setup completed!$(RESET)"

verify-athena: ## Verify Athena tables and connectivity
	@echo "$(BLUE)Verifying Athena setup...$(RESET)"
	$(PYTHON) -c "\
import boto3, yaml; \
config = yaml.safe_load(open('config.yaml')); \
athena = boto3.client('athena', region_name=config['aws']['region']); \
response = athena.start_query_execution( \
    QueryString='SHOW TABLES IN demand_stock_forecasting_mlops_feature_store', \
    ResultConfiguration={'OutputLocation': config['aws']['athena']['query_results_location']}, \
    WorkGroup=config['aws']['athena'].get('workgroup', 'primary') \
); \
print(f'✓ Athena query submitted: {response[\"QueryExecutionId\"]}'); \
print('✓ Athena tables are accessible!')"
	@echo "$(GREEN)Athena verification completed!$(RESET)"

# Query Examples
run-sample-queries: ## Run sample business intelligence queries
	@echo "$(BLUE)Running sample BI queries...$(RESET)"
	$(PYTHON) scripts/run_sample_queries.py
	@echo "$(GREEN)Sample queries completed!$(RESET)"

# Data Pipeline Workflows
pipeline-data-basic: validate-data process-features ## Basic data pipeline (validation + features)
	@echo "$(GREEN) Basic data pipeline completed!$(RESET)"

pipeline-data-analytics: validate-data process-features setup-athena ## Data pipeline with analytics (includes Athena)
	@echo "$(GREEN) Analytics data pipeline completed!$(RESET)"

pipeline-data-ml: validate-data process-features setup-feature-store ## Data pipeline optimized for ML (includes Feature Store)
	@echo "$(GREEN) ML data pipeline completed!$(RESET)"

pipeline-data-full: validate-data process-features setup-data-infrastructure ## Complete data pipeline with all integrations
	@echo "$(GREEN) Complete data pipeline executed!$(RESET)"

pipeline-bi: pipeline-data-full verify-athena run-sample-queries ## Business Intelligence pipeline with verification
	@echo "$(GREEN) Business Intelligence pipeline completed!$(RESET)"
	@echo "$(CYAN) Your data is ready for analysis in Athena!$(RESET)"
	@echo "$(YELLOW)Next steps:$(RESET)"
	@echo "  1. Open AWS Athena Console"
	@echo "  2. Select 'demand_stock_forecasting_mlops_feature_store' database"
	@echo "  3. Run: SHOW TABLES"
	@echo "  4. Query features: SELECT * FROM features_complete LIMIT 10"

# =============================================================================
# 4. MODEL TRAINING & REGISTRY
# =============================================================================

train-models: ## Train machine learning models with evaluation
	@echo "$(BLUE)Training models...$(RESET)"
	@mkdir -p $(MODEL_OUTPUT_PATH)
	$(PYTHON) $(SRC_PATH)/training/train_model.py \
		--config config.yaml \
		--data-path $(PROCESSED_DATA_PATH) \
		--output-path $(MODEL_OUTPUT_PATH) \
		--model-version $(MODEL_VERSION) \
		--forecast-horizon $(FORECAST_HORIZON)
	@echo "$(GREEN) Model training completed!$(RESET)"

register-models: ## Register trained models to SageMaker Model Registry
	@echo "$(BLUE)Registering trained models...$(RESET)"
	$(PYTHON) $(SRC_PATH)/deployment/model_registry.py \
		--config config.yaml \
		--action register \
		--models-dir models \
		--evaluation-file models/evaluation.json
	@echo "$(GREEN) Models registered successfully!$(RESET)"

list-models: ## List all registered model versions
	@echo "$(BLUE)Listing registered models...$(RESET)"
	$(PYTHON) $(SRC_PATH)/deployment/model_registry.py \
		--config config.yaml \
		--action list
	@echo "$(GREEN)Model listing completed!$(RESET)"

promote-model: ## Promote model to next stage (dev->staging->prod)
	@echo "$(BLUE)Promoting model: $(MODEL_NAME) from $(SOURCE_STAGE) to $(TARGET_STAGE)...$(RESET)"
	$(PYTHON) $(SRC_PATH)/deployment/model_registry.py \
		--config config.yaml \
		--action promote \
		--model-name $(MODEL_NAME) \
		--source-stage $(SOURCE_STAGE) \
		--target-stage $(TARGET_STAGE)
	@echo "$(GREEN) Model promoted successfully!$(RESET)"

retire-model: ## Retire a specific model version
	@echo "$(BLUE)Retiring model $(MODEL_NAME) version $(MODEL_VERSION)...$(RESET)"
	$(PYTHON) $(SRC_PATH)/deployment/model_registry.py \
		--config config.yaml \
		--action retire \
		--model-name $(MODEL_NAME) \
		--model-version $(MODEL_VERSION)
	@echo "$(GREEN)Model retired successfully!$(RESET)"

# Training Workflows
pipeline-train-basic: pipeline-data-ml train-models ## Basic training pipeline (ML data + training)
	@echo "$(GREEN) Basic training pipeline completed!$(RESET)"

pipeline-train-full: pipeline-data-full train-models register-models ## Complete training pipeline with registry
	@echo "$(GREEN) Complete training pipeline executed!$(RESET)"

# =============================================================================
# 5. MODEL DEPLOYMENT
# =============================================================================

deploy-dev: ## Deploy model to development environment
	@echo "$(BLUE)Deploying to development environment...$(RESET)"
	$(PYTHON) $(SRC_PATH)/deployment/sagemaker_deploy.py \
		--config config.yaml \
		--action deploy \
		--model-path $(MODEL_PATH) \
		--model-name $(MODEL_NAME) \
		--endpoint-name $(ENDPOINT_BASE_NAME)-dev \
		--environment dev
	@echo "$(GREEN) Development deployment completed!$(RESET)"

deploy-staging: ## Deploy model to staging environment
	@echo "$(BLUE)Deploying to staging environment...$(RESET)"
	$(PYTHON) $(SRC_PATH)/deployment/sagemaker_deploy.py \
		--config config.yaml \
		--action deploy \
		--model-path $(MODEL_PATH) \
		--model-name $(MODEL_NAME) \
		--endpoint-name $(ENDPOINT_BASE_NAME)-staging \
		--environment staging
	@echo "$(GREEN) Staging deployment completed!$(RESET)"

deploy-prod: ## Deploy model to production environment
	@echo "$(BLUE)Deploying to production environment...$(RESET)"
	$(PYTHON) $(SRC_PATH)/deployment/sagemaker_deploy.py \
		--config config.yaml \
		--action deploy \
		--model-path $(MODEL_PATH) \
		--model-name $(MODEL_NAME) \
		--endpoint-name $(ENDPOINT_BASE_NAME)-prod \
		--environment prod
	@echo "$(GREEN) Production deployment completed!$(RESET)"

list-endpoints: ## List all active SageMaker endpoints
	@echo "$(BLUE)Listing SageMaker endpoints...$(RESET)"
	$(PYTHON) $(SRC_PATH)/deployment/sagemaker_deploy.py \
		--config config.yaml \
		--action list
	@echo "$(GREEN)Endpoint listing completed!$(RESET)"

test-endpoint: ## Test deployed SageMaker endpoint functionality
	@echo "$(BLUE)Testing endpoint: $(ENDPOINT_NAME)...$(RESET)"
	$(PYTHON) $(SRC_PATH)/deployment/sagemaker_deploy.py \
		--config config.yaml \
		--action test \
		--endpoint-name $(ENDPOINT_NAME)
	@echo "$(GREEN) Endpoint testing completed!$(RESET)"

delete-endpoint: ## Delete specified SageMaker endpoint
	@echo "$(BLUE)Deleting endpoint: $(ENDPOINT_NAME)...$(RESET)"
	@echo "$(YELLOW)  This will permanently delete the endpoint. Continue? (Ctrl+C to cancel)$(RESET)"
	@echo "Press Enter to continue..."; read dummy
	$(PYTHON) $(SRC_PATH)/deployment/sagemaker_deploy.py \
		--config config.yaml \
		--action delete \
		--endpoint-name $(ENDPOINT_NAME)
	@echo "$(GREEN)Endpoint deleted successfully!$(RESET)"

# =============================================================================
# 6. API & INFERENCE SERVICES
# =============================================================================

api-start: ## Start FastAPI service for local inference
	@echo "$(BLUE)Starting FastAPI service...$(RESET)"
	@echo "$(YELLOW) API will be available at http://localhost:$(API_PORT)$(RESET)"
	@echo "$(YELLOW)📚 Documentation at http://localhost:$(API_PORT)/docs$(RESET)"
	export CONFIG_PATH=config.yaml && \
	$(PYTHON) $(SRC_PATH)/inference/api.py \
		--host $(API_HOST) \
		--port $(API_PORT)

api-dev: ## Start API in development mode with auto-reload
	@echo "$(BLUE)Starting FastAPI in development mode...$(RESET)"
	@echo "$(YELLOW) API: http://localhost:$(API_PORT) (auto-reload enabled)$(RESET)"
	export CONFIG_PATH=config.yaml && \
	uvicorn src.inference.api:app \
		--host $(API_HOST) \
		--port $(API_PORT) \
		--reload

api-test: ## Test API endpoints comprehensively
	@echo "$(BLUE)Testing API endpoints...$(RESET)"
	$(PYTHON) scripts/api_test.py --base-url http://localhost:$(API_PORT)
	@echo "$(GREEN) API testing completed!$(RESET)"

api-health: ## Check API health and responsiveness
	@echo "$(BLUE)Checking API health...$(RESET)"
	@curl -s http://localhost:$(API_PORT)/health > /dev/null && \
		echo "$(GREEN) API is healthy and responding$(RESET)" || \
		echo "$(RED) API is not responding$(RESET)"

api-stop: ## Stop running API service
	@echo "$(BLUE)Stopping API service...$(RESET)"
	@pkill -f "uvicorn.*src.inference.api" > /dev/null 2>&1 && \
		echo "$(GREEN) API service stopped$(RESET)" || \
		echo "$(YELLOW)  No API process found$(RESET)"

# API Workflows
api-local: api-start ## Start complete local API service
	@echo "$(GREEN) Local API service started!$(RESET)"

# =============================================================================
# 7. MONITORING & OBSERVABILITY - UPDATED TARGETS
# =============================================================================

monitoring-start: ## Start all monitoring systems
	@echo "$(BLUE)Starting comprehensive monitoring systems...$(RESET)"
	@echo "$(YELLOW)Stopping any existing monitoring processes...$(RESET)"
	@make monitoring-stop > /dev/null 2>&1 || true
	@sleep 2
	@echo "$(YELLOW)Performance Monitor: Starting...$(RESET)"
	$(PYTHON) $(SRC_PATH)/monitoring/performance_monitor.py \
		--config config.yaml \
		--action start \
		--interval $(MONITORING_INTERVAL) \
		--local-mode &
	@sleep 2
	@echo "$(YELLOW)Drift Detection: Starting...$(RESET)"
	$(PYTHON) $(SRC_PATH)/monitoring/drift_detector.py \
		--config config.yaml \
		--action monitor \
		--reference-data $(REFERENCE_DATA) \
		--current-data $(CURRENT_DATA) \
		--local-mode \
		--interval 60 &
	@sleep 3
	@echo "$(YELLOW)Dashboard: Starting on available port...$(RESET)"
	@DASHBOARD_PORT=$$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); port=s.getsockname()[1]; s.close(); print(port)"); \
	$(PYTHON) $(SRC_PATH)/monitoring/performance_monitor.py \
		--config config.yaml \
		--action dashboard \
		--port $$DASHBOARD_PORT \
		--local-mode & \
	echo "Dashboard starting on port: $$DASHBOARD_PORT"
	@sleep 3
	@echo "$(GREEN)All monitoring systems started!$(RESET)"
	@echo "$(CYAN)Check 'make monitoring-status' for details$(RESET)"

monitoring-status: ## Check detailed status of all monitoring systems
	@echo "$(CYAN) Monitoring System Status:$(RESET)"
	@echo "$(YELLOW) Port Status:$(RESET)"
	@for port in 8002 8050 8051 8052; do \
		if lsof -i:$$port > /dev/null 2>&1; then \
			pid=$$(lsof -ti:$$port 2>/dev/null); \
			echo "  Port $$port: $(GREEN)OCCUPIED$(RESET) (PID: $$pid)"; \
		else \
			echo "  Port $$port: $(RED)FREE$(RESET)"; \
		fi; \
	done
	@echo "$(YELLOW) Process Status:$(RESET)"
	@if pgrep -f "performance_monitor.py.*start" > /dev/null; then \
		echo "  $(GREEN) Performance Monitor: Running$(RESET)"; \
	else \
		echo "  $(RED) Performance Monitor: Stopped$(RESET)"; \
	fi
	@if pgrep -f "drift_detector.py.*monitor" > /dev/null; then \
		echo "  $(GREEN) Drift Detector: Running$(RESET)"; \
	else \
		echo "  $(RED) Drift Detector: Stopped$(RESET)"; \
	fi
	@if pgrep -f "performance_monitor.py.*dashboard" > /dev/null; then \
		echo "  $(GREEN) Dashboard: Running$(RESET)"; \
	else \
		echo "  $(RED) Dashboard: Stopped$(RESET)"; \
	fi
	@echo "$(YELLOW) Service Accessibility:$(RESET)"
	@for port in 8002 8050 8051 8052; do \
		if curl -s http://localhost:$$port > /dev/null 2>&1; then \
			echo "  $(GREEN) localhost:$$port accessible$(RESET)"; \
		elif curl -s http://localhost:$$port/health > /dev/null 2>&1; then \
			echo "  $(GREEN) localhost:$$port/health accessible$(RESET)"; \
		fi; \
	done

monitoring-stop: ## Stop all monitoring processes with enhanced cleanup
	@echo "$(BLUE)Stopping all monitoring systems...$(RESET)"
	@echo "$(YELLOW) Killing processes by port...$(RESET)"
	@for port in 8002 8050 8051 8052; do \
		pid=$$(lsof -ti:$$port 2>/dev/null); \
		if [ ! -z "$$pid" ]; then \
			echo "  Killing process on port $$port (PID: $$pid)"; \
			kill -9 $$pid 2>/dev/null || true; \
		fi; \
	done
	@echo "$(YELLOW) Killing processes by pattern...$(RESET)"
	@pkill -f "performance_monitor.py" > /dev/null 2>&1 || echo "  No performance monitor found"
	@pkill -f "drift_detector.py" > /dev/null 2>&1 || echo "  No drift detector found"
	@pkill -f "monitoring.*dashboard" > /dev/null 2>&1 || echo "  No dashboard found"
	@pkill -f "uvicorn.*monitoring" > /dev/null 2>&1 || echo "  No uvicorn monitoring found"
	@pkill -f "dash.*monitoring" > /dev/null 2>&1 || echo "  No dash monitoring found"
	@sleep 2
	@echo "$(GREEN) All monitoring systems stopped!$(RESET)"

monitoring-restart: monitoring-stop monitoring-start ## Restart all monitoring systems
	@echo "$(GREEN) Monitoring systems restarted!$(RESET)"

detect-drift: ## Run one-time drift detection analysis with enhanced output
	@echo "$(BLUE)Running drift detection analysis...$(RESET)"
	$(PYTHON) $(SRC_PATH)/monitoring/drift_detector.py \
		--config config.yaml \
		--action detect \
		--reference-data $(REFERENCE_DATA) \
		--current-data $(CURRENT_DATA) \
		--local-mode \
		--generate-report
	@echo "$(GREEN) Drift detection completed!$(RESET)"

performance-report: ## Generate comprehensive performance report
	@echo "$(BLUE)Generating performance report...$(RESET)"
	$(PYTHON) $(SRC_PATH)/monitoring/performance_monitor.py \
		--config config.yaml \
		--action health \
		--local-mode
	@echo "$(GREEN) Performance report generated!$(RESET)"

monitoring-logs: ## Show logs from monitoring processes
	@echo "$(CYAN) Recent Monitoring Logs:$(RESET)"
	@echo "$(YELLOW)Performance Monitor:$(RESET)"
	@tail -20 data/monitoring/logs/performance.log 2>/dev/null || echo "  No performance logs found"
	@echo "$(YELLOW)Drift Detection:$(RESET)"
	@tail -20 data/monitoring/logs/drift.log 2>/dev/null || echo "  No drift logs found"
	@echo "$(YELLOW)System Processes:$(RESET)"
	@ps aux | grep -E "(performance_monitor|drift_detector)" | grep -v grep || echo "  No monitoring processes found"

monitoring-test: ## Test monitoring system components
	@echo "$(BLUE)Testing monitoring system components...$(RESET)"
	@echo "$(YELLOW) Testing Performance Monitor...$(RESET)"
	$(PYTHON) $(SRC_PATH)/monitoring/performance_monitor.py \
		--config config.yaml \
		--action health \
		--local-mode || echo "  Performance monitor test failed"
	@echo "$(YELLOW) Testing Drift Detector...$(RESET)"
	$(PYTHON) $(SRC_PATH)/monitoring/drift_detector.py \
		--config config.yaml \
		--action alert-test \
		--local-mode || echo "  Drift detector test failed"
	@echo "$(GREEN) Monitoring tests completed!$(RESET)"

# Emergency Operations
emergency-stop: ## Emergency stop of all running services including monitoring
	@echo "$(RED) EMERGENCY STOP - Stopping all services...$(RESET)"
	@make api-stop > /dev/null 2>&1 || true
	@make monitoring-stop > /dev/null 2>&1 || true
	@pkill -f "uvicorn" > /dev/null 2>&1 || true
	@pkill -f "streamlit" > /dev/null 2>&1 || true
	@pkill -f "dash" > /dev/null 2>&1 || true
	@pkill -f "jupyter" > /dev/null 2>&1 || true
	@echo "$(GREEN) Emergency stop completed!$(RESET)"

monitoring-dashboard-only: ## Start only the dashboard component
	@echo "$(BLUE)Starting monitoring dashboard only...$(RESET)"
	@DASHBOARD_PORT=$$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); port=s.getsockname()[1]; s.close(); print(port)"); \
	echo "$(CYAN) Starting dashboard on port: $$DASHBOARD_PORT$(RESET)"; \
	$(PYTHON) $(SRC_PATH)/monitoring/performance_monitor.py \
		--config config.yaml \
		--action dashboard \
		--port $$DASHBOARD_PORT \
		--local-mode

monitoring-performance-only: ## Start only performance monitoring
	@echo "$(BLUE)Starting performance monitoring only...$(RESET)"
	$(PYTHON) $(SRC_PATH)/monitoring/performance_monitor.py \
		--config config.yaml \
		--action start \
		--interval $(MONITORING_INTERVAL) \
		--local-mode

monitoring-drift-only: ## Start only drift detection monitoring
	@echo "$(BLUE)Starting drift monitoring only...$(RESET)"
	$(PYTHON) $(SRC_PATH)/monitoring/drift_detector.py \
		--config config.yaml \
		--action monitor \
		--reference-data $(REFERENCE_DATA) \
		--current-data $(CURRENT_DATA) \
		--local-mode \
		--interval 60

# Helper targets for debugging
monitoring-debug: ## Debug monitoring system issues
	@echo "$(CYAN) Monitoring System Debug Information:$(RESET)"
	@echo "$(YELLOW)Python Path:$(RESET) $(shell which python3)"
	@echo "$(YELLOW)Current Directory:$(RESET) $(shell pwd)"
	@echo "$(YELLOW)Config File:$(RESET)"
	@ls -la config.yaml 2>/dev/null || echo "  config.yaml not found"
	@echo "$(YELLOW)Data Files:$(RESET)"
	@ls -la $(REFERENCE_DATA) $(CURRENT_DATA) 2>/dev/null || echo "  Data files not found"
	@echo "$(YELLOW)Monitoring Scripts:$(RESET)"
	@ls -la $(SRC_PATH)/monitoring/ 2>/dev/null || echo "  Monitoring scripts not found"
	@echo "$(YELLOW)Running Processes:$(RESET)"
	@ps aux | grep -E "(python|monitoring|dash)" | grep -v grep || echo "  No relevant processes found"

monitoring-clean: ## Clean monitoring temporary files and logs
	@echo "$(BLUE)Cleaning monitoring temporary files...$(RESET)"
	@rm -rf data/monitoring/logs/* 2>/dev/null || true
	@rm -rf data/monitoring/reports/* 2>/dev/null || true
	@rm -rf data/monitoring/metrics/* 2>/dev/null || true
	@echo "$(GREEN) Monitoring cleanup completed!$(RESET)"

# =============================================================================
# 8. TESTING & QUALITY ASSURANCE
# =============================================================================

test: ## Run all tests (unit + integration)
	@echo "$(BLUE)Running comprehensive test suite...$(RESET)"
	$(PYTHON) -m pytest $(TEST_PATH) -v --tb=short
	@echo "$(GREEN) All tests completed!$(RESET)"

test-unit: setup-test-env ## Run unit tests with proper environment setup
	@echo "$(BLUE)Running unit tests with test environment...$(RESET)"
	@if [ -f $(TEST_CONFIG_FILE) ]; then \
		export TEST_CONFIG=$(TEST_CONFIG_FILE); \
	fi && \
	$(PYTHON) -m pytest $(TEST_PATH) -v -m "unit or not integration" --tb=short
	@echo "$(GREEN) Unit tests completed!$(RESET)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(PYTHON) -m pytest $(TEST_PATH) -v -m "integration" --tb=short
	@echo "$(GREEN) Integration tests completed!$(RESET)"

test-coverage: setup-test-env ## Run tests with coverage analysis
	@echo "$(BLUE)Running tests with coverage analysis...$(RESET)"
	@if [ -f $(TEST_CONFIG_FILE) ]; then \
		export TEST_CONFIG=$(TEST_CONFIG_FILE); \
	fi && \
	$(PYTHON) -m pytest $(TEST_PATH) \
		--cov=$(SRC_PATH) \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-report=xml
	@echo "$(GREEN) Coverage report generated in htmlcov/$(RESET)"

test-fast: setup-test-env ## Run fast tests only (exclude slow tests)
	@echo "$(BLUE)Running fast test suite...$(RESET)"
	@if [ -f $(TEST_CONFIG_FILE) ]; then \
		export TEST_CONFIG=$(TEST_CONFIG_FILE); \
	fi && \
	$(PYTHON) -m pytest $(TEST_PATH) -v -m "not slow" --tb=short
	@echo "$(GREEN) Fast tests completed!$(RESET)"

test-with-config: setup-test-env ## Run tests with test configuration
	@echo "$(BLUE)Running tests with test configuration...$(RESET)"
	@export TEST_CONFIG=$(TEST_CONFIG_FILE) && \
	$(PYTHON) -m pytest $(TEST_PATH) -v --tb=short \
		--config-file=$(TEST_CONFIG_FILE)
	@echo "$(GREEN) Tests with configuration completed!$(RESET)"

# Code Quality
format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	black $(SRC_PATH) $(TEST_PATH) --line-length=100
	isort $(SRC_PATH) $(TEST_PATH) --profile black
	@echo "$(GREEN) Code formatting completed!$(RESET)"

lint: ## Run comprehensive code linting
	@echo "$(BLUE)Running code linting...$(RESET)"
	@echo "$(YELLOW) Flake8...$(RESET)"
	flake8 $(SRC_PATH) --max-line-length=100 --ignore=E203,W503
	@echo "$(YELLOW) MyPy...$(RESET)"
	mypy $(SRC_PATH) --ignore-missing-imports || echo "$(YELLOW)  MyPy warnings found$(RESET)"
	@echo "$(GREEN) Linting completed!$(RESET)"

security-check: ## Run security vulnerability checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	@echo "$(YELLOW) Bandit security scan...$(RESET)"
	bandit -r $(SRC_PATH) -f json -o security_report.json || \
		echo "$(YELLOW)  Security issues found, check security_report.json$(RESET)"
	@echo "$(YELLOW) Safety dependency scan...$(RESET)"
	safety check || echo "$(YELLOW)  Dependency vulnerabilities found$(RESET)"
	@echo "$(GREEN) Security checks completed!$(RESET)"

# Quality Workflows
quality-check: format lint ## Quick code quality check
	@echo "$(GREEN) Code quality check completed!$(RESET)"

test-full: format lint security-check test-coverage ## Complete testing pipeline
	@echo "$(GREEN) Full testing pipeline completed!$(RESET)"

# =============================================================================
# 9. CI/CD SIMULATION
# =============================================================================

ci-install: ## Simulate CI dependency installation
	@echo "$(BLUE) Simulating CI install step...$(RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 mypy bandit safety isort
	@echo "$(GREEN) CI install simulation completed!$(RESET)"

ci-test: setup-test-env ## Simulate CI testing phase with test environment
	@echo "$(BLUE) Simulating CI test step...$(RESET)"
	@export TEST_CONFIG=$(TEST_CONFIG_FILE) && \
	$(PYTHON) -m pytest $(TEST_PATH) -v --tb=short --junitxml=test-results.xml
	@echo "$(GREEN) CI test simulation completed!$(RESET)"

ci-quality: ## Simulate CI quality gate checks
	@echo "$(BLUE) Simulating CI quality checks...$(RESET)"
	black --check $(SRC_PATH) $(TEST_PATH) --line-length=100
	flake8 $(SRC_PATH) --max-line-length=100 --ignore=E203,W503
	bandit -r $(SRC_PATH) -f json -o security_report.json
	@echo "$(GREEN) CI quality checks completed!$(RESET)"

ci-full: ci-install ci-quality ci-test ## Complete CI/CD pipeline simulation
	@echo "$(GREEN) Full CI/CD simulation completed!$(RESET)"

# =============================================================================
# 10. MASTER WORKFLOWS
# =============================================================================

# Development Workflows
workflow-dev: setup-dev pipeline-data-full train-models register-models test-fast ## Complete development workflow
	@echo "$(GREEN) Development workflow completed!$(RESET)"

workflow-test: setup-test-env pipeline-data-basic test-full ## Testing-focused workflow with environment setup
	@echo "$(GREEN) Testing workflow completed!$(RESET)"

# Deployment Workflows
workflow-staging: pipeline-train-full deploy-staging test-endpoint ## Complete staging deployment
	@echo "$(GREEN) Staging deployment workflow completed!$(RESET)"

workflow-prod: deploy-prod monitoring-start ## Production deployment with monitoring
	@echo "$(GREEN) Production deployment workflow completed!$(RESET)"

# Master Pipelines
pipeline-full: pipeline-data-full train-models test ## Complete MLOps pipeline (data + models + tests)
	@echo "$(GREEN) Complete MLOps pipeline executed successfully!$(RESET)"

pipeline-production: pipeline-full deploy-staging ## Full pipeline + staging deployment
	@echo "$(GREEN) Production pipeline completed!$(RESET)"

# =============================================================================
# 11. UTILITIES & MAINTENANCE
# =============================================================================

clean: ## Clean temporary files and caches
	@echo "$(BLUE)Cleaning temporary files...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/ htmlcov/ .coverage build/ dist/
	rm -f security_report.json test-results.xml
	@echo "$(GREEN) Cleanup completed!$(RESET)"

clean-data: ## Clean processed data files (use with caution)
	@echo "$(YELLOW)  This will delete all processed data. Continue? (Ctrl+C to cancel)$(RESET)"
	@read -p "Press Enter to continue..."
	@echo "$(BLUE)Cleaning processed data...$(RESET)"
	rm -rf $(DATA_PATH)/processed/* $(DATA_PATH)/validation/* models/*.pkl
	@echo "$(GREEN) Data cleanup completed!$(RESET)"

clean-test-env: ## Clean test environment files
	@echo "$(BLUE)Cleaning test environment...$(RESET)"
	rm -rf tests/data/ tests/config/ 2>/dev/null || true
	@echo "$(GREEN) Test environment cleaned!$(RESET)"

clean-athena: ## Clean up Athena tables (use with caution)
	@echo "$(YELLOW)  This will drop all Athena tables. Continue? (Ctrl+C to cancel)$(RESET)"
	@read -p "Press Enter to continue..."
	@echo "$(BLUE)Cleaning Athena tables...$(RESET)"
	$(PYTHON) -c "\
import boto3, yaml; \
config = yaml.safe_load(open('config.yaml')); \
athena = boto3.client('athena', region_name=config['aws']['region']); \
tables = ['features_complete', 'train_data', 'validation_data', 'test_data']; \
database = config['aws']['athena']['database_name']; \
for table in tables: \
    try: \
        query = f'DROP TABLE IF EXISTS {database}.{table}'; \
        athena.start_query_execution( \
            QueryString=query, \
            ResultConfiguration={'OutputLocation': config['aws']['athena']['query_results_location']}, \
            WorkGroup=config['aws']['athena'].get('workgroup', 'primary') \
        ); \
        print(f'✓ Dropped table: {table}'); \
    except Exception as e: \
        print(f'  Could not drop {table}: {e}'); \
print(' Athena cleanup completed!')"
	@echo "$(GREEN)Athena tables cleaned up!$(RESET)"

# Git Workflow
git-setup: ## Setup git configuration
	@echo "$(BLUE)Setting up git configuration...$(RESET)"
	git config --local user.name "Bhupal Lambodhar" || echo "$(YELLOW)Could not set git name$(RESET)"
	git config --local user.email "btiduwarlambodhar@sandiego.edu" || echo "$(YELLOW)Could not set git email$(RESET)"
	@echo "$(GREEN) Git setup completed!$(RESET)"

git-status: ## Check git status and suggest workflow
	@echo "$(CYAN) Git Status Check:$(RESET)"
	@git status 2>/dev/null || echo "$(RED) Not a git repository$(RESET)"
	@echo "$(YELLOW) Suggested workflow:$(RESET)"
	@echo "  1. make workflow-test       # Test your changes"
	@echo "  2. git add .                # Stage changes"
	@echo "  3. git commit -m 'message'  # Commit changes"
	@echo "  4. git push                 # Push to remote"

# Documentation
docs-generate: ## Generate project documentation
	@echo "$(BLUE)Generating documentation...$(RESET)"
	@mkdir -p docs/
	@echo "# MLOps Project Documentation" > docs/README.md
	@echo "Generated on: $(shell date)" >> docs/README.md
	@echo "$(GREEN) Documentation generated in docs/$(RESET)"

# Build
build: clean format lint ## Build project for distribution
	@echo "$(BLUE)Building project...$(RESET)"
	$(PYTHON) setup.py sdist bdist_wheel 2>/dev/null || \
		echo "$(YELLOW)  No setup.py found, skipping package build$(RESET)"
	@echo "$(GREEN) Project build completed!$(RESET)"