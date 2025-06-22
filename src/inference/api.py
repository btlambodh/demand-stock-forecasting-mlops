#!/usr/bin/env python3
"""
FastAPI Real-time Inference API for Chinese Produce Market Forecasting
Production-grade API with validation, monitoring, and error handling

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Union
import sys

import yaml
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import start_http_server
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Prometheus metrics
PREDICTION_REQUESTS = Counter('prediction_requests_total', 'Total prediction requests', ['model_name', 'status'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency in seconds', ['model_name'])
API_REQUESTS = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method', 'status'])


# Pydantic models for request/response validation
class FeatureInput(BaseModel):
    """Single feature input for prediction"""
    
    # Core features
    Total_Quantity: float = Field(..., gt=0, description="Total quantity sold")
    Avg_Price: float = Field(..., gt=0, description="Average price")
    Transaction_Count: int = Field(..., gt=0, description="Number of transactions")
    
    # Price features
    Price_Volatility: Optional[float] = Field(0.0, ge=0, description="Price volatility")
    Min_Price: Optional[float] = Field(None, gt=0, description="Minimum price")
    Max_Price: Optional[float] = Field(None, gt=0, description="Maximum price")
    Retail_Wholesale_Ratio: Optional[float] = Field(1.0, gt=0, description="Retail to wholesale price ratio")
    
    # Temporal features
    Month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    DayOfWeek: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    Quarter: Optional[int] = Field(None, ge=1, le=4, description="Quarter")
    IsWeekend: int = Field(..., ge=0, le=1, description="Is weekend (0/1)")
    
    # Seasonal features
    Month_Sin: Optional[float] = Field(None, ge=-1, le=1, description="Cyclical month encoding (sin)")
    Month_Cos: Optional[float] = Field(None, ge=-1, le=1, description="Cyclical month encoding (cos)")
    DayOfWeek_Sin: Optional[float] = Field(None, ge=-1, le=1, description="Cyclical day encoding (sin)")
    DayOfWeek_Cos: Optional[float] = Field(None, ge=-1, le=1, description="Cyclical day encoding (cos)")
    
    # Category features
    Category_Avg_Price: Optional[float] = Field(None, gt=0, description="Category average price")
    Item_Revenue_Share: Optional[float] = Field(0.0, ge=0, le=1, description="Item revenue share")
    Price_Relative_to_Category: Optional[float] = Field(1.0, gt=0, description="Price relative to category")
    
    # Lag features
    Avg_Price_Lag_1: Optional[float] = Field(None, gt=0, description="Price lag 1 day")
    Avg_Price_Lag_7: Optional[float] = Field(None, gt=0, description="Price lag 7 days")
    Total_Quantity_Lag_1: Optional[float] = Field(None, ge=0, description="Quantity lag 1 day")
    Total_Quantity_Lag_7: Optional[float] = Field(None, ge=0, description="Quantity lag 7 days")
    
    # Rolling features
    Avg_Price_MA_7: Optional[float] = Field(None, gt=0, description="7-day price moving average")
    Avg_Price_MA_30: Optional[float] = Field(None, gt=0, description="30-day price moving average")
    Total_Quantity_MA_7: Optional[float] = Field(None, ge=0, description="7-day quantity moving average")
    
    # External features
    Loss_Rate_Percent: Optional[float] = Field(10.0, ge=0, le=100, description="Loss rate percentage")
    
    @validator('Max_Price')
    def max_price_greater_than_min(cls, v, values):
        if v is not None and 'Min_Price' in values and values['Min_Price'] is not None:
            if v <= values['Min_Price']:
                raise ValueError('Max_Price must be greater than Min_Price')
        return v
    
    @validator('Month_Sin', 'Month_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos')
    def auto_calculate_cyclical(cls, v, values, field):
        """Auto-calculate cyclical encodings if not provided"""
        if v is None:
            if field.name == 'Month_Sin' and 'Month' in values:
                return np.sin(2 * np.pi * values['Month'] / 12)
            elif field.name == 'Month_Cos' and 'Month' in values:
                return np.cos(2 * np.pi * values['Month'] / 12)
            elif field.name == 'DayOfWeek_Sin' and 'DayOfWeek' in values:
                return np.sin(2 * np.pi * values['DayOfWeek'] / 7)
            elif field.name == 'DayOfWeek_Cos' and 'DayOfWeek' in values:
                return np.cos(2 * np.pi * values['DayOfWeek'] / 7)
        return v


class BatchFeatureInput(BaseModel):
    """Batch prediction input"""
    instances: List[FeatureInput] = Field(..., min_items=1, max_items=1000, 
                                         description="List of feature instances")
    model_name: Optional[str] = Field("best_model", description="Model to use for prediction")


class PredictionOutput(BaseModel):
    """Single prediction output"""
    predicted_price: float = Field(..., description="Predicted price")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    model_used: str = Field(..., description="Model used for prediction")
    prediction_timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionOutput(BaseModel):
    """Batch prediction output"""
    predictions: List[PredictionOutput] = Field(..., description="List of predictions")
    batch_id: str = Field(..., description="Batch identifier")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_used: str = Field(..., description="Model used for predictions")


class HealthCheck(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    models_loaded: int = Field(..., description="Number of models loaded")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class APIError(BaseModel):
    """API error response"""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


# Global variables
app_start_time = time.time()
config = None
predictor_instance = None


class ProduceForecasterAPI:
    """Main API class for produce price forecasting"""
    
    def __init__(self, config_path: str):
        """Initialize API with configuration"""
        global config
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config
        self.models = {}
        self.app_version = config['project']['version']
        
        # Import predictor here to avoid circular imports
        try:
            from .predictor import ModelPredictor
            self.predictor = ModelPredictor(config_path)
            self.load_models()
        except ImportError:
            logger.error("Could not import ModelPredictor. Make sure predictor.py is available.")
            self.predictor = None
        
        logger.info("ProduceForecasterAPI initialized", version=self.app_version)
    
    def load_models(self):
        """Load available models"""
        try:
            if self.predictor:
                self.models = self.predictor.load_production_models()
                logger.info("Models loaded successfully", model_count=len(self.models))
            else:
                logger.warning("Predictor not available, using mock models")
                self.models = {"mock_model": {"type": "mock", "loaded": True}}
        except Exception as e:
            logger.error("Error loading models", error=str(e))
            self.models = {}
    
    def get_model_list(self) -> List[str]:
        """Get list of available models"""
        return list(self.models.keys())
    
    def predict_single(self, features: FeatureInput, model_name: str = "best_model") -> PredictionOutput:
        """Make single prediction"""
        start_time = time.time()
        
        try:
            # Convert features to DataFrame
            feature_dict = features.dict()
            df = pd.DataFrame([feature_dict])
            
            # Make prediction
            if self.predictor and model_name in self.models:
                prediction_result = self.predictor.predict(df, model_name)
                predicted_price = prediction_result['predictions'][0]
                confidence = prediction_result['confidence'][0]
                actual_model = prediction_result['model_used']
            else:
                # Mock prediction for testing
                predicted_price = features.Avg_Price * 1.05  # 5% increase
                confidence = 0.85
                actual_model = "mock_model"
            
            # Record metrics
            latency = time.time() - start_time
            PREDICTION_LATENCY.labels(model_name=model_name).observe(latency)
            PREDICTION_REQUESTS.labels(model_name=model_name, status='success').inc()
            
            logger.info("Single prediction completed", 
                       model=actual_model, 
                       latency=latency,
                       predicted_price=predicted_price)
            
            return PredictionOutput(
                predicted_price=round(predicted_price, 2),
                confidence=round(confidence, 3),
                model_used=actual_model,
                prediction_timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            PREDICTION_REQUESTS.labels(model_name=model_name, status='error').inc()
            logger.error("Error in single prediction", error=str(e))
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    def predict_batch(self, batch_input: BatchFeatureInput) -> BatchPredictionOutput:
        """Make batch predictions"""
        start_time = time.time()
        batch_id = f"batch_{int(time.time())}_{len(batch_input.instances)}"
        
        try:
            # Convert features to DataFrame
            feature_dicts = [instance.dict() for instance in batch_input.instances]
            df = pd.DataFrame(feature_dicts)
            
            model_name = batch_input.model_name
            
            # Make predictions
            if self.predictor and model_name in self.models:
                prediction_result = self.predictor.predict(df, model_name)
                predictions = prediction_result['predictions']
                confidences = prediction_result['confidence']
                actual_model = prediction_result['model_used']
            else:
                # Mock predictions for testing
                predictions = [feat['Avg_Price'] * 1.05 for feat in feature_dicts]
                confidences = [0.85] * len(feature_dicts)
                actual_model = "mock_model"
            
            # Create output objects
            outputs = []
            timestamp = datetime.now().isoformat()
            
            for pred, conf in zip(predictions, confidences):
                outputs.append(PredictionOutput(
                    predicted_price=round(pred, 2),
                    confidence=round(conf, 3),
                    model_used=actual_model,
                    prediction_timestamp=timestamp
                ))
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # milliseconds
            
            # Record metrics
            PREDICTION_LATENCY.labels(model_name=model_name).observe(time.time() - start_time)
            PREDICTION_REQUESTS.labels(model_name=model_name, status='success').inc()
            
            logger.info("Batch prediction completed", 
                       model=actual_model,
                       batch_id=batch_id,
                       batch_size=len(batch_input.instances),
                       processing_time_ms=processing_time)
            
            return BatchPredictionOutput(
                predictions=outputs,
                batch_id=batch_id,
                processing_time_ms=round(processing_time, 2),
                model_used=actual_model
            )
            
        except Exception as e:
            PREDICTION_REQUESTS.labels(model_name=batch_input.model_name, status='error').inc()
            logger.error("Error in batch prediction", batch_id=batch_id, error=str(e))
            raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# Initialize FastAPI app
app = FastAPI(
    title="Chinese Produce Market Forecasting API",
    description="Production-grade API for real-time produce price forecasting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request tracking middleware
@app.middleware("http")
async def track_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    # Record API metrics
    API_REQUESTS.labels(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code
    ).inc()
    
    # Add response headers
    response.headers["X-Processing-Time"] = str(time.time() - start_time)
    response.headers["X-API-Version"] = "1.0.0"
    
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", 
                path=request.url.path,
                method=request.method,
                error=str(exc))
    
    return JSONResponse(
        status_code=500,
        content=APIError(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=datetime.now().isoformat(),
            request_id=str(int(time.time()))
        ).dict()
    )


# Dependency to get API instance
def get_api_instance():
    global predictor_instance
    if predictor_instance is None:
        config_path = os.getenv('CONFIG_PATH', 'config.yaml')
        predictor_instance = ProduceForecasterAPI(config_path)
    return predictor_instance


# API Routes
@app.get("/health", response_model=HealthCheck)
async def health_check(api: ProduceForecasterAPI = Depends(get_api_instance)):
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version=api.app_version,
        models_loaded=len(api.models),
        uptime_seconds=round(time.time() - app_start_time, 2)
    )


@app.get("/models")
async def list_models(api: ProduceForecasterAPI = Depends(get_api_instance)):
    """List available models"""
    return {
        "models": api.get_model_list(),
        "default_model": "best_model",
        "total_models": len(api.models)
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict_single(
    features: FeatureInput,
    model_name: str = "best_model",
    api: ProduceForecasterAPI = Depends(get_api_instance)
):
    """Make single price prediction"""
    return api.predict_single(features, model_name)


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(
    batch_input: BatchFeatureInput,
    api: ProduceForecasterAPI = Depends(get_api_instance)
):
    """Make batch price predictions"""
    return api.predict_batch(batch_input)


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload-models")
async def reload_models(
    background_tasks: BackgroundTasks,
    api: ProduceForecasterAPI = Depends(get_api_instance)
):
    """Reload models in background"""
    def reload_task():
        try:
            api.load_models()
            logger.info("Models reloaded successfully")
        except Exception as e:
            logger.error("Error reloading models", error=str(e))
    
    background_tasks.add_task(reload_task)
    return {"message": "Model reload initiated", "timestamp": datetime.now().isoformat()}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Chinese Produce Market Forecasting API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


# Example usage function
def create_sample_prediction():
    """Create sample prediction request"""
    sample_features = FeatureInput(
        Total_Quantity=150.5,
        Avg_Price=12.80,
        Transaction_Count=25,
        Price_Volatility=0.15,
        Min_Price=11.50,
        Max_Price=14.20,
        Month=6,
        DayOfWeek=2,
        IsWeekend=0,
        Category_Avg_Price=13.50,
        Item_Revenue_Share=0.08,
        Avg_Price_Lag_1=12.75,
        Avg_Price_Lag_7=12.60,
        Avg_Price_MA_7=12.70,
        Loss_Rate_Percent=8.5
    )
    return sample_features


if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8001)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the API
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_config=None
    )