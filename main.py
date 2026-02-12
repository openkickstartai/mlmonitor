#!/usr/bin/env python3
"""
MLMonitor - Real-time ML Model Monitoring System
Main FastAPI application entry point
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
from datetime import datetime

from core.monitor import ModelMonitor
from core.database import DatabaseManager
from core.metrics import MetricsCollector

app = FastAPI(title="MLMonitor", description="Real-time ML Model Monitoring System", version="1.0.0")

# Initialize components
db_manager = DatabaseManager()
metrics_collector = MetricsCollector()
model_monitor = ModelMonitor(db_manager, metrics_collector)

class ModelPrediction(BaseModel):
    model_id: str
    prediction: float
    actual: Optional[float] = None
    features: Dict[str, float]
    timestamp: Optional[datetime] = None

class AlertThreshold(BaseModel):
    model_id: str
    metric_name: str
    threshold: float
    operator: str  # 'gt', 'lt', 'eq'

@app.on_event("startup")
async def startup_event():
    """Initialize database and start monitoring"""
    await db_manager.initialize()
    await model_monitor.start_monitoring()

@app.post("/models/{model_id}/predictions")
async def log_prediction(model_id: str, prediction: ModelPrediction, background_tasks: BackgroundTasks):
    """Log a model prediction for monitoring"""
    try:
        prediction.model_id = model_id
        if not prediction.timestamp:
            prediction.timestamp = datetime.utcnow()
        
        background_tasks.add_task(model_monitor.process_prediction, prediction)
        return {"status": "logged", "model_id": model_id, "timestamp": prediction.timestamp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_id}/metrics")
async def get_model_metrics(model_id: str, hours: int = 24):
    """Get model performance metrics"""
    try:
        metrics = await model_monitor.get_metrics(model_id, hours)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_id}/drift")
async def check_drift(model_id: str):
    """Check for data drift in model"""
    try:
        drift_status = await model_monitor.check_drift(model_id)
        return drift_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_id}/thresholds")
async def set_alert_threshold(model_id: str, threshold: AlertThreshold):
    """Set alert threshold for model metrics"""
    try:
        threshold.model_id = model_id
        await model_monitor.set_threshold(threshold)
        return {"status": "threshold_set", "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List all monitored models"""
    try:
        models = await model_monitor.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)