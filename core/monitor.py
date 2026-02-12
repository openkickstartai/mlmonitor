"""
Core monitoring logic for ML models
Handles drift detection, performance tracking, and alerting
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from scipy import stats
import logging

from .database import DatabaseManager
from .metrics import MetricsCollector
from .alerts import AlertManager

logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, db_manager: DatabaseManager, metrics_collector: MetricsCollector):
        self.db = db_manager
        self.metrics = metrics_collector
        self.alert_manager = AlertManager()
        self.monitoring_active = False
        self.drift_threshold = 0.05  # p-value threshold for drift detection
    
    async def start_monitoring(self):
        """Start background monitoring tasks"""
        self.monitoring_active = True
        asyncio.create_task(self._monitoring_loop())
        logger.info("Model monitoring started")
    
    async def process_prediction(self, prediction):
        """Process a new prediction and update metrics"""
        try:
            # Store prediction in database
            await self.db.store_prediction(prediction)
            
            # Update real-time metrics
            await self.metrics.update_metrics(prediction)
            
            # Check for immediate alerts if actual value provided
            if prediction.actual is not None:
                await self._check_performance_alerts(prediction)
            
            logger.debug(f"Processed prediction for model {prediction.model_id}")
        except Exception as e:
            logger.error(f"Error processing prediction: {e}")
            raise
    
    async def get_metrics(self, model_id: str, hours: int = 24) -> Dict:
        """Get performance metrics for a model"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            predictions = await self.db.get_predictions(model_id, start_time, end_time)
            
            if not predictions:
                return {"model_id": model_id, "metrics": {}, "message": "No data available"}
            
            # Calculate metrics
            metrics = await self.metrics.calculate_metrics(predictions)
            
            return {
                "model_id": model_id,
                "time_range": {"start": start_time, "end": end_time},
                "metrics": metrics,
                "prediction_count": len(predictions)
            }
        except Exception as e:
            logger.error(f"Error getting metrics for model {model_id}: {e}")
            raise
    
    async def check_drift(self, model_id: str) -> Dict:
        """Check for data drift using statistical tests"""
        try:
            # Get recent data (last 7 days) vs baseline (previous 30 days)
            now = datetime.utcnow()
            recent_start = now - timedelta(days=7)
            baseline_start = now - timedelta(days=37)
            baseline_end = now - timedelta(days=7)
            
            recent_data = await self.db.get_predictions(model_id, recent_start, now)
            baseline_data = await self.db.get_predictions(model_id, baseline_start, baseline_end)
            
            if len(recent_data) < 30 or len(baseline_data) < 30:
                return {
                    "model_id": model_id,
                    "drift_detected": False,
                    "message": "Insufficient data for drift detection"
                }
            
            # Perform Kolmogorov-Smirnov test for each feature
            drift_results = {}
            overall_drift = False
            
            # Get feature names from first prediction
            if recent_data and baseline_data:
                feature_names = list(recent_data[0].get('features', {}).keys())
                
                for feature in feature_names:
                    recent_values = [p['features'].get(feature, 0) for p in recent_data if 'features' in p]
                    baseline_values = [p['features'].get(feature, 0) for p in baseline_data if 'features' in p]
                    
                    if recent_values and baseline_values:
                        ks_stat, p_value = stats.ks_2samp(baseline_values, recent_values)
                        drift_detected = p_value < self.drift_threshold
                        
                        drift_results[feature] = {
                            "ks_statistic": ks_stat,
                            "p_value": p_value,
                            "drift_detected": drift_detected
                        }
                        
                        if drift_detected:
                            overall_drift = True
            
            if overall_drift:
                await self.alert_manager.send_drift_alert(model_id, drift_results)
            
            return {
                "model_id": model_id,
                "drift_detected": overall_drift,
                "feature_drift": drift_results,
                "threshold": self.drift_threshold,
                "timestamp": now
            }
        except Exception as e:
            logger.error(f"Error checking drift for model {model_id}: {e}")
            raise
    
    async def set_threshold(self, threshold):
        """Set alert threshold for model metrics"""
        try:
            await self.db.store_threshold(threshold)
            logger.info(f"Threshold set for model {threshold.model_id}: {threshold.metric_name} {threshold.operator} {threshold.threshold}")
        except Exception as e:
            logger.error(f"Error setting threshold: {e}")
            raise
    
    async def list_models(self) -> List[str]:
        """List all monitored models"""
        try:
            return await self.db.get_model_list()
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                models = await self.list_models()
                for model_id in models:
                    await self._check_model_health(model_id)
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_model_health(self, model_id: str):
        """Check overall model health and trigger alerts if needed"""
        try:
            metrics = await self.get_metrics(model_id, hours=1)
            if metrics.get('metrics'):
                await self._evaluate_thresholds(model_id, metrics['metrics'])
        except Exception as e:
            logger.error(f"Error checking health for model {model_id}: {e}")
    
    async def _check_performance_alerts(self, prediction):
        """Check if prediction triggers any performance alerts"""
        try:
            error = abs(prediction.prediction - prediction.actual)
            if error > 0.1:  # Example threshold
                await self.alert_manager.send_performance_alert(prediction.model_id, error)
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    async def _evaluate_thresholds(self, model_id: str, metrics: Dict):
        """Evaluate metrics against configured thresholds"""
        try:
            thresholds = await self.db.get_thresholds(model_id)
            for threshold in thresholds:
                metric_value = metrics.get(threshold['metric_name'])
                if metric_value is not None:
                    if self._threshold_exceeded(metric_value, threshold):
                        await self.alert_manager.send_threshold_alert(model_id, threshold, metric_value)
        except Exception as e:
            logger.error(f"Error evaluating thresholds for model {model_id}: {e}")
    
    def _threshold_exceeded(self, value: float, threshold: Dict) -> bool:
        """Check if metric value exceeds threshold"""
        op = threshold['operator']
        thresh = threshold['threshold']
        
        if op == 'gt':
            return value > thresh
        elif op == 'lt':
            return value < thresh
        elif op == 'eq':
            return abs(value - thresh) < 0.001
        return False