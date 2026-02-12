"""
Metrics collection and calculation for ML model monitoring
Integrates with Prometheus for metrics export
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self, prometheus_port: int = 8001):
        self.prometheus_port = prometheus_port
        self._setup_prometheus_metrics()
        self._start_prometheus_server()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        self.prediction_counter = Counter(
            'ml_predictions_total',
            'Total number of predictions',
            ['model_id']
        )
        
        self.prediction_error = Histogram(
            'ml_prediction_error',
            'Prediction error distribution',
            ['model_id'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.model_accuracy = Gauge(
            'ml_model_accuracy',
            'Current model accuracy',
            ['model_id']
        )
        
        self.drift_score = Gauge(
            'ml_drift_score',
            'Data drift score',
            ['model_id', 'feature']
        )
        
        self.alert_counter = Counter(
            'ml_alerts_total',
            'Total number of alerts',
            ['model_id', 'alert_type']
        )
        
        logger.info("Prometheus metrics initialized")
    
    def _start_prometheus_server(self):
        """Start Prometheus metrics server"""
        try:
            start_http_server(self.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    async def update_metrics(self, prediction):
        """Update real-time metrics with new prediction"""
        try:
            model_id = prediction.model_id
            
            # Increment prediction counter
            self.prediction_counter.labels(model_id=model_id).inc()
            
            # Update error metrics if actual value is available
            if prediction.actual is not None:
                error = abs(prediction.prediction - prediction.actual)
                self.prediction_error.labels(model_id=model_id).observe(error)
            
            logger.debug(f"Updated metrics for model {model_id}")
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def calculate_metrics(self, predictions: List[Dict]) -> Dict:
        """Calculate comprehensive metrics from prediction data"""
        try:
            if not predictions:
                return {}
            
            # Filter predictions with actual values for accuracy calculations
            labeled_predictions = [p for p in predictions if p.get('actual') is not None]
            
            metrics = {
                'total_predictions': len(predictions),
                'labeled_predictions': len(labeled_predictions)
            }
            
            if labeled_predictions:
                # Calculate accuracy metrics
                errors = [abs(p['prediction'] - p['actual']) for p in labeled_predictions]
                
                metrics.update({
                    'mean_absolute_error': np.mean(errors),
                    'median_absolute_error': np.median(errors),
                    'max_error': np.max(errors),
                    'min_error': np.min(errors),
                    'std_error': np.std(errors),
                    'rmse': np.sqrt(np.mean([e**2 for e in errors]))
                })
                
                # Calculate accuracy (within 10% threshold)
                accurate_predictions = sum(1 for e in errors if e <= 0.1)
                metrics['accuracy_10pct'] = accurate_predictions / len(labeled_predictions)
                
                # Calculate R-squared
                actual_values = [p['actual'] for p in labeled_predictions]
                predicted_values = [p['prediction'] for p in labeled_predictions]
                
                if len(set(actual_values)) > 1:  # Avoid division by zero
                    ss_res = sum((a - p)**2 for a, p in zip(actual_values, predicted_values))
                    ss_tot = sum((a - np.mean(actual_values))**2 for a in actual_values)
                    metrics['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                else:
                    metrics['r_squared'] = 0
            
            # Calculate feature statistics
            if predictions and 'features' in predictions[0]:
                feature_stats = self._calculate_feature_stats(predictions)
                metrics['feature_stats'] = feature_stats
            
            # Calculate prediction distribution
            pred_values = [p['prediction'] for p in predictions]
            metrics['prediction_stats'] = {
                'mean': np.mean(pred_values),
                'median': np.median(pred_values),
                'std': np.std(pred_values),
                'min': np.min(pred_values),
                'max': np.max(pred_values),
                'percentile_25': np.percentile(pred_values, 25),
                'percentile_75': np.percentile(pred_values, 75)
            }
            
            # Update Prometheus gauges
            if labeled_predictions:
                model_id = predictions[0]['model_id']
                self.model_accuracy.labels(model_id=model_id).set(metrics.get('accuracy_10pct', 0))
            
            logger.debug(f"Calculated {len(metrics)} metrics")
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def _calculate_feature_stats(self, predictions: List[Dict]) -> Dict:
        """Calculate statistics for each feature"""
        try:
            feature_stats = {}
            
            # Get all feature names
            all_features = set()
            for p in predictions:
                if 'features' in p and p['features']:
                    all_features.update(p['features'].keys())
            
            # Calculate stats for each feature
            for feature in all_features:
                values = []
                for p in predictions:
                    if 'features' in p and feature in p['features']:
                        values.append(p['features'][feature])
                
                if values:
                    feature_stats[feature] = {
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
            
            return feature_stats
        except Exception as e:
            logger.error(f"Error calculating feature stats: {e}")
            return {}
    
    def record_alert(self, model_id: str, alert_type: str):
        """Record an alert in Prometheus metrics"""
        try:
            self.alert_counter.labels(model_id=model_id, alert_type=alert_type).inc()
            logger.debug(f"Recorded alert: {alert_type} for model {model_id}")
        except Exception as e:
            logger.error(f"Error recording alert: {e}")
    
    def update_drift_score(self, model_id: str, feature: str, score: float):
        """Update drift score for a specific feature"""
        try:
            self.drift_score.labels(model_id=model_id, feature=feature).set(score)
            logger.debug(f"Updated drift score for {model_id}.{feature}: {score}")
        except Exception as e:
            logger.error(f"Error updating drift score: {e}")