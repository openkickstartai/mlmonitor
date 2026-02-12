"""
Alert management system for MLMonitor
Handles different types of alerts and notification delivery
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self):
        self.alert_handlers = {
            'drift': self._handle_drift_alert,
            'performance': self._handle_performance_alert,
            'threshold': self._handle_threshold_alert
        }
        logger.info("Alert manager initialized")
    
    async def send_drift_alert(self, model_id: str, drift_results: Dict):
        """Send alert for data drift detection"""
        try:
            drifted_features = [f for f, r in drift_results.items() if r.get('drift_detected', False)]
            
            if drifted_features:
                message = f"Data drift detected in model {model_id} for features: {', '.join(drifted_features)}"
                
                alert_data = {
                    'model_id': model_id,
                    'alert_type': 'drift',
                    'message': message,
                    'severity': 'high',
                    'details': drift_results,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                await self._process_alert(alert_data)
                logger.warning(f"Drift alert sent for model {model_id}")
        except Exception as e:
            logger.error(f"Error sending drift alert: {e}")
    
    async def send_performance_alert(self, model_id: str, error: float):
        """Send alert for performance degradation"""
        try:
            message = f"High prediction error detected in model {model_id}: {error:.4f}"
            
            alert_data = {
                'model_id': model_id,
                'alert_type': 'performance',
                'message': message,
                'severity': 'medium',
                'details': {'error': error},
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await self._process_alert(alert_data)
            logger.warning(f"Performance alert sent for model {model_id}")
        except Exception as e:
            logger.error(f"Error sending performance alert: {e}")
    
    async def send_threshold_alert(self, model_id: str, threshold: Dict, current_value: float):
        """Send alert for threshold violation"""
        try:
            message = f"Threshold violated for model {model_id}: {threshold['metric_name']} = {current_value:.4f} (threshold: {threshold['operator']} {threshold['threshold']})"
            
            alert_data = {
                'model_id': model_id,
                'alert_type': 'threshold',
                'message': message,
                'severity': 'medium',
                'details': {
                    'metric_name': threshold['metric_name'],
                    'current_value': current_value,
                    'threshold': threshold['threshold'],
                    'operator': threshold['operator']
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await self._process_alert(alert_data)
            logger.warning(f"Threshold alert sent for model {model_id}")
        except Exception as e:
            logger.error(f"Error sending threshold alert: {e}")
    
    async def _process_alert(self, alert_data: Dict):
        """Process and route alert to appropriate handlers"""
        try:
            alert_type = alert_data['alert_type']
            
            # Log alert
            logger.info(f"Processing {alert_type} alert: {alert_data['message']}")
            
            # Call specific handler
            if alert_type in self.alert_handlers:
                await self.alert_handlers[alert_type](alert_data)
            
            # Store alert in database (would need database reference)
            # await self.db.store_alert(alert_data['model_id'], alert_type, alert_data['message'], alert_data['severity'])
            
            # Send notifications (email, Slack, etc.)
            await self._send_notifications(alert_data)
            
        except Exception as e:
            logger.error(f"Error processing alert: {e}")
    
    async def _handle_drift_alert(self, alert_data: Dict):
        """Handle drift-specific alert logic"""
        try:
            model_id = alert_data['model_id']
            drift_details = alert_data['details']
            
            # Log detailed drift information
            logger.info(f"Drift alert details for model {model_id}:")
            for feature, details in drift_details.items():
                if details.get('drift_detected'):
                    logger.info(f"  {feature}: KS={details['ks_statistic']:.4f}, p-value={details['p_value']:.6f}")
            
            # Could implement model retraining trigger here
            # await self._trigger_model_retraining(model_id)
            
        except Exception as e:
            logger.error(f"Error handling drift alert: {e}")
    
    async def _handle_performance_alert(self, alert_data: Dict):
        """Handle performance-specific alert logic"""
        try:
            model_id = alert_data['model_id']
            error = alert_data['details']['error']
            
            logger.info(f"Performance degradation detected for model {model_id}: error={error:.4f}")
            
            # Could implement automatic model rollback or investigation
            # await self._investigate_performance_issue(model_id)
            
        except Exception as e:
            logger.error(f"Error handling performance alert: {e}")
    
    async def _handle_threshold_alert(self, alert_data: Dict):
        """Handle threshold-specific alert logic"""
        try:
            model_id = alert_data['model_id']
            details = alert_data['details']
            
            logger.info(f"Threshold violation for model {model_id}: {details['metric_name']} = {details['current_value']:.4f}")
            
            # Could implement escalation logic based on severity
            # await self._escalate_if_needed(alert_data)
            
        except Exception as e:
            logger.error(f"Error handling threshold alert: {e}")
    
    async def _send_notifications(self, alert_data: Dict):
        """Send notifications via configured channels"""
        try:
            # This would integrate with actual notification services
            # For now, just log the notification
            
            severity = alert_data['severity']
            message = alert_data['message']
            
            if severity == 'high':
                logger.critical(f"HIGH SEVERITY ALERT: {message}")
                # await self._send_email_alert(alert_data)
                # await self._send_slack_alert(alert_data)
            elif severity == 'medium':
                logger.warning(f"MEDIUM SEVERITY ALERT: {message}")
                # await self._send_slack_alert(alert_data)
            else:
                logger.info(f"LOW SEVERITY ALERT: {message}")
            
            # Could also send to external monitoring systems
            # await self._send_to_pagerduty(alert_data)
            # await self._send_to_datadog(alert_data)
            
        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
    
    def get_alert_summary(self, alerts: List[Dict]) -> Dict:
        """Generate summary of alerts"""
        try:
            if not alerts:
                return {'total': 0, 'by_type': {}, 'by_severity': {}}
            
            summary = {
                'total': len(alerts),
                'by_type': {},
                'by_severity': {},
                'recent_count': 0
            }
            
            # Count by type and severity
            for alert in alerts:
                alert_type = alert.get('alert_type', 'unknown')
                severity = alert.get('severity', 'unknown')
                
                summary['by_type'][alert_type] = summary['by_type'].get(alert_type, 0) + 1
                summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            return summary
        except Exception as e:
            logger.error(f"Error generating alert summary: {e}")
            return {'total': 0, 'by_type': {}, 'by_severity': {}}