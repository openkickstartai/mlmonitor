"""
Database management for MLMonitor
Handles SQLite operations for storing predictions, metrics, and thresholds
"""

import sqlite3
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "mlmonitor.db"):
        self.db_path = db_path
        self.connection = None
    
    async def initialize(self):
        """Initialize database and create tables"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            
            await self._create_tables()
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    async def _create_tables(self):
        """Create necessary database tables"""
        cursor = self.connection.cursor()
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                prediction REAL NOT NULL,
                actual REAL,
                features TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Thresholds table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS thresholds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                threshold REAL NOT NULL,
                operator TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model_id, metric_name)
            )
        """)
        
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                resolved BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_model_time ON predictions(model_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_model_time ON alerts(model_id, timestamp)")
        
        self.connection.commit()
        logger.info("Database tables created successfully")
    
    async def store_prediction(self, prediction):
        """Store a prediction in the database"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO predictions (model_id, prediction, actual, features, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                prediction.model_id,
                prediction.prediction,
                prediction.actual,
                json.dumps(prediction.features),
                prediction.timestamp
            ))
            self.connection.commit()
            logger.debug(f"Stored prediction for model {prediction.model_id}")
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            raise
    
    async def get_predictions(self, model_id: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get predictions for a model within a time range"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM predictions 
                WHERE model_id = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            """, (model_id, start_time, end_time))
            
            rows = cursor.fetchall()
            predictions = []
            
            for row in rows:
                prediction = {
                    'id': row['id'],
                    'model_id': row['model_id'],
                    'prediction': row['prediction'],
                    'actual': row['actual'],
                    'features': json.loads(row['features']),
                    'timestamp': row['timestamp']
                }
                predictions.append(prediction)
            
            logger.debug(f"Retrieved {len(predictions)} predictions for model {model_id}")
            return predictions
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            raise
    
    async def store_threshold(self, threshold):
        """Store or update an alert threshold"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO thresholds (model_id, metric_name, threshold, operator)
                VALUES (?, ?, ?, ?)
            """, (
                threshold.model_id,
                threshold.metric_name,
                threshold.threshold,
                threshold.operator
            ))
            self.connection.commit()
            logger.debug(f"Stored threshold for model {threshold.model_id}")
        except Exception as e:
            logger.error(f"Error storing threshold: {e}")
            raise
    
    async def get_thresholds(self, model_id: str) -> List[Dict]:
        """Get all thresholds for a model"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM thresholds WHERE model_id = ?
            """, (model_id,))
            
            rows = cursor.fetchall()
            thresholds = [dict(row) for row in rows]
            
            logger.debug(f"Retrieved {len(thresholds)} thresholds for model {model_id}")
            return thresholds
        except Exception as e:
            logger.error(f"Error getting thresholds: {e}")
            raise
    
    async def get_model_list(self) -> List[str]:
        """Get list of all monitored models"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT DISTINCT model_id FROM predictions")
            
            rows = cursor.fetchall()
            models = [row['model_id'] for row in rows]
            
            logger.debug(f"Retrieved {len(models)} models")
            return models
        except Exception as e:
            logger.error(f"Error getting model list: {e}")
            raise
    
    async def store_alert(self, model_id: str, alert_type: str, message: str, severity: str = "medium"):
        """Store an alert in the database"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO alerts (model_id, alert_type, message, severity, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (model_id, alert_type, message, severity, datetime.utcnow()))
            self.connection.commit()
            logger.info(f"Stored alert for model {model_id}: {message}")
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
            raise
    
    async def get_recent_alerts(self, model_id: Optional[str] = None, hours: int = 24) -> List[Dict]:
        """Get recent alerts"""
        try:
            cursor = self.connection.cursor()
            
            if model_id:
                cursor.execute("""
                    SELECT * FROM alerts 
                    WHERE model_id = ? AND timestamp > datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                """.format(hours), (model_id,))
            else:
                cursor.execute("""
                    SELECT * FROM alerts 
                    WHERE timestamp > datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                """.format(hours))
            
            rows = cursor.fetchall()
            alerts = [dict(row) for row in rows]
            
            logger.debug(f"Retrieved {len(alerts)} recent alerts")
            return alerts
        except Exception as e:
            logger.error(f"Error getting recent alerts: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")