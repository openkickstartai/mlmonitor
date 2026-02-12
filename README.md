# MLMonitor

Real-time machine learning model monitoring system that tracks model performance, detects drift, and provides alerts for production ML models.

## Features

- **Real-time Performance Monitoring**: Track model accuracy, error rates, and prediction quality
- **Data Drift Detection**: Statistical tests to detect changes in input data distribution
- **Automated Alerting**: Configurable thresholds with multi-channel notifications
- **Metrics Visualization**: Prometheus integration for monitoring dashboards
- **Model Comparison**: Compare performance across different models and time periods
- **API Integration**: RESTful API for easy integration with ML pipelines
- **Historical Analysis**: Store and analyze historical performance data

## Tech Stack

- **FastAPI**: High-performance web framework for the API
- **SQLite**: Lightweight database for storing predictions and metrics
- **Prometheus**: Metrics collection and monitoring
- **Python**: Core implementation with scientific computing libraries

## Quick Start

### Installation

```bash
pip install fastapi uvicorn sqlite3 prometheus-client scipy numpy
```

### Running the Service

```bash
python main.py
```

The API will be available at `http://localhost:8000` and Prometheus metrics at `http://localhost:8001`.

### API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

## Usage Examples

### Log a Prediction

```bash
curl -X POST "http://localhost:8000/models/my-model/predictions" \
  -H "Content-Type: application/json" \
  -d '{
    "prediction": 0.85,
    "actual": 0.82,
    "features": {
      "feature1": 1.2,
      "feature2": 0.8,
      "feature3": 2.1
    }
  }'
```

### Get Model Metrics

```bash
curl "http://localhost:8000/models/my-model/metrics?hours=24"
```

### Check for Data Drift

```bash
curl "http://localhost:8000/models/my-model/drift"
```

### Set Alert Threshold

```bash
curl -X POST "http://localhost:8000/models/my-model/thresholds" \
  -H "Content-Type: application/json" \
  -d '{
    "metric_name": "mean_absolute_error",
    "threshold": 0.1,
    "operator": "gt"
  }'
```

## Architecture

### Core Components

- **main.py**: FastAPI application and API endpoints
- **core/monitor.py**: Main monitoring logic and drift detection
- **core/database.py**: SQLite database operations
- **core/metrics.py**: Metrics calculation and Prometheus integration
- **core/alerts.py**: Alert management and notification system

### Data Flow

1. **Prediction Logging**: Models send predictions via API
2. **Real-time Processing**: Metrics updated and stored in database
3. **Background Monitoring**: Periodic drift detection and health checks
4. **Alert Generation**: Threshold violations trigger notifications
5. **Metrics Export**: Prometheus scrapes metrics for visualization

## Monitoring Capabilities

### Performance Metrics

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R-squared coefficient
- Accuracy within thresholds
- Prediction distribution statistics

### Drift Detection

- Kolmogorov-Smirnov test for feature drift
- Configurable p-value thresholds
- Per-feature drift analysis
- Historical baseline comparison

### Alert Types

- **Performance Alerts**: High prediction errors
- **Drift Alerts**: Statistical distribution changes
- **Threshold Alerts**: Custom metric violations

## Configuration

### Environment Variables

- `DB_PATH`: SQLite database file path (default: `mlmonitor.db`)
- `PROMETHEUS_PORT`: Metrics server port (default: `8001`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

### Alert Thresholds

Configure custom thresholds via the API:

- `mean_absolute_error`
- `rmse`
- `accuracy_10pct`
- `r_squared`

## Integration

### With ML Pipelines

```python
import requests

def log_prediction(model_id, prediction, actual, features):
    response = requests.post(
        f"http://mlmonitor:8000/models/{model_id}/predictions",
        json={
            "prediction": prediction,
            "actual": actual,
            "features": features
        }
    )
    return response.json()
```

### With Monitoring Stack

- **Grafana**: Create dashboards using Prometheus metrics
- **AlertManager**: Route alerts to Slack, email, PagerDuty
- **Kubernetes**: Deploy with health checks and auto-scaling

## Development

### Project Structure

```
mlmonitor/
├── main.py              # FastAPI application
├── core/
│   ├── __init__.py
│   ├── monitor.py       # Core monitoring logic
│   ├── database.py      # Database operations
│   ├── metrics.py       # Metrics and Prometheus
│   └── alerts.py        # Alert management
├── tests/               # Test suite
└── README.md
```

### Running Tests

```bash
python -m pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
