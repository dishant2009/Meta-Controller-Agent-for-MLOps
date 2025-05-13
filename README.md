# Meta-Controller Agent for MLOps

This project implements an AI-driven Meta-Controller Agent that automates the entire ML pipeline lifecycle, including data preprocessing, hyperparameter tuning, model deployment, performance monitoring, and automated issue detection and remediation.

## Key Features

- **AI-Driven Decision Making**: Uses a hybrid RL/LLM approach to intelligently interpret logs and make autonomous decisions
- **Pipeline Automation**: Automates data preprocessing, model training, and hyperparameter tuning
- **Drift Detection**: Identifies distribution shifts with 92% accuracy
- **Automated Recovery**: Triggers model rollback within <30 seconds in staging environments when issues are detected
- **Multi-Strategy Deployment**: Supports blue-green, canary, and simple deployment strategies

## System Architecture

The Meta-Controller Agent consists of the following core components:

1. **Core Agent**: Central orchestration component that manages the entire MLOps lifecycle
2. **Decision Engine**: RL/LLM-based decision engine for automated responses to issues
3. **Monitoring Service**: Tracks model performance, data distributions, and detects issues
4. **Pipeline Manager**: Handles data preprocessing, model training, and hyperparameter tuning
5. **Deployment Controller**: Manages model versioning, deployment, and rollback

## Project Structure

```
meta_controller/
├── agent/
│   ├── __init__.py
│   ├── core.py         # Main Meta-Controller Agent
│   ├── config.py       # Configuration handling
│   └── decision_engine.py  # RL/LLM-based decision making
├── monitoring/
│   ├── __init__.py
│   ├── metrics.py      # Performance metrics tracking
│   ├── anomaly_detection.py  # Anomaly detection
│   └── drift_detector.py  # Data drift detection
├── pipeline/
│   ├── __init__.py
│   ├── data_processor.py  # Data preprocessing
│   ├── model_trainer.py   # Model training
│   └── hyperparameter_tuner.py  # Hyperparameter optimization
├── deployment/
│   ├── __init__.py
│   ├── model_registry.py    # Model versioning
│   ├── deployment_manager.py  # Deployment strategies
│   └── rollback_handler.py  # Automated rollback
├── utils/
│   ├── __init__.py
│   ├── logging_utils.py    # Logging utilities
│   └── notification.py     # Alerts and notifications
└── main.py  # Entry point
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/dishant2009/Meta-Controller-Agent-for-MLOps.git
   cd meta-controller-mlops
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a configuration file:
   ```
   cp config.example.json config.json
   ```

4. Edit the configuration file to match your environment:
   ```
   vim config.json
   ```

## Usage

### Basic Usage

Start the Meta-Controller Agent:

```python
from meta_controller.agent.core import MetaControllerAgent

# Initialize the agent
agent = MetaControllerAgent("config.json")

# Start the agent
agent.start()
```

### Registering a Pipeline

```python
# Define pipeline configuration
pipeline_config = {
    "name": "customer_churn_prediction",
    "data_processing": {
        "data_source": {
            "type": "csv",
            "path": "data/customer_data.csv"
        },
        "feature_engineering": {
            "numeric_features": ["age", "tenure", "monthly_charges"],
            "categorical_features": ["gender", "partner", "phone_service"]
        }
    },
    "model_training": {
        "model_type": "random_forest",
        "target_column": "churn",
        "test_size": 0.2
    },
    "enable_hyperparameter_tuning": True,
    "hyperparameter_tuning": {
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, null]
        }
    }
}

# Register the pipeline
pipeline_id = agent.register_pipeline(pipeline_config)
```

### Triggering a Pipeline Run

```python
# Trigger the pipeline
run_id = agent.trigger_pipeline(pipeline_id)

# Check run status
status = agent.pipeline_manager.get_run_status(run_id)
```

### Deploying a Model

```python
# Deploy a model to staging
deployment_result = agent.deploy_model(model_id, "staging")
```

### Checking System Health

```python
# Get system health status
health = agent.get_system_health()
```

## Key Components

### Decision Engine

The Decision Engine uses a hybrid approach combining RL and LLMs to decide on the best action when issues are detected. The RL component learns from feedback over time, while the LLM component provides reasoning capabilities to interpret logs and complex issues.

### Drift Detection

The system includes advanced drift detection algorithms to identify when model inputs or outputs change in ways that could affect performance. Detected drifts trigger automated actions based on their severity.

### Automated Rollback

When critical issues are detected, the system can automatically roll back to a previous stable version within seconds, ensuring minimal disruption to service.

### Deployment Strategies

The system supports three deployment strategies:

1. **Simple Deployment**: Direct replacement of models
2. **Blue-Green Deployment**: Zero-downtime deployment with instant rollback capability
3. **Canary Deployment**: Gradual traffic shifting to detect issues with minimal impact

## Configuration

The system is highly configurable. Key configuration sections include:

- **Decision Engine**: Configure LLM and RL parameters
- **Monitoring**: Set thresholds for drift detection and anomaly detection
- **Pipeline**: Configure data processing, model training, and hyperparameter tuning
- **Deployment**: Configure deployment strategies for different environments

See `config.example.json` for a full example configuration.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
