"""
Sample code demonstrating how to use the Meta-Controller Agent for MLOps system.
"""

import time
import logging
import json
from meta_controller.agent.core import MetaControllerAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("mlops_demo")

def main():
    """Demonstrate using the Meta-Controller Agent for MLOps."""
    
    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # Initialize the Meta-Controller Agent
    logger.info("Initializing Meta-Controller Agent")
    agent = MetaControllerAgent("config.json")
    
    # Start the agent
    logger.info("Starting agent")
    agent.start()
    
    # Example 1: Register an ML pipeline
    logger.info("Registering a pipeline")
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
            },
            "validation": {
                "enabled": True,
                "rules": [
                    {"type": "not_null", "columns": ["customer_id", "tenure"]},
                    {"type": "range", "column": "monthly_charges", "min": 0, "max": 1000}
                ]
            }
        },
        "model_training": {
            "model_type": "random_forest",
            "target_column": "churn",
            "test_size": 0.2,
            "model_params": {
                "n_estimators": 100,
                "max_depth": 10
            },
            "cross_validation": {
                "enabled": True,
                "cv_folds": 5
            }
        },
        "enable_hyperparameter_tuning": True,
        "hyperparameter_tuning": {
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 20, null],
                "min_samples_split": [2, 5, 10]
            }
        },
        "model_evaluation": {
            "metrics": ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        }
    }
    
    pipeline_id = agent.register_pipeline(pipeline_config)
    logger.info(f"Pipeline registered with ID: {pipeline_id}")
    
    # Example 2: Trigger the pipeline
    logger.info("Triggering pipeline execution")
    run_id = agent.trigger_pipeline(pipeline_id)
    logger.info(f"Pipeline triggered with run ID: {run_id}")
    
    # Wait for the pipeline to complete
    logger.info("Waiting for pipeline to complete...")
    while True:
        status = agent.pipeline_manager.get_run_status(run_id)
        if status["status"] in ["COMPLETED", "FAILED"]:
            break
        logger.info(f"Pipeline status: {status['status']}")
        time.sleep(5)
    
    logger.info(f"Pipeline completed with status: {status['status']}")
    
    # If the pipeline was successful, deploy the model
    if status["status"] == "COMPLETED":
        # Example 3: Register and deploy the model
        model_path = status["artifacts"].get("trained_model")
        model_metadata = {
            "name": "customer_churn_model",
            "version": "1.0.0",
            "framework": "sklearn",
            "performance_metrics": status["metrics"].get("evaluation", {}),
            "feature_distributions": {
                "age": {"mean": 35.4, "std": 12.3},
                "tenure": {"mean": 34.6, "std": 24.8},
                "monthly_charges": {"mean": 64.8, "std": 30.1}
            },
            "output_distribution": {
                "values": [0, 1],
                "distribution": [0.73, 0.27]
            }
        }
        
        # Register model
        logger.info("Registering model")
        model_id = agent.deployment_controller.register_model(model_path, model_metadata)
        logger.info(f"Model registered with ID: {model_id}")
        
        # Deploy to staging
        logger.info("Deploying model to staging")
        deployment_result = agent.deploy_model(model_id, "staging")
        logger.info(f"Deployment result: {deployment_result}")
        
        # Simulate traffic and detection of an issue
        logger.info("Simulating model operation in staging...")
        time.sleep(10)
        
        # Example 4: Simulate an issue detection
        logger.info("Simulating detection of a data drift issue")
        issue = {
            "id": f"simulated_drift_{int(time.time())}",
            "type": "data_drift",
            "severity": "high",
            "model_id": model_id,
            "pipeline_id": pipeline_id,
            "description": "High severity data drift detected in feature distributions",
            "details": {
                "drifting_features": ["monthly_charges", "tenure"],
                "output_drift": True,
                "max_drift_score": 0.85
            },
            "detected_at": time.time()
        }
        
        # Let the decision engine decide what to do
        logger.info("Getting decision from decision engine")
        metrics = {
            "drift_score": 0.85,
            "performance_drop": 12.5,
            "error_rate": 0.05,
            "latency": 120
        }
        logs = []
        
        action = agent.decision_engine.decide_action(issue, metrics, logs)
        logger.info(f"Decision engine recommended action: {action}")
        
        # Execute the action
        logger.info("Executing recommended action")
        agent._execute_action(action, issue)
        
        # Check status after action
        logger.info("Checking system health after action")
        health = agent.get_system_health()
        logger.info(f"System health: {health}")
        
    else:
        logger.error(f"Pipeline failed with errors: {status.get('errors', [])}")
    
    logger.info("Demo completed!")

if __name__ == "__main__":
    main()
