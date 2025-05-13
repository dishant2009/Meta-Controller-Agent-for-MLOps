"""
Implementation of the pipeline manager for data processing and model training.
"""

import logging
import time
import threading
from typing import Dict, List, Optional
from datetime import datetime

from meta_controller.pipeline.data_processor import DataProcessor
from meta_controller.pipeline.model_trainer import ModelTrainer
from meta_controller.pipeline.hyperparameter_tuner import HyperparameterTuner

# Setup logging
logger = logging.getLogger("meta_controller.pipeline_manager")

class PipelineManager:
    """
    Manager for data preprocessing pipelines and model training.
    """
    def __init__(self, config: Dict):
        """
        Initialize the Pipeline Manager.
        
        Args:
            config: Configuration for the pipeline manager
        """
        self.config = config
        self.health_status = "INITIALIZING"
        self.registered_pipelines = {}
        self.pipeline_runs = {}
        self.active_runs = set()
        
        self.data_processor = DataProcessor(config.get("data_processor", {}))
        self.model_trainer = ModelTrainer(config.get("model_trainer", {}))
        self.hyperparameter_tuner = HyperparameterTuner(config.get("hyperparameter_tuner", {}))
        
        logger.info("Pipeline Manager initialized")
    
    def start(self):
        """Start the pipeline manager."""
        self.data_processor.start()
        self.model_trainer.start()
        self.hyperparameter_tuner.start()
        
        self.health_status = "HEALTHY"
        logger.info("Pipeline Manager started")
    
    def get_health(self) -> str:
        """Get the health status of the pipeline manager."""
        components_health = {
            "data_processor": self.data_processor.get_health(),
            "model_trainer": self.model_trainer.get_health(),
            "hyperparameter_tuner": self.hyperparameter_tuner.get_health()
        }
        
        if all(status == "HEALTHY" for status in components_health.values()):
            self.health_status = "HEALTHY"
        elif any(status == "CRITICAL" for status in components_health.values()):
            self.health_status = "CRITICAL"
        else:
            self.health_status = "DEGRADED"
            
        return self.health_status
    
    def register_pipeline(self, pipeline_config: Dict) -> str:
        """
        Register a new ML pipeline.
        
        Args:
            pipeline_config: Configuration for the pipeline
            
        Returns:
            str: Unique ID for the registered pipeline
        """
        pipeline_id = f"pipeline_{int(datetime.now().timestamp())}_{len(self.registered_pipelines)}"
        
        self.registered_pipelines[pipeline_id] = {
            "config": pipeline_config,
            "status": "REGISTERED",
            "created_at": datetime.now().isoformat(),
            "last_run": None,
            "run_count": 0
        }
        
        logger.info(f"Pipeline {pipeline_id} registered with config: {pipeline_config}")
        return pipeline_id
    
    def get_pipeline_status(self, pipeline_id: str) -> Dict:
        """
        Get the current status of a pipeline.
        
        Args:
            pipeline_id: ID of the pipeline
            
        Returns:
            Dict: Pipeline status information
        """
        if pipeline_id not in self.registered_pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        return {
            "status": self.registered_pipelines[pipeline_id]["status"],
            "last_run": self.registered_pipelines[pipeline_id]["last_run"],
            "run_count": self.registered_pipelines[pipeline_id]["run_count"],
            "active_run": pipeline_id in self.active_runs
        }
    
    def trigger_pipeline(self, pipeline_id: str, override_params: Optional[Dict] = None) -> str:
        """
        Trigger a pipeline run.
        
        Args:
            pipeline_id: ID of the pipeline to run
            override_params: Optional parameters to override in the pipeline config
            
        Returns:
            str: Run ID for the triggered pipeline
        """
        if pipeline_id not in self.registered_pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        # Create a run configuration by merging the pipeline config with overrides
        run_config = dict(self.registered_pipelines[pipeline_id]["config"])
        
        if override_params:
            # Deep merge the override params
            self._deep_merge(run_config, override_params)
        
        # Generate a run ID
        run_id = f"run_{pipeline_id}_{int(datetime.now().timestamp())}"
        
        # Create run record
        self.pipeline_runs[run_id] = {
            "pipeline_id": pipeline_id,
            "config": run_config,
            "status": "PENDING",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "steps": [],
            "artifacts": {},
            "metrics": {},
            "errors": []
        }
        
        # Update pipeline record
        self.registered_pipelines[pipeline_id]["status"] = "RUNNING"
        self.registered_pipelines[pipeline_id]["last_run"] = run_id
        self.registered_pipelines[pipeline_id]["run_count"] += 1
        
        # Add to active runs
        self.active_runs.add(pipeline_id)
        
        # Start the pipeline in a background thread
        threading.Thread(
            target=self._run_pipeline, 
            args=(run_id, run_config),
            daemon=True
        ).start()
        
        logger.info(f"Pipeline {pipeline_id} triggered with run ID: {run_id}")
        return run_id
    
    def _deep_merge(self, base_dict: Dict, override_dict: Dict):
        """
        Deep merge two dictionaries.
        
        Args:
            base_dict: Base dictionary to merge into
            override_dict: Dictionary with values to override
        """
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _run_pipeline(self, run_id: str, config: Dict):
        """
        Run a pipeline with the given configuration.
        
        Args:
            run_id: ID of the run
            config: Configuration for the run
        """
        try:
            pipeline_id = self.pipeline_runs[run_id]["pipeline_id"]
            logger.info(f"Starting pipeline run {run_id} for pipeline {pipeline_id}")
            
            # Update run status
            self.pipeline_runs[run_id]["status"] = "RUNNING"
            self.pipeline_runs[run_id]["started_at"] = datetime.now().isoformat()
            
            # 1. Data preprocessing
            self._update_run_step(run_id, "data_preprocessing", "RUNNING")
            data_result = self.data_processor.process_data(config.get("data_processing", {}))
            
            if data_result.get("status") != "SUCCESS":
                raise Exception(f"Data preprocessing failed: {data_result.get('error')}")
                
            self._update_run_step(run_id, "data_preprocessing", "COMPLETED", data_result)
            self.pipeline_runs[run_id]["artifacts"]["processed_data"] = data_result.get("output_path")
            
            # 2. Hyperparameter tuning (if enabled)
            if config.get("enable_hyperparameter_tuning", False):
                self._update_run_step(run_id, "hyperparameter_tuning", "RUNNING")
                
                tuning_config = config.get("hyperparameter_tuning", {})
                tuning_config["data_path"] = data_result.get("output_path")
                
                tuning_result = self.hyperparameter_tuner.tune_hyperparameters(tuning_config)
                
                if tuning_result.get("status") != "SUCCESS":
                    raise Exception(f"Hyperparameter tuning failed: {tuning_result.get('error')}")
                    
                self._update_run_step(run_id, "hyperparameter_tuning", "COMPLETED", tuning_result)
                self.pipeline_runs[run_id]["artifacts"]["best_hyperparameters"] = tuning_result.get("best_hyperparameters")
                
                # Update training config with best hyperparameters
                training_config = config.get("model_training", {})
                training_config.update(tuning_result.get("best_hyperparameters", {}))
            else:
                # Use the provided training config
                training_config = config.get("model_training", {})
            
            # 3. Model training
            self._update_run_step(run_id, "model_training", "RUNNING")
            
            # Update training config with data path
            training_config["data_path"] = data_result.get("output_path")
            
            training_result = self.model_trainer.train_model(training_config)
            
            if training_result.get("status") != "SUCCESS":
                raise Exception(f"Model training failed: {training_result.get('error')}")
                
            self._update_run_step(run_id, "model_training", "COMPLETED", training_result)
            self.pipeline_runs[run_id]["artifacts"]["trained_model"] = training_result.get("model_path")
            self.pipeline_runs[run_id]["metrics"]["training"] = training_result.get("metrics", {})
            
            # 4. Model evaluation
            self._update_run_step(run_id, "model_evaluation", "RUNNING")
            
            eval_config = config.get("model_evaluation", {})
            eval_config["model_path"] = training_result.get("model_path")
            eval_config["data_path"] = data_result.get("output_path")
            
            eval_result = self.model_trainer.evaluate_model(eval_config)
            
            if eval_result.get("status") != "SUCCESS":
                raise Exception(f"Model evaluation failed: {eval_result.get('error')}")
                
            self._update_run_step(run_id, "model_evaluation", "COMPLETED", eval_result)
            self.pipeline_runs[run_id]["metrics"]["evaluation"] = eval_result.get("metrics", {})
            
            # Update run status to completed
            self.pipeline_runs[run_id]["status"] = "COMPLETED"
            self.pipeline_runs[run_id]["completed_at"] = datetime.now().isoformat()
            
            # Update pipeline status
            self.registered_pipelines[pipeline_id]["status"] = "IDLE"
            
            logger.info(f"Pipeline run {run_id} completed successfully")
            
        except Exception as e:
            # Handle errors
            error_message = str(e)
            logger.error(f"Pipeline run {run_id} failed: {error_message}")
            
            # Update run status
            self.pipeline_runs[run_id]["status"] = "FAILED"
            self.pipeline_runs[run_id]["completed_at"] = datetime.now().isoformat()
            self.pipeline_runs[run_id]["errors"].append({
                "message": error_message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update pipeline status
            pipeline_id = self.pipeline_runs[run_id]["pipeline_id"]
            self.registered_pipelines[pipeline_id]["status"] = "FAILED"
            
        finally:
            # Remove from active runs
            pipeline_id = self.pipeline_runs[run_id]["pipeline_id"]
            self.active_runs.discard(pipeline_id)
    
    def _update_run_step(self, run_id: str, step_name: str, status: str, result: Optional[Dict] = None):
        """
        Update the status of a pipeline run step.
        
        Args:
            run_id: ID of the run
            step_name: Name of the step
            status: New status for the step
            result: Optional result data for the step
        """
        # Find the step if it exists
        step_exists = False
        for step in self.pipeline_runs[run_id]["steps"]:
            if step["name"] == step_name:
                step["status"] = status
                step["updated_at"] = datetime.now().isoformat()
                if status == "COMPLETED" and result:
                    step["result"] = result
                step_exists = True
                break
        
        # Create the step if it doesn't exist
        if not step_exists:
            step = {
                "name": step_name,
                "status": status,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            if status == "COMPLETED" and result:
                step["result"] = result
                
            self.pipeline_runs[run_id]["steps"].append(step)
    
    def get_run_status(self, run_id: str) -> Dict:
        """
        Get the status of a pipeline run.
        
        Args:
            run_id: ID of the run
            
        Returns:
            Dict: Run status information
        """
        if run_id not in self.pipeline_runs:
            raise ValueError(f"Run {run_id} not found")
        
        return {
            "run_id": run_id,
            "pipeline_id": self.pipeline_runs[run_id]["pipeline_id"],
            "status": self.pipeline_runs[run_id]["status"],
            "created_at": self.pipeline_runs[run_id]["created_at"],
            "started_at": self.pipeline_runs[run_id]["started_at"],
            "completed_at": self.pipeline_runs[run_id]["completed_at"],
            "steps": [
                {
                    "name": step["name"],
                    "status": step["status"],
                    "updated_at": step["updated_at"]
                }
                for step in self.pipeline_runs[run_id]["steps"]
            ],
            "metrics": self.pipeline_runs[run_id]["metrics"],
            "artifacts": self.pipeline_runs[run_id]["artifacts"],
            "error_count": len(self.pipeline_runs[run_id]["errors"])
        }
    
    def apply_pipeline_fix(self, pipeline_id: str, fix_type: str, fix_params: Dict) -> Dict:
        """
        Apply a fix to a pipeline.
        
        Args:
            pipeline_id: ID of the pipeline to fix
            fix_type: Type of fix to apply
            fix_params: Parameters for the fix
            
        Returns:
            Dict: Result of the fix operation
        """
        if pipeline_id not in self.registered_pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        logger.info(f"Applying {fix_type} fix to pipeline {pipeline_id}")
        
        pipeline_config = self.registered_pipelines[pipeline_id]["config"]
        
        if fix_type == "data_validation":
            # Add data validation step to pipeline
            if "data_processing" not in pipeline_config:
                pipeline_config["data_processing"] = {}
            
            pipeline_config["data_processing"]["validation"] = {
                "enabled": True,
                "rules": fix_params.get("rules", []),
                "strict": fix_params.get("strict", True)
            }
            
            result = {
                "status": "SUCCESS",
                "message": "Data validation added to pipeline",
                "details": "Added validation step to data processing"
            }
            
        elif fix_type == "error_handling":
            # Add error handling to pipeline
            for section in ["data_processing", "model_training", "model_evaluation"]:
                if section in pipeline_config:
                    if "error_handling" not in pipeline_config[section]:
                        pipeline_config[section]["error_handling"] = {}
                    
                    pipeline_config[section]["error_handling"].update({
                        "retry_count": fix_params.get("retry_count", 3),
                        "timeout": fix_params.get("timeout", 300),
                        "fallback_strategy": fix_params.get("fallback_strategy", "abort")
                    })
            
            result = {
                "status": "SUCCESS",
                "message": "Error handling improved across pipeline",
                "details": "Updated retry count and timeout values"
            }
            
        elif fix_type == "performance_optimization":
            # Optimize pipeline for performance
            for section in ["data_processing", "model_training"]:
                if section in pipeline_config:
                    if "performance" not in pipeline_config[section]:
                        pipeline_config[section]["performance"] = {}
                    
                    pipeline_config[section]["performance"].update({
                        "batch_size": fix_params.get("batch_size", 64),
                        "num_workers": fix_params.get("num_workers", 4),
                        "prefetch_factor": fix_params.get("prefetch_factor", 2)
                    })
            
            result = {
                "status": "SUCCESS",
                "message": "Performance settings optimized",
                "details": "Updated batch size and worker settings"
            }
            
        else:
            result = {
                "status": "ERROR",
                "message": f"Unknown fix type: {fix_type}",
                "details": "Supported fix types: data_validation, error_handling, performance_optimization"
            }
        
        # Update pipeline record with the modified config
        self.registered_pipelines[pipeline_id]["config"] = pipeline_config
        
        return result
