"""
Core implementation of the Meta-Controller Agent.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
import json
from datetime import datetime

from meta_controller.agent.decision_engine import DecisionEngine
from meta_controller.monitoring.monitoring_service import MonitoringService
from meta_controller.pipeline.pipeline_manager import PipelineManager
from meta_controller.deployment.deployment_controller import DeploymentController

# Setup logging
logger = logging.getLogger("meta_controller.core")

class MetaControllerAgent:
    """
    Main orchestration agent that manages the entire ML pipeline lifecycle.
    """
    def __init__(self, config_path: str):
        """
        Initialize the Meta-Controller Agent.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        logger.info("Initializing Meta-Controller Agent")
        
        # Initialize components
        self.decision_engine = DecisionEngine(self.config["decision_engine"])
        self.monitoring_service = MonitoringService(self.config["monitoring"])
        self.pipeline_manager = PipelineManager(self.config["pipeline"])
        self.deployment_controller = DeploymentController(self.config["deployment"])
        
        # Setup internal state
        self.active_pipelines = {}
        self.health_status = "INITIALIZING"
        self._setup_watchdog()
        
        logger.info("Meta-Controller Agent initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def _setup_watchdog(self):
        """Setup watchdog thread to monitor system health."""
        self.watchdog_thread = threading.Thread(
            target=self._watchdog_process, 
            daemon=True
        )
        self.watchdog_thread.start()
        
    def _watchdog_process(self):
        """Watchdog process to monitor system health."""
        while True:
            try:
                # Check all components health
                components_health = {
                    "decision_engine": self.decision_engine.get_health(),
                    "monitoring_service": self.monitoring_service.get_health(),
                    "pipeline_manager": self.pipeline_manager.get_health(),
                    "deployment_controller": self.deployment_controller.get_health()
                }
                
                # Update overall health status
                if all(status == "HEALTHY" for status in components_health.values()):
                    self.health_status = "HEALTHY"
                elif any(status == "CRITICAL" for status in components_health.values()):
                    self.health_status = "CRITICAL"
                else:
                    self.health_status = "DEGRADED"
                
                logger.debug(f"System health: {self.health_status}")
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Watchdog error: {str(e)}")
                self.health_status = "CRITICAL"
                time.sleep(60)  # Retry after 1 minute on error
    
    def start(self):
        """Start the Meta-Controller Agent."""
        logger.info("Starting Meta-Controller Agent")
        self.health_status = "STARTING"
        
        # Start all components
        self.monitoring_service.start()
        self.deployment_controller.start()
        self.pipeline_manager.start()
        self.decision_engine.start()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_process,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.health_status = "HEALTHY"
        logger.info("Meta-Controller Agent started successfully")
    
    def _monitoring_process(self):
        """Main monitoring process that periodically checks for issues."""
        while True:
            try:
                # Get current metrics and logs
                metrics = self.monitoring_service.get_current_metrics()
                logs = self.monitoring_service.get_recent_logs()
                
                # Check for issues
                issues = self.monitoring_service.detect_issues(metrics)
                
                if issues:
                    logger.info(f"Detected {len(issues)} issues")
                    for issue in issues:
                        # Let decision engine decide what to do
                        action = self.decision_engine.decide_action(issue, metrics, logs)
                        self._execute_action(action, issue)
                
                time.sleep(self.config.get("monitoring_interval", 10))
            except Exception as e:
                logger.error(f"Error in monitoring process: {str(e)}")
                time.sleep(30)  # Retry after delay
    
    def _execute_action(self, action: Dict, issue: Dict):
        """Execute an action decided by the decision engine."""
        action_type = action.get("type")
        logger.info(f"Executing action: {action_type} for issue: {issue.get('id')}")
        
        try:
            if action_type == "ROLLBACK":
                model_id = action.get("model_id")
                version = action.get("version")
                self.deployment_controller.rollback_model(model_id, version)
                
            elif action_type == "RETRAIN":
                pipeline_id = action.get("pipeline_id")
                config_updates = action.get("config_updates", {})
                self.pipeline_manager.trigger_pipeline(pipeline_id, config_updates)
                
            elif action_type == "INVESTIGATE":
                issue_id = issue.get("id")
                self.monitoring_service.deep_investigate(issue_id)
                
            elif action_type == "FIX_PIPELINE":
                pipeline_id = action.get("pipeline_id")
                fix_type = action.get("fix_type")
                fix_params = action.get("fix_params", {})
                self.pipeline_manager.apply_pipeline_fix(pipeline_id, fix_type, fix_params)
                
            else:
                logger.warning(f"Unknown action type: {action_type}")
                
            # Record the action in the issue log
            self.monitoring_service.record_action(issue.get("id"), action)
            
        except Exception as e:
            logger.error(f"Failed to execute action {action_type}: {str(e)}")
    
    def register_pipeline(self, pipeline_config: Dict) -> str:
        """
        Register a new ML pipeline with the system.
        
        Args:
            pipeline_config: Configuration for the pipeline
            
        Returns:
            str: Unique ID for the registered pipeline
        """
        pipeline_id = self.pipeline_manager.register_pipeline(pipeline_config)
        self.active_pipelines[pipeline_id] = {
            "status": "REGISTERED",
            "created_at": datetime.now().isoformat(),
            "config": pipeline_config
        }
        return pipeline_id
    
    def get_pipeline_status(self, pipeline_id: str) -> Dict:
        """Get the current status of a pipeline."""
        if pipeline_id not in self.active_pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        # Get latest status
        pipeline_status = self.pipeline_manager.get_pipeline_status(pipeline_id)
        self.active_pipelines[pipeline_id]["status"] = pipeline_status.get("status")
        
        return pipeline_status
    
    def trigger_pipeline(self, pipeline_id: str, override_params: Optional[Dict] = None) -> str:
        """
        Manually trigger a pipeline run.
        
        Args:
            pipeline_id: ID of the pipeline to trigger
            override_params: Optional parameters to override in the pipeline config
            
        Returns:
            str: Run ID for the triggered pipeline
        """
        if pipeline_id not in self.active_pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        run_id = self.pipeline_manager.trigger_pipeline(pipeline_id, override_params)
        return run_id
    
    def deploy_model(self, model_id: str, environment: str) -> Dict:
        """
        Deploy a trained model to the specified environment.
        
        Args:
            model_id: ID of the model to deploy
            environment: Target environment (e.g., "staging", "production")
            
        Returns:
            Dict: Deployment status
        """
        return self.deployment_controller.deploy_model(model_id, environment)
    
    def get_system_health(self) -> Dict:
        """Get the overall health status of the system."""
        return {
            "status": self.health_status,
            "components": {
                "decision_engine": self.decision_engine.get_health(),
                "monitoring_service": self.monitoring_service.get_health(),
                "pipeline_manager": self.pipeline_manager.get_health(),
                "deployment_controller": self.deployment_controller.get_health()
            },
            "active_pipelines": len(self.active_pipelines),
            "timestamp": datetime.now().isoformat()
        }
