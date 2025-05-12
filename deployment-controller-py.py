"""
Implementation of the deployment controller that manages model deployment and versioning.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from meta_controller.deployment.model_registry import ModelRegistry
from meta_controller.deployment.deployment_manager import DeploymentManager
from meta_controller.deployment.rollback_handler import RollbackHandler

# Setup logging
logger = logging.getLogger("meta_controller.deployment_controller")

class DeploymentController:
    """
    Controller for model deployment and versioning.
    """
    def __init__(self, config: Dict):
        """
        Initialize the Deployment Controller.
        
        Args:
            config: Configuration for the deployment controller
        """
        self.config = config
        self.health_status = "INITIALIZING"
        self.model_registry = ModelRegistry(config.get("model_registry", {}))
        self.deployment_manager = DeploymentManager(config.get("deployment_manager", {}))
        self.rollback_handler = RollbackHandler(config.get("rollback_handler", {}))
        
        self.environments = config.get("environments", ["dev", "staging", "production"])
        self.deployment_history = {}
        
        logger.info("Deployment Controller initialized")
    
    def start(self):
        """Start the deployment controller."""
        self.model_registry.start()
        self.deployment_manager.start()
        self.rollback_handler.start()
        
        self.health_status = "HEALTHY"
        logger.info("Deployment Controller started")
    
    def get_health(self) -> str:
        """Get the health status of the deployment controller."""
        components_health = {
            "model_registry": self.model_registry.get_health(),
            "deployment_manager": self.deployment_manager.get_health(),
            "rollback_handler": self.rollback_handler.get_health()
        }
        
        if all(status == "HEALTHY" for status in components_health.values()):
            self.health_status = "HEALTHY"
        elif any(status == "CRITICAL" for status in components_health.values()):
            self.health_status = "CRITICAL"
        else:
            self.health_status = "DEGRADED"
            
        return self.health_status
    
    def register_model(self, model_path: str, metadata: Dict) -> str:
        """
        Register a model in the model registry.
        
        Args:
            model_path: Path to the model file
            metadata: Metadata for the model
            
        Returns:
            str: Model ID of the registered model
        """
        return self.model_registry.register_model(model_path, metadata)
    
    def deploy_model(self, model_id: str, environment: str) -> Dict:
        """
        Deploy a model to the specified environment.
        
        Args:
            model_id: ID of the model to deploy
            environment: Target environment
            
        Returns:
            Dict: Deployment status
        """
        if environment not in self.environments:
            raise ValueError(f"Invalid environment: {environment}")
        
        # Get model details from registry
        model_details = self.model_registry.get_model_details(model_id)
        
        if not model_details:
            raise ValueError(f"Model {model_id} not found in registry")
        
        logger.info(f"Deploying model {model_id} to {environment}")
        
        # Deploy the model
        deployment_result = self.deployment_manager.deploy_model(
            model_id, 
            model_details.get("path"), 
            environment
        )
        
        if deployment_result.get("status") == "SUCCESS":
            # Record deployment in history
            if environment not in self.deployment_history:
                self.deployment_history[environment] = []
            
            self.deployment_history[environment].append({
                "model_id": model_id,
                "version": model_details.get("version"),
                "timestamp": datetime.now().isoformat(),
                "deployment_id": deployment_result.get("deployment_id")
            })
            
            # Update model registry
            self.model_registry.update_model_status(
                model_id, 
                f"DEPLOYED_{environment.upper()}"
            )
        
        return deployment_result
    
    def get_deployment_status(self, deployment_id: str) -> Dict:
        """
        Get the status of a deployment.
        
        Args:
            deployment_id: ID of the deployment
            
        Returns:
            Dict: Deployment status
        """
        return self.deployment_manager.get_deployment_status(deployment_id)
    
    def list_deployments(self, environment: Optional[str] = None) -> List[Dict]:
        """
        List all deployments in an environment.
        
        Args:
            environment: Optional environment to filter by
            
        Returns:
            List[Dict]: List of deployments
        """
        deployments = []
        
        if environment:
            if environment not in self.environments:
                raise ValueError(f"Invalid environment: {environment}")
            
            env_list = [environment]
        else:
            env_list = self.environments
        
        for env in env_list:
            env_deployments = self.deployment_history.get(env, [])
            
            for deployment in env_deployments:
                deployment_id = deployment.get("deployment_id")
                if deployment_id:
                    status = self.deployment_manager.get_deployment_status(deployment_id)
                    deployments.append({
                        "environment": env,
                        "model_id": deployment.get("model_id"),
                        "version": deployment.get("version"),
                        "timestamp": deployment.get("timestamp"),
                        "deployment_id": deployment_id,
                        "status": status.get("status", "UNKNOWN")
                    })
        
        return deployments
    
    def rollback_model(self, model_id: str, version: Optional[str] = None, environment: Optional[str] = None) -> Dict:
        """
        Rollback a model to a previous version.
        
        Args:
            model_id: ID of the model to rollback
            version: Optional specific version to rollback to
            environment: Optional specific environment to rollback in
            
        Returns:
            Dict: Rollback status
        """
        # Determine environments to rollback
        if environment:
            if environment not in self.environments:
                raise ValueError(f"Invalid environment: {environment}")
            
            env_list = [environment]
        else:
            # Default to staging only for safety
            env_list = ["staging"]
        
        # Get the current deployments for the model
        current_deployments = {}
        for env in env_list:
            if env in self.deployment_history:
                for deployment in reversed(self.deployment_history[env]):
                    if deployment.get("model_id") == model_id:
                        current_deployments[env] = deployment
                        break
        
        if not current_deployments:
            return {
                "status": "ERROR",
                "message": f"No deployments found for model {model_id} in specified environments"
            }
        
        # Determine the version to rollback to
        if version:
            # Use specified version
            target_version = version
        else:
            # Find the previous version
            prev_versions = self.model_registry.get_model_versions(model_id)
            
            if len(prev_versions) <= 1:
                return {
                    "status": "ERROR",
                    "message": f"No previous versions found for model {model_id}"
                }
            
            # Get the current version from the first environment
            current_version = current_deployments[env_list[0]].get("version")
            
            # Find the previous version
            prev_index = None
            for i, ver in enumerate(prev_versions):
                if ver.get("version") == current_version:
                    prev_index = i - 1
                    break
            
            if prev_index is None or prev_index < 0:
                return {
                    "status": "ERROR",
                    "message": f"Could not determine previous version for model {model_id}"
                }
            
            target_version = prev_versions[prev_index].get("version")
        
        # Get model details for target version
        target_model = self.model_registry.get_model_by_version(model_id, target_version)
        
        if not target_model:
            return {
                "status": "ERROR",
                "message": f"Target version {target_version} not found for model {model_id}"
            }
        
        # Perform rollback in each environment
        rollback_results = {}
        
        for env in env_list:
            logger.info(f"Rolling back model {model_id} to version {target_version} in {env}")
            
            rollback_result = self.rollback_handler.perform_rollback(
                model_id, 
                target_version, 
                target_model.get("path"),
                env
            )
            
            rollback_results[env] = rollback_result
            
            if rollback_result.get("status") == "SUCCESS":
                # Record new deployment in history
                if env not in self.deployment_history:
                    self.deployment_history[env] = []
                
                self.deployment_history[env].append({
                    "model_id": model_id,
                    "version": target_version,
                    "timestamp": datetime.now().isoformat(),
                    "deployment_id": rollback_result.get("deployment_id"),
                    "rollback": True
                })
        
        # Determine overall status
        all_success = all(result.get("status") == "SUCCESS" for result in rollback_results.values())
        any_success = any(result.get("status") == "SUCCESS" for result in rollback_results.values())
        
        if all_success:
            overall_status = "SUCCESS"
            message = f"Successfully rolled back model {model_id} to version {target_version} in all environments"
        elif any_success:
            overall_status = "PARTIAL_SUCCESS"
            message = f"Partially rolled back model {model_id} to version {target_version}"
        else:
            overall_status = "ERROR"
            message = f"Failed to rollback model {model_id} to version {target_version}"
        
        return {
            "status": overall_status,
            "message": message,
            "model_id": model_id,
            "target_version": target_version,
            "environment_results": rollback_results,
            "timestamp": datetime.now().isoformat()
        }
