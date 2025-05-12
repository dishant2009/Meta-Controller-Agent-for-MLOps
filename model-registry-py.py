"""
Implementation of the model registry for tracking and versioning ML models.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

# Setup logging
logger = logging.getLogger("meta_controller.model_registry")

class ModelRegistry:
    """Registry for tracking and versioning ML models."""
    
    def __init__(self, config: Dict):
        """Initialize the model registry."""
        self.config = config
        self.health_status = "INITIALIZING"
        self.storage_path = config.get("storage_path", "model_registry")
        self.models = {}
        
        logger.info("Model Registry initialized")
    
    def start(self):
        """Start the model registry."""
        self.health_status = "HEALTHY"
        logger.info("Model Registry started")
    
    def get_health(self) -> str:
        """Get the health status of the model registry."""
        return self.health_status
    
    def register_model(self, model_path: str, metadata: Dict) -> str:
        """
        Register a model in the registry.
        
        Args:
            model_path: Path to the model file
            metadata: Metadata for the model
            
        Returns:
            str: Model ID of the registered model
        """
        # Generate model ID if not provided
        model_id = metadata.get("model_id")
        if not model_id:
            model_id = f"model_{int(datetime.now().timestamp())}_{len(self.models)}"
        
        # Determine version
        if model_id in self.models:
            # Increment version for existing model
            versions = [int(v.get("version", "0").split("v")[1]) 
                      for v in self.models[model_id]]
            next_version = max(versions) + 1 if versions else 1
            version = f"v{next_version}"
        else:
            # Initial version for new model
            version = "v1"
            self.models[model_id] = []
        
        # Add additional metadata
        model_info = {
            "model_id": model_id,
            "version": version,
            "path": model_path,
            "registered_at": datetime.now().isoformat(),
            "status": "REGISTERED",
            "metadata": metadata
        }
        
        # Add to registry
        self.models[model_id].append(model_info)
        
        logger.info(f"Model {model_id} version {version} registered")
        
        return model_id
    
    def get_model_details(self, model_id: str) -> Optional[Dict]:
        """
        Get details of the latest version of a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Optional[Dict]: Model details or None if not found
        """
        if model_id not in self.models or not self.models[model_id]:
            return None
        
        # Return the latest version
        return max(self.models[model_id], 
                  key=lambda v: int(v.get("version", "v0").split("v")[1]))
    
    def get_model_versions(self, model_id: str) -> List[Dict]:
        """
        Get all versions of a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            List[Dict]: List of model versions
        """
        if model_id not in self.models:
            return []
        
        # Sort by version number
        return sorted(self.models[model_id], 
                     key=lambda v: int(v.get("version", "v0").split("v")[1]))
    
    def get_model_by_version(self, model_id: str, version: str) -> Optional[Dict]:
        """
        Get a specific version of a model.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            
        Returns:
            Optional[Dict]: Model details or None if not found
        """
        if model_id not in self.models:
            return None
        
        # Find the specified version
        for model_version in self.models[model_id]:
            if model_version.get("version") == version:
                return model_version
        
        return None
    
    def update_model_status(self, model_id: str, status: str, version: Optional[str] = None) -> bool:
        """
        Update the status of a model.
        
        Args:
            model_id: ID of the model
            status: New status
            version: Optional specific version to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        if model_id not in self.models:
            return False
        
        if version:
            # Update specific version
            for model_version in self.models[model_id]:
                if model_version.get("version") == version:
                    model_version["status"] = status
                    model_version["updated_at"] = datetime.now().isoformat()
                    return True
            
            return False
        else:
            # Update latest version
            latest = max(self.models[model_id], 
                        key=lambda v: int(v.get("version", "v0").split("v")[1]))
            latest["status"] = status
            latest["updated_at"] = datetime.now().isoformat()
            return True
