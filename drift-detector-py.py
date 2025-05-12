"""
Implementation of the data drift detector.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# Setup logging
logger = logging.getLogger("meta_controller.drift_detector")

class DriftDetector:
    """Detector for data and concept drift in model inputs and outputs."""
    
    def __init__(self, config: Dict):
        """Initialize the drift detector."""
        self.config = config
        self.detection_threshold = config.get("detection_threshold", 0.5)
        self.window_size = config.get("window_size", 1000)
        self.health_status = "INITIALIZING"
        self.model_baselines = {}
        self.current_distributions = {}
        
        # Select drift detection algorithm
        self.algorithm = config.get("algorithm", "ks_test")
        logger.info(f"Drift Detector initialized with algorithm: {self.algorithm}")
    
    def start(self):
        """Start the drift detector."""
        self.health_status = "HEALTHY"
        logger.info("Drift Detector started")
    
    def get_health(self) -> str:
        """Get the health status of the drift detector."""
        return self.health_status
    
    def register_model(self, model_id: str, metadata: Dict):
        """Register a model for drift detection."""
        if "feature_distributions" not in metadata:
            logger.warning(f"No feature distributions provided for model {model_id}")
            return
        
        self.model_baselines[model_id] = {
            "feature_distributions": metadata.get("feature_distributions", {}),
            "output_distribution": metadata.get("output_distribution", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        # Initialize current distributions
        self.current_distributions[model_id] = {
            "feature_distributions": {},
            "output_distribution": {},
            "last_updated": datetime.now().isoformat()
        }
        
        logger.info(f"Model {model_id} registered for drift detection")
    
    def update_distributions(self, model_id: str, feature_data: Dict, output_data: Dict):
        """Update current distributions with new data."""
        if model_id not in self.model_baselines:
            logger.warning(f"Model {model_id} not registered for drift detection")
            return
        
        # Update feature distributions
        for feature_name, values in feature_data.items():
            if feature_name not in self.current_distributions[model_id]["feature_distributions"]:
                self.current_distributions[model_id]["feature_distributions"][feature_name] = []
            
            # Add new values, keeping only the last window_size values
            self.current_distributions[model_id]["feature_distributions"][feature_name].extend(values)
            self.current_distributions[model_id]["feature_distributions"][feature_name] = \
                self.current_distributions[model_id]["feature_distributions"][feature_name][-self.window_size:]
        
        # Update output distribution
        if "values" in output_data:
            if "values" not in self.current_distributions[model_id]["output_distribution"]:
                self.current_distributions[model_id]["output_distribution"]["values"] = []
            
            self.current_distributions[model_id]["output_distribution"]["values"].extend(output_data["values"])
            self.current_distributions[model_id]["output_distribution"]["values"] = \
                self.current_distributions[model_id]["output_distribution"]["values"][-self.window_size:]
        
        # Update timestamp
        self.current_distributions[model_id]["last_updated"] = datetime.now().isoformat()
    
    def detect_drift(self, model_id: str) -> Dict:
        """
        Detect data and concept drift for a model.
        
        Args:
            model_id: ID of the model to check
            
        Returns:
            Dict: Drift detection results
        """
        if model_id not in self.model_baselines or model_id not in self.current_distributions:
            logger.warning(f"Model {model_id} not registered for drift detection")
            return {"drift_detected": False, "reason": "Model not registered"}
        
        # Check for feature drift
        feature_drift = {}
        for feature_name, baseline_dist in self.model_baselines[model_id]["feature_distributions"].items():
            if feature_name in self.current_distributions[model_id]["feature_distributions"]:
                current_dist = self.current_distributions[model_id]["feature_distributions"][feature_name]
                
                if len(current_dist) < 100:  # Need enough samples
                    continue
                
                # Calculate drift score based on selected algorithm
                drift_score = self._calculate_drift_score(baseline_dist, current_dist)
                
                # Check if drift exceeds threshold
                if drift_score > self.detection_threshold:
                    feature_drift[feature_name] = {
                        "drift_score": drift_score,
                        "threshold": self.detection_threshold
                    }
        
        # Check for output drift
        output_drift = None
        if "values" in self.model_baselines[model_id]["output_distribution"] and \
           "values" in self.current_distributions[model_id]["output_distribution"]:
            
            baseline_output = self.model_baselines[model_id]["output_distribution"]["values"]
            current_output = self.current_distributions[model_id]["output_distribution"]["values"]
            
            if len(current_output) >= 100:  # Need enough samples
                output_drift_score = self._calculate_drift_score(baseline_output, current_output)
                
                if output_drift_score > self.detection_threshold:
                    output_drift = {
                        "drift_score": output_drift_score,
                        "threshold": self.detection_threshold
                    }
        
        # Determine overall drift
        drift_detected = len(feature_drift) > 0 or output_drift is not None
        
        return {
            "drift_detected": drift_detected,
            "feature_drift": feature_drift,
            "output_drift": output_drift,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_drift_score(self, baseline_dist, current_dist) -> float:
        """
        Calculate drift score between baseline and current distributions.
        
        Args:
            baseline_dist: Baseline distribution
            current_dist: Current distribution
            
        Returns:
            float: Drift score between 0 and 1
        """
        # In a real implementation, this would use proper statistical tests
        # For this example, we'll use a simple approach for illustration
        
        if self.algorithm == "ks_test":
            # Simulate Kolmogorov-Smirnov test
            # In practice, this would use scipy.stats.ks_2samp
            random_score = np.random.uniform(0.1, 0.9)  # Simulate for example
            return random_score
            
        elif self.algorithm == "js_divergence":
            # Simulate Jensen-Shannon divergence
            # In practice, this would calculate actual JS divergence
            random_score = np.random.uniform(0.1, 0.7)  # Simulate for example
            return random_score
            
        else:
            # Default to a random score for simulation
            return np.random.uniform(0.1, 0.5)
    
    def get_metrics(self) -> Dict:
        """Get current drift metrics for all models."""
        metrics = {
            "models_with_drift": 0,
            "average_drift_score": 0.0,
            "max_drift_score": 0.0
        }
        
        total_drift_score = 0.0
        models_count = 0
        
        for model_id in self.model_baselines.keys():
            drift_result = self.detect_drift(model_id)
            
            if drift_result["drift_detected"]:
                metrics["models_with_drift"] += 1
            
            # Calculate average and max drift scores
            drift_scores = []
            
            for feature in drift_result.get("feature_drift", {}).values():
                drift_scores.append(feature.get("drift_score", 0.0))
            
            if drift_result.get("output_drift"):
                drift_scores.append(drift_result["output_drift"].get("drift_score", 0.0))
            
            if drift_scores:
                avg_score = sum(drift_scores) / len(drift_scores)
                total_drift_score += avg_score
                models_count += 1
                
                if max(drift_scores) > metrics["max_drift_score"]:
                    metrics["max_drift_score"] = max(drift_scores)
        
        if models_count > 0:
            metrics["average_drift_score"] = total_drift_score / models_count
        
        return metrics
    
    def detect_issues(self, metrics: Dict) -> List[Dict]:
        """Detect drift-related issues based on current metrics."""
        issues = []
        
        # Check each model for drift
        for model_id in self.model_baselines.keys():
            drift_result = self.detect_drift(model_id)
            
            if drift_result["drift_detected"]:
                # Create an issue for each drifting model
                drift_features = list(drift_result.get("feature_drift", {}).keys())
                max_drift_score = 0.0
                
                # Find max drift score
                for feature in drift_result.get("feature_drift", {}).values():
                    if feature.get("drift_score", 0.0) > max_drift_score:
                        max_drift_score = feature.get("drift_score", 0.0)
                
                if drift_result.get("output_drift"):
                    output_score = drift_result["output_drift"].get("drift_score", 0.0)
                    if output_score > max_drift_score:
                        max_drift_score = output_score
                
                # Determine severity based on drift score
                severity = "low"
                if max_drift_score > 0.8:
                    severity = "high"
                elif max_drift_score > 0.6:
                    severity = "medium"
                
                # Create issue
                issue = {
                    "id": f"drift_{model_id}_{int(time.time())}",
                    "type": "data_drift",
                    "severity": severity,
                    "model_id": model_id,
                    "description": f"Data drift detected in model {model_id}",
                    "details": {
                        "drifting_features": drift_features,
                        "output_drift": drift_result.get("output_drift") is not None,
                        "max_drift_score": max_drift_score
                    },
                    "detected_at": datetime.now().isoformat()
                }
                
                issues.append(issue)
        
        return issues
    
    def investigate(self, issue: Dict) -> Dict:
        """
        Investigate a drift issue in detail.
        
        Args:
            issue: Issue to investigate
            
        Returns:
            Dict: Investigation results
        """
        model_id = issue.get("model_id")
        if not model_id or model_id not in self.model_baselines:
            return {"error": "Model not found"}
        
        # Get drift details
        drift_result = self.detect_drift(model_id)
        
        # Identify the most affected features
        feature_impacts = []
        for feature_name, drift_info in drift_result.get("feature_drift", {}).items():
            feature_impacts.append({
                "feature": feature_name,
                "drift_score": drift_info.get("drift_score", 0.0),
                "impact": self._estimate_feature_impact(model_id, feature_name)
            })
        
        # Sort by impact
        feature_impacts.sort(key=lambda x: x["impact"], reverse=True)
        
        # Generate recommendations
        recommendations = []
        if feature_impacts:
            recommendations.append(f"Focus on features: {', '.join([f['feature'] for f in feature_impacts[:3]])}")
            
            if any(f["drift_score"] > 0.7 for f in feature_impacts):
                recommendations.append("Consider rolling back model due to high drift")
            else:
                recommendations.append("Retrain model with recent data")
        
        if drift_result.get("output_drift"):
            recommendations.append("Investigate potential concept drift")
        
        return {
            "feature_impacts": feature_impacts,
            "output_drift": drift_result.get("output_drift"),
            "drift_started": self._estimate_drift_start(model_id),
            "recommendations": recommendations
        }
    
    def _estimate_feature_impact(self, model_id: str, feature_name: str) -> float:
        """Estimate the impact of a feature on model performance."""
        # In a real implementation, this would use feature importance metrics
        # For this example, we'll return a random score
        return np.random.uniform(0.1, 1.0)
    
    def _estimate_drift_start(self, model_id: str) -> str:
        """Estimate when drift started for a model."""
        # In a real implementation, this would analyze historical data
        # For this example, we'll return a recent timestamp
        hours_ago = np.random.randint(1, 48)
        drift_start = datetime.now() - timedelta(hours=hours_ago)
        return drift_start.isoformat()
