"""
Implementation of the metrics tracker for model performance metrics and SLAs.
"""

import logging
from typing import Dict, List
from datetime import datetime, timedelta

# Setup logging
logger = logging.getLogger("meta_controller.metrics_tracker")

class MetricsTracker:
    """Tracker for model performance metrics and SLAs."""
    
    def __init__(self, config: Dict):
        """Initialize the metrics tracker."""
        self.config = config
        self.health_status = "INITIALIZING"
        self.model_metrics = {}
        self.sla_definitions = config.get("sla_definitions", {})
        
        logger.info("Metrics Tracker initialized")
    
    def start(self):
        """Start the metrics tracker."""
        self.health_status = "HEALTHY"
        logger.info("Metrics Tracker started")
    
    def get_health(self) -> str:
        """Get the health status of the metrics tracker."""
        return self.health_status
    
    def register_model(self, model_id: str, metadata: Dict):
        """Register a model for metrics tracking."""
        self.model_metrics[model_id] = {
            "baseline_metrics": metadata.get("performance_metrics", {}),
            "current_metrics": {},
            "historical_metrics": [],
            "sla_violations": [],
            "registered_at": datetime.now().isoformat()
        }
        
        logger.info(f"Model {model_id} registered for metrics tracking")
    
    def update_metrics(self, model_id: str, metrics: Dict):
        """
        Update metrics for a model.
        
        Args:
            model_id: ID of the model
            metrics: Current metrics for the model
        """
        if model_id not in self.model_metrics:
            logger.warning(f"Model {model_id} not registered for metrics tracking")
            return
        
        # Update current metrics
        self.model_metrics[model_id]["current_metrics"] = metrics
        
        # Add to historical metrics
        self.model_metrics[model_id]["historical_metrics"].append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        })
        
        # Limit historical metrics size
        max_history = self.config.get("max_history_size", 1000)
        if len(self.model_metrics[model_id]["historical_metrics"]) > max_history:
            self.model_metrics[model_id]["historical_metrics"] = \
                self.model_metrics[model_id]["historical_metrics"][-max_history:]
        
        # Check for SLA violations
        self._check_sla_violations(model_id, metrics)
    
    def _check_sla_violations(self, model_id: str, metrics: Dict):
        """Check if current metrics violate any SLAs."""
        if model_id not in self.model_metrics:
            return
        
        violations = []
        
        for sla_name, sla_def in self.sla_definitions.items():
            metric_name = sla_def.get("metric")
            threshold = sla_def.get("threshold")
            comparison = sla_def.get("comparison", "less_than")
            
            if metric_name in metrics and threshold is not None:
                current_value = metrics[metric_name]
                
                violated = False
                if comparison == "less_than" and current_value >= threshold:
                    violated = True
                elif comparison == "greater_than" and current_value <= threshold:
                    violated = True
                elif comparison == "equal" and current_value != threshold:
                    violated = True
                
                if violated:
                    violation = {
                        "timestamp": datetime.now().isoformat(),
                        "sla_name": sla_name,
                        "metric_name": metric_name,
                        "threshold": threshold,
                        "current_value": current_value,
                        "comparison": comparison
                    }
                    
                    violations.append(violation)
                    self.model_metrics[model_id]["sla_violations"].append(violation)
                    
                    logger.warning(f"SLA violation for model {model_id}: {sla_name}")
    
    def get_metrics(self) -> Dict:
        """Get current metrics for all models."""
        metrics = {
            "models_count": len(self.model_metrics),
            "models_with_sla_violations": 0,
            "total_sla_violations": 0
        }
        
        # Count models with recent SLA violations
        for model_id, model_data in self.model_metrics.items():
            # Check for violations in the last 24 hours
            recent_violations = [
                v for v in model_data.get("sla_violations", [])
                if datetime.fromisoformat(v["timestamp"]) > datetime.now() - timedelta(hours=24)
            ]
            
            if recent_violations:
                metrics["models_with_sla_violations"] += 1
                metrics["total_sla_violations"] += len(recent_violations)
        
        return metrics
    
    def detect_issues(self, metrics: Dict) -> List[Dict]:
        """Detect metrics-related issues based on current metrics."""
        issues = []
        
        # Check each model for SLA violations
        for model_id, model_data in self.model_metrics.items():
            # Check for violations in the last hour
            recent_violations = [
                v for v in model_data.get("sla_violations", [])
                if datetime.fromisoformat(v["timestamp"]) > datetime.now() - timedelta(hours=1)
            ]
            
            if recent_violations:
                # Group violations by SLA
                sla_groups = {}
                for violation in recent_violations:
                    sla_name = violation["sla_name"]
                    if sla_name not in sla_groups:
                        sla_groups[sla_name] = []
                    sla_groups[sla_name].append(violation)
                
                # Create an issue for each violated SLA
                for sla_name, violations in sla_groups.items():
                    # Determine severity based on violation count and magnitude
                    severity = "low"
                    if len(violations) > 5:
                        severity = "high"
                    elif len(violations) > 2:
                        severity = "medium"
                    
                    # Get the most recent violation
                    latest = max(violations, key=lambda v: v["timestamp"])
                    
                    # Create issue
                    issue = {
                        "id": f"sla_{model_id}_{sla_name}_{int(datetime.now().timestamp())}",
                        "type": "sla_violation",
                        "severity": severity,
                        "model_id": model_id,
                        "description": f"SLA violation for {sla_name} in model {model_id}",
                        "details": {
                            "sla_name": sla_name,
                            "metric_name": latest["metric_name"],
                            "threshold": latest["threshold"],
                            "current_value": latest["current_value"],
                            "violation_count": len(violations)
                        },
                        "detected_at": datetime.now().isoformat()
                    }
                    
                    issues.append(issue)
        
        return issues
    
    def investigate(self, issue: Dict) -> Dict:
        """
        Investigate a metrics-related issue in detail.
        
        Args:
            issue: Issue to investigate
            
        Returns:
            Dict: Investigation results
        """
        model_id = issue.get("model_id")
        if not model_id or model_id not in self.model_metrics:
            return {"error": "Model not found"}
        
        sla_name = issue.get("details", {}).get("sla_name")
        metric_name = issue.get("details", {}).get("metric_name")
        
        if not sla_name or not metric_name:
            return {"error": "Missing SLA or metric information"}
        
        # Analyze historical data for the metric
        history = self.model_metrics[model_id]["historical_metrics"]
        
        # Extract values for the specific metric
        values = []
        timestamps = []
        
        for entry in history:
            if metric_name in entry["metrics"]:
                values.append(entry["metrics"][metric_name])
                timestamps.append(entry["timestamp"])
        
        if not values:
            return {"error": "No historical data available for analysis"}
        
        # Calculate statistics
        mean_value = sum(values) / len(values)
        sorted_values = sorted(values)
        median_value = sorted_values[len(sorted_values) // 2]
        
        # Analyze trend
        trend = "stable"
        if len(values) > 5:
            recent_values = values[-5:]
            if all(recent_values[i] > recent_values[i-1] for i in range(1, len(recent_values))):
                trend = "increasing"
            elif all(recent_values[i] < recent_values[i-1] for i in range(1, len(recent_values))):
                trend = "decreasing"
        
        # Generate recommendations
        recommendations = []
        
        if trend == "increasing" and issue.get("details", {}).get("comparison") == "less_than":
            recommendations.append("Investigate resource constraints")
            recommendations.append("Check for increased traffic or data volume")
        elif trend == "decreasing" and issue.get("details", {}).get("comparison") == "greater_than":
            recommendations.append("Check for data quality issues")
            recommendations.append("Verify model features for degradation")
        
        recommendations.append("Consider adjusting SLA thresholds if consistently violated")
        
        return {
            "metric_history": {
                "mean": mean_value,
                "median": median_value,
                "trend": trend,
                "sample_size": len(values)
            },
            "insights": [
                f"Metric has been {trend} recently",
                f"Current value is {issue.get('details', {}).get('current_value')} vs threshold {issue.get('details', {}).get('threshold')}"
            ],
            "recommendations": recommendations
        }
