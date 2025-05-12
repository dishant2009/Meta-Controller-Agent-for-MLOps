"""
Implementation of the RL/LLM-based decision engine that determines actions based on monitoring data.
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

# Setup logging
logger = logging.getLogger("meta_controller.decision_engine")

class DecisionEngine:
    """
    RL/LLM-based decision engine that determines actions based on monitoring data.
    """
    def __init__(self, config: Dict):
        """
        Initialize the Decision Engine.
        
        Args:
            config: Configuration for the decision engine
        """
        self.config = config
        self.model_type = config.get("model_type", "llm")  # "llm", "rl", or "hybrid"
        self.health_status = "INITIALIZING"
        self.last_action = None
        self.action_history = []
        
        # Initialize appropriate model based on config
        if self.model_type == "llm":
            self._init_llm_model()
        elif self.model_type == "rl":
            self._init_rl_model()
        elif self.model_type == "hybrid":
            self._init_hybrid_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Decision Engine initialized with model type: {self.model_type}")
    
    def _init_llm_model(self):
        """Initialize LLM-based decision model."""
        # In a real implementation, this would initialize an LLM client
        # For this example, we'll simulate with a simple decision process
        self.llm_config = self.config.get("llm_config", {})
        self.model_loaded = True
        logger.info("LLM decision model initialized")
    
    def _init_rl_model(self):
        """Initialize RL-based decision model."""
        # In a real implementation, this would load a trained RL model
        self.rl_config = self.config.get("rl_config", {})
        
        # Load state and action spaces
        self.state_features = self.rl_config.get("state_features", [
            "drift_severity", "performance_drop", "error_rate", 
            "latency_increase", "data_quality_score"
        ])
        self.action_space = self.rl_config.get("action_space", [
            "ROLLBACK", "RETRAIN", "INVESTIGATE", "FIX_PIPELINE", "NOTIFY"
        ])
        
        # For simulation, we'll use a simple Q-table approach
        # In a real implementation, this would be a proper RL model
        self.q_table = {}
        self.model_loaded = True
        logger.info("RL decision model initialized")
    
    def _init_hybrid_model(self):
        """Initialize hybrid (RL+LLM) decision model."""
        self._init_llm_model()
        self._init_rl_model()
        logger.info("Hybrid decision model initialized")
    
    def start(self):
        """Start the decision engine."""
        self.health_status = "HEALTHY"
        logger.info("Decision Engine started")
    
    def get_health(self) -> str:
        """Get the health status of the decision engine."""
        return self.health_status
    
    def decide_action(self, issue: Dict, metrics: Dict, logs: List[Dict]) -> Dict:
        """
        Determine the best action to take based on issue, metrics, and logs.
        
        Args:
            issue: Details of the detected issue
            metrics: Current system metrics
            logs: Recent system logs
            
        Returns:
            Dict: Action to take
        """
        if self.model_type == "llm":
            action = self._llm_decide(issue, metrics, logs)
        elif self.model_type == "rl":
            action = self._rl_decide(issue, metrics)
        else:  # hybrid
            llm_action = self._llm_decide(issue, metrics, logs)
            rl_action = self._rl_decide(issue, metrics)
            
            # Use confidence scores to select final action
            if llm_action.get("confidence", 0) > rl_action.get("confidence", 0):
                action = llm_action
                action["model_used"] = "llm"
            else:
                action = rl_action
                action["model_used"] = "rl"
        
        # Record action for history
        self.last_action = action
        self.action_history.append({
            "timestamp": datetime.now().isoformat(),
            "issue": issue,
            "action": action
        })
        
        return action
    
    def _llm_decide(self, issue: Dict, metrics: Dict, logs: List[Dict]) -> Dict:
        """Use LLM to decide on an action."""
        # In a real implementation, this would call an LLM API
        # For this example, we'll simulate an LLM decision
        
        issue_type = issue.get("type", "")
        issue_severity = issue.get("severity", "medium")
        
        # Create a prompt from the issue, metrics, and logs
        # This would be sent to the LLM in a real implementation
        prompt = self._create_llm_prompt(issue, metrics, logs)
        
        # Simulate LLM response based on issue type
        if "drift" in issue_type.lower():
            if issue_severity == "high":
                # For high severity drift, recommend rollback
                return {
                    "type": "ROLLBACK",
                    "model_id": issue.get("model_id"),
                    "version": issue.get("last_stable_version"),
                    "reason": "Critical data drift detected",
                    "confidence": 0.92
                }
            else:
                # For lower severity drift, recommend retraining
                return {
                    "type": "RETRAIN",
                    "pipeline_id": issue.get("pipeline_id"),
                    "config_updates": {
                        "learning_rate": 0.001,
                        "epochs": 20,
                        "include_recent_data": True
                    },
                    "reason": "Data drift detected, model needs updating",
                    "confidence": 0.85
                }
        elif "performance" in issue_type.lower():
            # For performance issues, recommend investigation
            return {
                "type": "INVESTIGATE",
                "focus_areas": ["data_quality", "model_complexity", "feature_importance"],
                "reason": "Performance degradation detected",
                "confidence": 0.78
            }
        elif "error" in issue_type.lower():
            # For error rate issues, recommend pipeline fix
            return {
                "type": "FIX_PIPELINE",
                "pipeline_id": issue.get("pipeline_id"),
                "fix_type": "error_handling",
                "fix_params": {
                    "retry_count": 3,
                    "timeout": 300,
                    "add_validation": True
                },
                "reason": "Increased error rate in pipeline",
                "confidence": 0.88
            }
        else:
            # Default to investigation for unknown issues
            return {
                "type": "INVESTIGATE",
                "focus_areas": ["general"],
                "reason": "Unknown issue requires investigation",
                "confidence": 0.65
            }
    
    def _create_llm_prompt(self, issue: Dict, metrics: Dict, logs: List[Dict]) -> str:
        """Create a prompt for the LLM based on issue, metrics, and logs."""
        # Convert recent logs to a formatted string
        log_text = "\n".join([f"{log.get('timestamp')}: {log.get('message')}" 
                            for log in logs[:5]])
        
        # Format metrics
        metrics_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
        
        prompt = f"""
        You are an ML Operations assistant. Based on the following issue, metrics, and logs,
        decide on the best action to take to resolve the issue.
        
        ISSUE:
        Type: {issue.get('type')}
        Severity: {issue.get('severity')}
        Description: {issue.get('description')}
        Model ID: {issue.get('model_id')}
        Pipeline ID: {issue.get('pipeline_id')}
        
        CURRENT METRICS:
        {metrics_text}
        
        RECENT LOGS:
        {log_text}
        
        Please respond with one of the following actions:
        1. ROLLBACK - Roll back to a previous model version
        2. RETRAIN - Trigger model retraining
        3. INVESTIGATE - Conduct deeper investigation into the issue
        4. FIX_PIPELINE - Apply a fix to the data pipeline
        
        Include the action type, reason for the action, and any specific parameters needed.
        """
        
        return prompt
    
    def _rl_decide(self, issue: Dict, metrics: Dict) -> Dict:
        """Use RL model to decide on an action."""
        # Extract state features from issue and metrics
        state = self._extract_state_features(issue, metrics)
        
        # Convert state to a tuple for Q-table lookup
        state_tuple = tuple(state)
        
        # If we don't have this state in our Q-table, initialize it
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = {action: 0.0 for action in self.action_space}
        
        # Get the best action according to our Q-table
        action_type = max(self.q_table[state_tuple], key=self.q_table[state_tuple].get)
        
        # In a real implementation, this would use the actual RL model's prediction
        # Here we simulate based on the action type
        if action_type == "ROLLBACK":
            action = {
                "type": "ROLLBACK",
                "model_id": issue.get("model_id"),
                "version": issue.get("last_stable_version"),
                "reason": "RL model determined rollback as optimal action",
                "confidence": 0.85
            }
        elif action_type == "RETRAIN":
            action = {
                "type": "RETRAIN",
                "pipeline_id": issue.get("pipeline_id"),
                "config_updates": {
                    "learning_rate": 0.001,
                    "batch_size": 64
                },
                "reason": "RL model recommends retraining",
                "confidence": 0.79
            }
        elif action_type == "INVESTIGATE":
            action = {
                "type": "INVESTIGATE",
                "focus_areas": ["model_inputs", "prediction_distribution"],
                "reason": "RL model suggests deeper investigation",
                "confidence": 0.72
            }
        elif action_type == "FIX_PIPELINE":
            action = {
                "type": "FIX_PIPELINE",
                "pipeline_id": issue.get("pipeline_id"),
                "fix_type": "data_validation",
                "fix_params": {
                    "add_validation_step": True
                },
                "reason": "RL model recommends pipeline fix",
                "confidence": 0.81
            }
        else:
            action = {
                "type": "NOTIFY",
                "recipients": ["mlops_team"],
                "reason": "RL model cannot determine optimal action",
                "confidence": 0.50
            }
        
        return action
    
    def _extract_state_features(self, issue: Dict, metrics: Dict) -> List[float]:
        """Extract state features for RL model from issue and metrics."""
        state = []
        
        # Extract features based on our state_features list
        for feature in self.state_features:
            if feature == "drift_severity":
                if "drift" in issue.get("type", "").lower():
                    severity_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
                    state.append(severity_map.get(issue.get("severity", "medium"), 0.5))
                else:
                    state.append(0.0)
            
            elif feature == "performance_drop":
                perf_drop = metrics.get("performance_drop", 0.0)
                state.append(min(perf_drop / 100.0, 1.0))  # Normalize to [0,1]
            
            elif feature == "error_rate":
                error_rate = metrics.get("error_rate", 0.0)
                state.append(min(error_rate / 0.2, 1.0))  # Normalize to [0,1]
            
            elif feature == "latency_increase":
                latency_inc = metrics.get("latency_increase_pct", 0.0)
                state.append(min(latency_inc / 200.0, 1.0))  # Normalize to [0,1]
            
            elif feature == "data_quality_score":
                data_quality = metrics.get("data_quality_score", 0.5)
                state.append(1.0 - data_quality)  # Invert so higher means worse
            
            else:
                # Default for unknown features
                state.append(0.5)
        
        return state

    def update_model(self, feedback: Dict):
        """
        Update the decision model based on feedback.
        
        Args:
            feedback: Feedback about an action, including the outcome
        """
        # In a real implementation, this would update the model based on feedback
        # For now, we'll just log it
        logger.info(f"Received feedback for action: {feedback}")
        
        # If we're using RL, we could update our Q-table
        if self.model_type in ["rl", "hybrid"] and feedback.get("action_id"):
            # Find the action in our history
            for entry in reversed(self.action_history):
                if entry["action"].get("id") == feedback.get("action_id"):
                    # Update Q-values based on feedback
                    reward = feedback.get("score", 0.0)
                    logger.info(f"Updating RL model with reward: {reward}")
                    break
