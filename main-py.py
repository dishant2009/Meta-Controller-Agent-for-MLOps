"""
Main entry point for the Meta-Controller Agent for MLOps.
"""

import os
import logging
import time
from meta_controller.agent.core import MetaControllerAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("meta_controller")

def main():
    """Main entry point for the application."""
    logger.info("Starting Meta-Controller Agent for MLOps")
    
    # Load configuration
    try:
        config_path = os.environ.get("CONFIG_PATH", "config.json")
        logger.info(f"Loading configuration from {config_path}")
        
        # Initialize the Meta-Controller Agent
        agent = MetaControllerAgent(config_path)
        
        # Start the agent
        agent.start()
        
        # Keep running
        while True:
            # Print health status periodically
            logger.info(f"System health: {agent.get_system_health()['status']}")
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Shutting down Meta-Controller Agent")
    except Exception as e:
        logger.error(f"Error in main application: {str(e)}")
        raise


if __name__ == "__main__":
    main()
