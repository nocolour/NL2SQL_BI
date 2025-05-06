import logging
from datetime import datetime
import os

def setup_logger():
    """Set up and configure the logger with daily log files"""
    log_date = datetime.now().strftime("%Y-%m-%d")
    log_filename = f"error_{log_date}.log"
    
    # Ensure the logs directory exists
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    log_path = os.path.join("logs", log_filename)
    
    logging.basicConfig(
        filename=log_filename,
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return logging.getLogger('nl2sql')