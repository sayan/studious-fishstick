"""Logging utilities for election data analysis system."""

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logger(name: str, log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with both console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file. If None, logs only to console
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_analysis_progress(ac_name: str, stage: str, logger: logging.Logger):
    """
    Log analysis progress for an assembly constituency.
    
    Args:
        ac_name: Assembly constituency name
        stage: Current analysis stage
        logger: Logger instance
    """
    logger.info(f"[{ac_name}] {stage}")


def log_error(error: Exception, context: str, logger: logging.Logger):
    """
    Log error with context information.
    
    Args:
        error: Exception that occurred
        context: Context where error occurred
        logger: Logger instance
    """
    logger.error(f"Error in {context}: {str(error)}", exc_info=True)
