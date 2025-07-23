"""
Configuration module for PDF Outline Extractor

This module contains configuration settings and constants used
throughout the application.
"""

import logging
import sys
from pathlib import Path

# Application constants
DEFAULT_INPUT_DIR = Path("/app/input")
DEFAULT_OUTPUT_DIR = Path("/app/output")

# Performance constraints
MAX_PROCESSING_TIME_SECONDS = 10  # For 50-page PDFs
MAX_MODEL_SIZE_MB = 200

# Supported heading levels
SUPPORTED_HEADING_LEVELS = ["H1", "H2", "H3"]


def setup_logging(level: int = logging.INFO) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger("fitz").setLevel(logging.WARNING)  # Reduce PyMuPDF verbosity
    logging.getLogger("PIL").setLevel(logging.WARNING)   # Reduce PIL verbosity if used