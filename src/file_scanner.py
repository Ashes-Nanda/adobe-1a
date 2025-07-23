"""
File Scanner Component

This module handles scanning the input directory for PDF files
and validating their accessibility.
"""

import logging
import os
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class FileScanner:
    """Handles discovery and validation of PDF files for processing."""
    
    def __init__(self):
        """Initialize the FileScanner."""
        pass
    
    def scan_input_directory(self, input_path: Path) -> List[Path]:
        """
        Scan the input directory for PDF files.
        
        Args:
            input_path: Path to the input directory
            
        Returns:
            List of PDF file paths found
        """
        logger.info(f"Scanning directory: {input_path}")
        pdf_files = []
        
        try:
            # Check if input directory exists
            if not input_path.exists():
                logger.error(f"Input directory does not exist: {input_path}")
                return pdf_files
            
            if not input_path.is_dir():
                logger.error(f"Input path is not a directory: {input_path}")
                return pdf_files
            
            # Scan for PDF files
            for file_path in input_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == '.pdf':
                    if self.validate_pdf_file(file_path):
                        pdf_files.append(file_path)
                        logger.debug(f"Found valid PDF: {file_path}")
                    else:
                        logger.warning(f"Invalid or unreadable PDF: {file_path}")
            
            logger.info(f"Found {len(pdf_files)} valid PDF files")
            # Sort files for consistent processing order
            pdf_files.sort()
            
        except PermissionError as e:
            logger.error(f"Permission denied accessing directory {input_path}: {e}")
        except OSError as e:
            logger.error(f"OS error accessing directory {input_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error scanning directory {input_path}: {e}")
        
        return pdf_files
    
    def validate_pdf_file(self, file_path: Path) -> bool:
        """
        Validate if a file is a readable PDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            True if file is valid and readable
        """
        try:
            # Check if file exists and is readable
            if not file_path.exists():
                logger.debug(f"File does not exist: {file_path}")
                return False
            
            if not file_path.is_file():
                logger.debug(f"Path is not a file: {file_path}")
                return False
            
            # Check file permissions
            if not os.access(file_path, os.R_OK):
                logger.debug(f"File is not readable: {file_path}")
                return False
            
            # Check file size (empty files are invalid)
            if file_path.stat().st_size == 0:
                logger.debug(f"File is empty: {file_path}")
                return False
            
            # Basic PDF header validation
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(8)
                    if not header.startswith(b'%PDF-'):
                        logger.debug(f"File does not have PDF header: {file_path}")
                        return False
            except (IOError, OSError) as e:
                logger.debug(f"Error reading file header {file_path}: {e}")
                return False
            
            return True
            
        except PermissionError as e:
            logger.debug(f"Permission error validating {file_path}: {e}")
            return False
        except OSError as e:
            logger.debug(f"OS error validating {file_path}: {e}")
            return False
        except Exception as e:
            logger.debug(f"Unexpected error validating {file_path}: {e}")
            return False