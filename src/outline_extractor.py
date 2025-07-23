"""
Outline Extractor Component

This module handles the extraction of document outlines from PDFs,
combining embedded outline extraction with font-based heading detection.
"""

import logging
from pathlib import Path
from typing import List, Optional
import fitz  # PyMuPDF

from .models import HeadingInfo
from .pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)


class OutlineExtractor:
    """Handles extraction of document outlines using multiple strategies."""
    
    def __init__(self):
        """Initialize the OutlineExtractor."""
        self.pdf_processor = PDFProcessor()
    
    def extract_outline(self, doc: fitz.Document) -> List[HeadingInfo]:
        """
        Extract outline from PDF using primary and fallback methods.
        
        Args:
            doc: PyMuPDF Document object
            
        Returns:
            List of HeadingInfo objects representing the document outline
        """
        try:
            if doc is None or doc.is_closed:
                logger.debug("Cannot extract outline from invalid document")
                return []
            
            # Primary method: try embedded outline first
            if self.pdf_processor.has_embedded_outline(doc):
                logger.debug("Using embedded outline extraction")
                headings = self.pdf_processor.extract_embedded_outline(doc)
                
                if headings:
                    logger.debug(f"Successfully extracted {len(headings)} headings from embedded outline")
                    return headings
                else:
                    logger.debug("Embedded outline extraction returned no headings")
            
            # Fallback method: font-based heading detection
            logger.debug("Using font-based heading detection")
            headings = self.pdf_processor.detect_font_based_headings(doc)
            
            if headings:
                logger.debug(f"Successfully extracted {len(headings)} headings from font analysis")
                return headings
            else:
                logger.debug("Font-based detection returned no headings")
            
            # No headings found with any method
            logger.debug("No headings found using any extraction method")
            return []
            
        except Exception as e:
            logger.error(f"Error extracting outline: {e}")
            return []
    
    def extract_title_and_outline(self, file_path: Path) -> tuple[str, List[HeadingInfo]]:
        """
        Extract both title and outline from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (title, outline_headings)
        """
        try:
            # Open the PDF document
            doc = self.pdf_processor.open_pdf(file_path)
            
            if doc is None:
                logger.warning(f"Failed to open PDF: {file_path}")
                return "", []
            
            try:
                # Validate document
                if not self.pdf_processor.validate_document(doc):
                    logger.warning(f"Invalid or unprocessable PDF: {file_path}")
                    return "", []
                
                # Extract title
                title = self.pdf_processor.extract_title(doc)
                logger.debug(f"Extracted title: '{title}'")
                
                # Extract outline
                outline = self.extract_outline(doc)
                logger.debug(f"Extracted {len(outline)} headings")
                
                return title, outline
                
            finally:
                # Always close the document
                if not doc.is_closed:
                    doc.close()
                    
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return "", []
    
    def extract_title_and_outline_optimized(self, file_path: Path) -> tuple[str, List[HeadingInfo]]:
        """
        Memory-efficient version of title and outline extraction with page-by-page processing.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (title, outline_headings)
        """
        try:
            # Open the PDF document
            doc = self.pdf_processor.open_pdf(file_path)
            
            if doc is None:
                logger.warning(f"Failed to open PDF: {file_path}")
                return "", []
            
            try:
                # Validate document
                if not self.pdf_processor.validate_document(doc):
                    logger.warning(f"Invalid or unprocessable PDF: {file_path}")
                    return "", []
                
                # Extract title (this is fast)
                title = self.pdf_processor.extract_title(doc)
                logger.debug(f"Extracted title: '{title}'")
                
                # Use memory-efficient outline extraction
                outline = self.extract_outline_memory_efficient(doc)
                logger.debug(f"Extracted {len(outline)} headings")
                
                return title, outline
                
            finally:
                # Always close the document
                if not doc.is_closed:
                    doc.close()
                    
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return "", []
    
    def extract_outline_memory_efficient(self, doc: fitz.Document) -> List[HeadingInfo]:
        """
        Memory-efficient outline extraction with page-by-page processing for large documents.
        
        Args:
            doc: PyMuPDF Document object
            
        Returns:
            List of HeadingInfo objects representing the document outline
        """
        try:
            if doc is None or doc.is_closed:
                logger.debug("Cannot extract outline from invalid document")
                return []
            
            # Primary method: try embedded outline first (fast path)
            if self.pdf_processor.has_embedded_outline(doc):
                logger.debug("Using embedded outline extraction (fast path)")
                headings = self.pdf_processor.extract_embedded_outline(doc)
                
                if headings:
                    logger.debug(f"Successfully extracted {len(headings)} headings from embedded outline")
                    return headings
                else:
                    logger.debug("Embedded outline extraction returned no headings")
            
            # Fallback method: memory-efficient font-based heading detection
            logger.debug("Using memory-efficient font-based heading detection")
            headings = self.pdf_processor.detect_font_based_headings_memory_efficient(doc)
            
            if headings:
                logger.debug(f"Successfully extracted {len(headings)} headings from font analysis")
                return headings
            else:
                logger.debug("Font-based detection returned no headings")
            
            # No headings found with any method
            logger.debug("No headings found using any extraction method")
            return []
            
        except Exception as e:
            logger.error(f"Error extracting outline: {e}")
            return []
    
    def _dict_to_heading_info(self, heading_dict: dict) -> HeadingInfo:
        """
        Convert dictionary representation back to HeadingInfo object.
        
        Args:
            heading_dict: Dictionary with level, text, page keys
            
        Returns:
            HeadingInfo object
        """
        from .models import HeadingInfo
        return HeadingInfo(
            level=heading_dict.get("level", "H1"),
            text=heading_dict.get("text", ""),
            page=heading_dict.get("page", 1)
        )