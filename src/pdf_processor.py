"""
PDF Processor Component

This module handles PDF file operations including opening, metadata extraction,
title extraction, and basic document validation using PyMuPDF.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import fitz  # PyMuPDF
from .models import HeadingInfo, TextBlock, HeadingCandidate

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF file operations and basic document processing."""
    
    def __init__(self):
        """Initialize the PDFProcessor."""
        pass
    
    def open_pdf(self, file_path: Path) -> Optional[fitz.Document]:
        """
        Open a PDF file and return a PyMuPDF document object.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            PyMuPDF Document object if successful, None if failed
        """
        try:
            logger.debug(f"Opening PDF: {file_path}")
            doc = fitz.open(str(file_path))
            
            # Basic validation - check if document can be accessed
            if doc.is_closed:
                logger.error(f"Document is closed after opening: {file_path}")
                return None
            
            # Check if document is encrypted and we can't access it
            if doc.needs_pass:
                logger.warning(f"Document is password protected: {file_path}")
                doc.close()
                return None
            
            logger.debug(f"Successfully opened PDF: {file_path}")
            return doc
            
        except fitz.FileDataError as e:
            logger.error(f"File data error opening {file_path}: {e}")
            return None
        except fitz.FileNotFoundError as e:
            logger.error(f"File not found error opening {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error opening {file_path}: {e}")
            return None
    
    def extract_metadata(self, doc: fitz.Document) -> Dict:
        """
        Extract metadata from a PDF document.
        
        Args:
            doc: PyMuPDF Document object
            
        Returns:
            Dictionary containing metadata fields
        """
        try:
            if doc.is_closed:
                logger.error("Cannot extract metadata from closed document")
                return {}
            
            metadata = doc.metadata
            logger.debug(f"Extracted metadata: {metadata}")
            
            # Normalize metadata - ensure all values are strings and handle None values
            normalized_metadata = {}
            for key, value in metadata.items():
                if value is not None:
                    normalized_metadata[key] = str(value).strip()
                else:
                    normalized_metadata[key] = ""
            
            return normalized_metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
    
    def get_page_count(self, doc: fitz.Document) -> int:
        """
        Get the number of pages in a PDF document.
        
        Args:
            doc: PyMuPDF Document object
            
        Returns:
            Number of pages, 0 if error
        """
        try:
            if doc.is_closed:
                logger.error("Cannot get page count from closed document")
                return 0
            
            page_count = doc.page_count
            logger.debug(f"Document has {page_count} pages")
            return page_count
            
        except Exception as e:
            logger.error(f"Error getting page count: {e}")
            return 0
    
    def validate_document(self, doc: fitz.Document) -> bool:
        """
        Validate that a PDF document is accessible and processable.
        
        Args:
            doc: PyMuPDF Document object
            
        Returns:
            True if document is valid and processable
        """
        try:
            if doc is None:
                logger.debug("Document is None")
                return False
            
            if doc.is_closed:
                logger.debug("Document is closed")
                return False
            
            if doc.needs_pass:
                logger.debug("Document requires password")
                return False
            
            # Check if document has at least one page
            page_count = self.get_page_count(doc)
            if page_count <= 0:
                logger.debug("Document has no pages")
                return False
            
            # Try to access the first page to ensure document is readable
            try:
                first_page = doc[0]
                if first_page is None:
                    logger.debug("Cannot access first page")
                    return False
            except Exception as e:
                logger.debug(f"Error accessing first page: {e}")
                return False
            
            logger.debug("Document validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Error validating document: {e}")
            return False
    
    def extract_title_from_metadata(self, doc: fitz.Document) -> str:
        """
        Extract title from PDF metadata with text cleaning and normalization.
        
        Args:
            doc: PyMuPDF Document object
            
        Returns:
            Cleaned title string, empty string if no title found or error
        """
        try:
            if doc is None or doc.is_closed:
                logger.debug("Cannot extract title from invalid document")
                return ""
            
            metadata = self.extract_metadata(doc)
            title = metadata.get('title', '')
            
            if not title:
                logger.debug("No title found in metadata")
                return ""
            
            # Clean and normalize the title
            cleaned_title = self._clean_title_text(title)
            
            if not cleaned_title:
                logger.debug("Title is empty after cleaning")
                return ""
            
            logger.debug(f"Extracted title from metadata: '{cleaned_title}'")
            return cleaned_title
            
        except Exception as e:
            logger.error(f"Error extracting title from metadata: {e}")
            return ""
    
    def extract_title_from_first_page(self, doc: fitz.Document) -> str:
        """
        Extract title from first page by analyzing text for title candidates.
        Identifies largest font text and central positioning as title indicators.
        
        Args:
            doc: PyMuPDF Document object
            
        Returns:
            Extracted title string, empty string if no clear title found
        """
        try:
            if doc is None or doc.is_closed:
                logger.debug("Cannot extract title from invalid document")
                return ""
            
            if self.get_page_count(doc) == 0:
                logger.debug("Document has no pages")
                return ""
            
            # Get the first page
            first_page = doc[0]
            
            # Extract text blocks with font information
            text_blocks = self._extract_text_blocks_with_fonts(first_page)
            
            if not text_blocks:
                logger.debug("No text blocks found on first page")
                return ""
            
            # Find title candidates based on font size and positioning
            title_candidates = self._find_title_candidates(text_blocks)
            
            if not title_candidates:
                logger.debug("No title candidates found")
                return ""
            
            # Select the best title candidate
            best_title = self._select_best_title_candidate(title_candidates)
            
            if best_title:
                cleaned_title = self._clean_title_text(best_title)
                logger.debug(f"Extracted title from first page: '{cleaned_title}'")
                return cleaned_title
            
            logger.debug("No clear title found on first page")
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting title from first page: {e}")
            return ""
    
    def extract_title(self, doc: fitz.Document) -> str:
        """
        Extract title using primary (metadata) and fallback (first page) methods.
        
        Args:
            doc: PyMuPDF Document object
            
        Returns:
            Extracted title string, empty string if no title found
        """
        try:
            # Primary method: extract from metadata
            title = self.extract_title_from_metadata(doc)
            
            if title:
                logger.debug(f"Title extracted from metadata: '{title}'")
                return title
            
            # Fallback method: analyze first page
            title = self.extract_title_from_first_page(doc)
            
            if title:
                logger.debug(f"Title extracted from first page: '{title}'")
                return title
            
            logger.debug("No title found using any method")
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting title: {e}")
            return ""
    
    def _clean_title_text(self, title: str) -> str:
        """
        Clean and normalize title text.
        
        Args:
            title: Raw title text
            
        Returns:
            Cleaned title text
        """
        if not title:
            return ""
        
        # Strip whitespace
        cleaned = title.strip()
        
        # Replace control characters that should be spaces with spaces
        cleaned = re.sub(r'[\x00\x0C\x0D]', ' ', cleaned)
        
        # Remove other PDF artifacts
        cleaned = re.sub(r'[\x01-\x08\x0B\x0E-\x1F\x7F]', '', cleaned)
        
        # Normalize whitespace (replace multiple spaces/tabs/newlines with single space)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove excessive punctuation at the end
        cleaned = re.sub(r'[.]{2,}$', '', cleaned)
        
        return cleaned.strip()
    
    def _extract_text_blocks_with_fonts(self, page: fitz.Page) -> List[Dict]:
        """
        Extract text blocks with font information from a page.
        
        Args:
            page: PyMuPDF Page object
            
        Returns:
            List of text blocks with font information
        """
        try:
            text_blocks = []
            
            # Get text with font information
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        
                        # Extract font information
                        font_size = span.get("size", 0)
                        font_flags = span.get("flags", 0)
                        bbox = span.get("bbox", (0, 0, 0, 0))
                        
                        # Determine if text is bold (flag 16) or italic (flag 2)
                        is_bold = bool(font_flags & 16)
                        is_italic = bool(font_flags & 2)
                        
                        text_blocks.append({
                            "text": text,
                            "font_size": font_size,
                            "is_bold": is_bold,
                            "is_italic": is_italic,
                            "bbox": bbox
                        })
            
            return text_blocks
            
        except Exception as e:
            logger.error(f"Error extracting text blocks with fonts: {e}")
            return []
    
    def _find_title_candidates(self, text_blocks: List[Dict]) -> List[Dict]:
        """
        Find potential title candidates based on font size and positioning.
        
        Args:
            text_blocks: List of text blocks with font information
            
        Returns:
            List of title candidates with confidence scores
        """
        if not text_blocks:
            return []
        
        candidates = []
        
        # Calculate average font size for comparison
        font_sizes = [block["font_size"] for block in text_blocks if block["font_size"] > 0]
        if not font_sizes:
            return []
        
        avg_font_size = sum(font_sizes) / len(font_sizes)
        max_font_size = max(font_sizes)
        
        # Get page dimensions for positioning analysis
        page_width = 0
        page_height = 0
        if text_blocks:
            # Estimate page dimensions from bounding boxes
            all_x = [bbox[0] for block in text_blocks for bbox in [block["bbox"]] if bbox]
            all_x.extend([bbox[2] for block in text_blocks for bbox in [block["bbox"]] if bbox])
            all_y = [bbox[1] for block in text_blocks for bbox in [block["bbox"]] if bbox]
            all_y.extend([bbox[3] for block in text_blocks for bbox in [block["bbox"]] if bbox])
            
            if all_x and all_y:
                page_width = max(all_x) - min(all_x)
                page_height = max(all_y) - min(all_y)
        
        for block in text_blocks:
            text = block["text"]
            font_size = block["font_size"]
            is_bold = block["is_bold"]
            bbox = block["bbox"]
            
            # Skip very short text (likely not a title)
            if len(text.strip()) < 3:
                continue
            
            # Skip very long text (likely paragraph text)
            if len(text.strip()) > 200:
                continue
            
            # Calculate confidence score based on various factors
            confidence = 0.0
            
            # Font size factor (larger fonts get higher scores)
            if font_size > avg_font_size * 1.2:
                confidence += 0.3
            if font_size >= max_font_size * 0.9:
                confidence += 0.2
            
            # Bold text gets bonus points
            if is_bold:
                confidence += 0.2
            
            # Position factor (text near top and center gets bonus)
            if bbox and page_width > 0 and page_height > 0:
                x_center = (bbox[0] + bbox[2]) / 2
                y_pos = bbox[1]
                
                # Bonus for being in upper portion of page
                if y_pos < page_height * 0.3:
                    confidence += 0.2
                
                # Bonus for being horizontally centered
                page_center_x = page_width / 2
                distance_from_center = abs(x_center - page_center_x) / page_width
                if distance_from_center < 0.3:  # Within 30% of center
                    confidence += 0.1
            
            # Text characteristics
            # Bonus for title-like characteristics
            if text.istitle() or text.isupper():
                confidence += 0.1
            
            # Penalty for text that looks like body content
            if any(word in text.lower() for word in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']):
                if len(text.split()) > 5:  # Only penalize longer phrases
                    confidence -= 0.1
            
            candidates.append({
                "text": text,
                "font_size": font_size,
                "is_bold": is_bold,
                "bbox": bbox,
                "confidence": max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
            })
        
        # Sort by confidence score (highest first)
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Return top candidates with confidence > 0.2
        return [c for c in candidates if c["confidence"] > 0.2]
    
    def _select_best_title_candidate(self, candidates: List[Dict]) -> str:
        """
        Select the best title candidate from the list.
        
        Args:
            candidates: List of title candidates with confidence scores
            
        Returns:
            Best title text, empty string if no good candidate
        """
        if not candidates:
            return ""
        
        # Return the highest confidence candidate
        best_candidate = candidates[0]
        
        # Additional validation - ensure minimum confidence
        if best_candidate["confidence"] < 0.3:
            return ""
        
        return best_candidate["text"]
    
    def extract_embedded_outline(self, doc: fitz.Document) -> List[HeadingInfo]:
        """
        Extract PDF table of contents using doc.get_toc() and map to H1, H2, H3 levels.
        
        Args:
            doc: PyMuPDF Document object
            
        Returns:
            List of HeadingInfo objects from embedded outline
        """
        try:
            if doc is None or doc.is_closed:
                logger.debug("Cannot extract outline from invalid document")
                return []
            
            # Get the table of contents
            toc = doc.get_toc()
            
            if not toc:
                logger.debug("No embedded outline found in document")
                return []
            
            logger.debug(f"Found embedded outline with {len(toc)} entries")
            
            headings = []
            
            for entry in toc:
                try:
                    # TOC entry format: [level, title, page_number]
                    if len(entry) < 3:
                        logger.warning(f"Invalid TOC entry format: {entry}")
                        continue
                    
                    level, title, page_num = entry[0], entry[1], entry[2]
                    
                    # Clean and validate title
                    cleaned_title = self._clean_heading_text(title)
                    if not cleaned_title:
                        logger.debug(f"Skipping empty title in TOC entry: {entry}")
                        continue
                    
                    # Map outline level to H1, H2, H3 (limit to 3 levels max)
                    heading_level = self._map_outline_level_to_heading(level)
                    
                    # Validate and normalize page number (ensure 1-based)
                    normalized_page = self._normalize_page_number(page_num)
                    if normalized_page < 1:
                        logger.warning(f"Invalid page number {page_num} in TOC entry, skipping")
                        continue
                    
                    # Create HeadingInfo object
                    heading = HeadingInfo(
                        level=heading_level,
                        text=cleaned_title,
                        page=normalized_page
                    )
                    
                    # Validate the heading
                    if heading.validate():
                        headings.append(heading)
                        logger.debug(f"Added heading: {heading_level} '{cleaned_title}' on page {normalized_page}")
                    else:
                        logger.warning(f"Invalid heading created from TOC entry: {entry}")
                
                except Exception as e:
                    logger.error(f"Error processing TOC entry {entry}: {e}")
                    continue
            
            logger.debug(f"Successfully extracted {len(headings)} headings from embedded outline")
            return headings
            
        except Exception as e:
            logger.error(f"Error extracting embedded outline: {e}")
            return []
    
    def has_embedded_outline(self, doc: fitz.Document) -> bool:
        """
        Check if document has an embedded outline/table of contents.
        
        Args:
            doc: PyMuPDF Document object
            
        Returns:
            True if document has embedded outline
        """
        try:
            if doc is None or doc.is_closed:
                return False
            
            toc = doc.get_toc()
            return bool(toc)
            
        except Exception as e:
            logger.error(f"Error checking for embedded outline: {e}")
            return False
    
    def _map_outline_level_to_heading(self, outline_level: int) -> str:
        """
        Map PDF outline level to heading level (H1, H2, H3).
        
        Args:
            outline_level: PDF outline level (typically 1, 2, 3, etc.)
            
        Returns:
            Heading level string ("H1", "H2", or "H3")
        """
        # Map outline levels to heading levels, capping at H3
        if outline_level <= 1:
            return "H1"
        elif outline_level == 2:
            return "H2"
        else:
            # All levels 3 and above map to H3
            return "H3"
    
    def _normalize_page_number(self, page_num: int) -> int:
        """
        Normalize page number to ensure it's 1-based and valid.
        
        Args:
            page_num: Raw page number from PDF outline
            
        Returns:
            Normalized 1-based page number
        """
        try:
            # Convert to integer if needed
            if isinstance(page_num, float):
                page_num = int(page_num)
            
            # Ensure it's a positive integer (1-based)
            if page_num <= 0:
                return 1  # Default to page 1 for invalid page numbers
            
            return page_num
            
        except (ValueError, TypeError):
            logger.warning(f"Invalid page number format: {page_num}, defaulting to 1")
            return 1
    
    def extract_text_with_font_info(self, doc: fitz.Document) -> List[TextBlock]:
        """
        Extract text from all pages with font information for heading detection.
        
        Args:
            doc: PyMuPDF Document object
            
        Returns:
            List of TextBlock objects with font information
        """
        try:
            if doc is None or doc.is_closed:
                logger.debug("Cannot extract text from invalid document")
                return []
            
            text_blocks = []
            page_count = self.get_page_count(doc)
            
            if page_count == 0:
                logger.debug("Document has no pages")
                return []
            
            logger.debug(f"Extracting text with font info from {page_count} pages")
            
            for page_num in range(page_count):
                try:
                    page = doc[page_num]
                    page_blocks = self._extract_page_text_blocks(page, page_num + 1)
                    text_blocks.extend(page_blocks)
                    
                except Exception as e:
                    logger.error(f"Error extracting text from page {page_num + 1}: {e}")
                    continue
            
            logger.debug(f"Extracted {len(text_blocks)} text blocks from document")
            return text_blocks
            
        except Exception as e:
            logger.error(f"Error extracting text with font info: {e}")
            return []
    
    def detect_font_based_headings(self, doc: fitz.Document) -> List[HeadingInfo]:
        """
        Detect headings based on font size analysis and font weight.
        
        Args:
            doc: PyMuPDF Document object
            
        Returns:
            List of HeadingInfo objects detected from font analysis
        """
        try:
            if doc is None or doc.is_closed:
                logger.debug("Cannot detect headings from invalid document")
                return []
            
            # Extract text blocks with font information
            text_blocks = self.extract_text_with_font_info(doc)
            
            if not text_blocks:
                logger.debug("No text blocks found for font-based heading detection")
                return []
            
            # Find heading candidates based on font analysis
            heading_candidates = self._analyze_font_based_headings(text_blocks)
            
            if not heading_candidates:
                logger.debug("No heading candidates found from font analysis")
                return []
            
            # Classify candidates into H1, H2, H3 levels
            classified_headings = self._classify_font_based_headings(heading_candidates)
            
            logger.debug(f"Detected {len(classified_headings)} font-based headings")
            return classified_headings
            
        except Exception as e:
            logger.error(f"Error detecting font-based headings: {e}")
            return []
    
    def detect_font_based_headings_memory_efficient(self, doc: fitz.Document) -> List[HeadingInfo]:
        """
        Memory-efficient version of font-based heading detection with page-by-page processing.
        
        Args:
            doc: PyMuPDF Document object
            
        Returns:
            List of HeadingInfo objects detected from font analysis
        """
        try:
            if doc is None or doc.is_closed:
                logger.debug("Cannot detect headings from invalid document")
                return []
            
            page_count = self.get_page_count(doc)
            if page_count == 0:
                return []
            
            # Process pages in batches to manage memory
            batch_size = min(10, page_count)  # Process max 10 pages at a time
            all_candidates = []
            
            for batch_start in range(0, page_count, batch_size):
                batch_end = min(batch_start + batch_size, page_count)
                logger.debug(f"Processing pages {batch_start + 1}-{batch_end} for heading detection")
                
                # Extract text blocks for this batch
                batch_text_blocks = []
                for page_num in range(batch_start, batch_end):
                    try:
                        page = doc[page_num]
                        page_blocks = self._extract_page_text_blocks(page, page_num + 1)
                        batch_text_blocks.extend(page_blocks)
                    except Exception as e:
                        logger.error(f"Error processing page {page_num + 1}: {e}")
                        continue
                
                if batch_text_blocks:
                    # Find heading candidates for this batch
                    batch_candidates = self._analyze_font_based_headings(batch_text_blocks)
                    all_candidates.extend(batch_candidates)
                
                # Clear batch data to free memory
                batch_text_blocks.clear()
                
                # Force garbage collection after each batch
                import gc
                gc.collect()
            
            if not all_candidates:
                logger.debug("No heading candidates found from memory-efficient font analysis")
                return []
            
            # Classify all candidates into H1, H2, H3 levels
            classified_headings = self._classify_font_based_headings(all_candidates)
            
            logger.debug(f"Detected {len(classified_headings)} font-based headings (memory-efficient)")
            return classified_headings
            
        except Exception as e:
            logger.error(f"Error detecting font-based headings (memory-efficient): {e}")
            return []
    
    def _extract_page_text_blocks(self, page: fitz.Page, page_number: int) -> List[TextBlock]:
        """
        Extract text blocks with font information from a single page.
        
        Args:
            page: PyMuPDF Page object
            page_number: 1-based page number
            
        Returns:
            List of TextBlock objects from the page
        """
        try:
            text_blocks = []
            
            # Get text with detailed font information
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        
                        # Extract font information
                        font_name = span.get("font", "")
                        font_size = span.get("size", 0)
                        font_flags = span.get("flags", 0)
                        bbox = span.get("bbox", (0, 0, 0, 0))
                        
                        # Determine font characteristics
                        is_bold = bool(font_flags & 16)  # Bold flag
                        is_italic = bool(font_flags & 2)  # Italic flag
                        
                        # Create TextBlock object
                        text_block = TextBlock(
                            text=text,
                            font_name=font_name,
                            font_size=font_size,
                            is_bold=is_bold,
                            is_italic=is_italic,
                            bbox=bbox,
                            page_number=page_number
                        )
                        
                        # Validate and add to list
                        if text_block.validate():
                            text_blocks.append(text_block)
                        else:
                            logger.debug(f"Invalid text block skipped: {text}")
            
            return text_blocks
            
        except Exception as e:
            logger.error(f"Error extracting text blocks from page {page_number}: {e}")
            return []
    
    def _analyze_font_based_headings(self, text_blocks: List[TextBlock]) -> List[HeadingCandidate]:
        """
        Analyze text blocks to identify potential headings based on font characteristics.
        
        Args:
            text_blocks: List of TextBlock objects
            
        Returns:
            List of HeadingCandidate objects
        """
        try:
            if not text_blocks:
                return []
            
            # Calculate font size statistics
            font_sizes = [block.font_size for block in text_blocks if block.font_size > 0]
            if not font_sizes:
                return []
            
            avg_font_size = sum(font_sizes) / len(font_sizes)
            max_font_size = max(font_sizes)
            min_font_size = min(font_sizes)
            
            logger.debug(f"Font size stats - avg: {avg_font_size:.1f}, max: {max_font_size:.1f}, min: {min_font_size:.1f}")
            
            candidates = []
            
            for block in text_blocks:
                # Skip very short or very long text
                text_length = len(block.clean_text())
                if text_length < 3 or text_length > 150:
                    continue
                
                # Calculate confidence score based on font characteristics
                confidence = self._calculate_heading_confidence(block, avg_font_size, max_font_size)
                
                # Only consider candidates with reasonable confidence
                if confidence > 0.3:
                    # Determine font weight string
                    font_weight = "bold" if block.is_bold else "normal"
                    
                    # Get position from bounding box
                    position = (block.bbox[0], block.bbox[1]) if block.bbox else (0, 0)
                    
                    candidate = HeadingCandidate(
                        text=block.clean_text(),
                        font_size=block.font_size,
                        font_weight=font_weight,
                        position=position,
                        page_number=block.page_number,
                        confidence_score=confidence
                    )
                    
                    if candidate.validate():
                        candidates.append(candidate)
            
            # Sort by confidence score (highest first)
            candidates.sort(key=lambda x: x.confidence_score, reverse=True)
            
            logger.debug(f"Found {len(candidates)} heading candidates from font analysis")
            return candidates
            
        except Exception as e:
            logger.error(f"Error analyzing font-based headings: {e}")
            return []
    
    def _calculate_heading_confidence(self, block: TextBlock, avg_font_size: float, max_font_size: float) -> float:
        """
        Calculate confidence score for a text block being a heading.
        
        Args:
            block: TextBlock to analyze
            avg_font_size: Average font size in document
            max_font_size: Maximum font size in document
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.0
        
        # Font size factor (larger fonts get higher scores)
        if block.font_size > avg_font_size * 1.3:
            confidence += 0.4
        elif block.font_size > avg_font_size * 1.1:
            confidence += 0.2
        
        # Maximum font size bonus
        if block.font_size >= max_font_size * 0.9:
            confidence += 0.3
        
        # Bold text bonus
        if block.is_bold:
            confidence += 0.2
        
        # Text characteristics
        text = block.clean_text()
        
        # Length factor (moderate length preferred for headings)
        if 5 <= len(text) <= 80:
            confidence += 0.1
        
        # Title case or uppercase bonus
        if text.istitle() or text.isupper():
            confidence += 0.1
        
        # Penalty for text that looks like body content
        body_indicators = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by']
        word_count = len(text.split())
        body_word_count = sum(1 for word in text.lower().split() if word in body_indicators)
        
        if word_count > 3 and body_word_count / word_count > 0.3:
            confidence -= 0.2
        
        # Penalty for text with excessive punctuation (likely body text)
        punct_count = sum(1 for char in text if char in '.,;:!?')
        if punct_count > len(text) * 0.1:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _classify_font_based_headings(self, candidates: List[HeadingCandidate]) -> List[HeadingInfo]:
        """
        Classify heading candidates into H1, H2, H3 levels based on relative font sizes.
        
        Args:
            candidates: List of HeadingCandidate objects
            
        Returns:
            List of HeadingInfo objects with assigned levels
        """
        try:
            if not candidates:
                return []
            
            # Group candidates by font size for level classification
            font_sizes = [c.font_size for c in candidates]
            unique_sizes = sorted(set(font_sizes), reverse=True)  # Largest first
            
            # Create size-to-level mapping (limit to 3 levels)
            size_to_level = {}
            for i, size in enumerate(unique_sizes[:3]):  # Only take top 3 sizes
                if i == 0:
                    size_to_level[size] = "H1"
                elif i == 1:
                    size_to_level[size] = "H2"
                else:
                    size_to_level[size] = "H3"
            
            # For sizes beyond the top 3, map to H3
            for size in unique_sizes[3:]:
                size_to_level[size] = "H3"
            
            headings = []
            
            for candidate in candidates:
                level = size_to_level.get(candidate.font_size, "H3")
                
                heading = HeadingInfo(
                    level=level,
                    text=candidate.clean_text(),
                    page=candidate.page_number
                )
                
                if heading.validate():
                    headings.append(heading)
            
            # Sort by page number, then by position on page
            headings.sort(key=lambda h: (h.page, h.text))
            
            logger.debug(f"Classified {len(headings)} headings into levels")
            return headings
            
        except Exception as e:
            logger.error(f"Error classifying font-based headings: {e}")
            return []

    def _clean_heading_text(self, text: str) -> str:
        """
        Clean and normalize heading text.
        
        Args:
            text: Raw heading text
            
        Returns:
            Cleaned heading text
        """
        if not text:
            return ""
        
        # Strip whitespace
        cleaned = text.strip()
        
        # Replace control characters and PDF artifacts
        cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', ' ', cleaned)
        
        # Normalize whitespace (replace multiple spaces/tabs/newlines with single space)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove excessive punctuation at the end (3 or more dots)
        cleaned = re.sub(r'[.]{3,}$', '', cleaned)
        
        return cleaned.strip()
    
    def validate_heading_hierarchy(self, headings: List[HeadingInfo]) -> List[HeadingInfo]:
        """
        Validate and fix heading hierarchy to ensure proper H1 → H2 → H3 sequence.
        
        Args:
            headings: List of HeadingInfo objects to validate
            
        Returns:
            List of HeadingInfo objects with validated hierarchy
        """
        try:
            if not headings:
                return []
            
            # Sort headings by page and position for sequential processing
            sorted_headings = sorted(headings, key=lambda h: (h.page, h.text))
            
            validated_headings = []
            current_level_stack = []  # Track current heading level hierarchy
            
            for heading in sorted_headings:
                # Validate and potentially adjust the heading level
                adjusted_heading = self._validate_single_heading(heading, current_level_stack)
                
                if adjusted_heading:
                    validated_headings.append(adjusted_heading)
                    self._update_level_stack(current_level_stack, adjusted_heading.level)
            
            logger.debug(f"Validated {len(validated_headings)} headings with proper hierarchy")
            return validated_headings
            
        except Exception as e:
            logger.error(f"Error validating heading hierarchy: {e}")
            return headings  # Return original headings if validation fails
    
    def _validate_single_heading(self, heading: HeadingInfo, level_stack: List[str]) -> Optional[HeadingInfo]:
        """
        Validate a single heading against the current hierarchy context.
        
        Args:
            heading: HeadingInfo object to validate
            level_stack: Current hierarchy stack (e.g., ["H1", "H2"])
            
        Returns:
            Validated HeadingInfo object or None if invalid
        """
        try:
            current_level = heading.level
            
            # If this is the first heading, ensure it starts with H1
            if not level_stack:
                if current_level != "H1":
                    logger.debug(f"Adjusting first heading from {current_level} to H1: '{heading.text}'")
                    return HeadingInfo(level="H1", text=heading.text, page=heading.page)
                return heading
            
            # Get the last level in the stack
            last_level = level_stack[-1]
            
            # Validate level progression
            if self._is_valid_level_progression(last_level, current_level):
                return heading
            
            # Try to fix invalid progression
            corrected_level = self._correct_heading_level(last_level, current_level, level_stack)
            
            if corrected_level != current_level:
                logger.debug(f"Corrected heading level from {current_level} to {corrected_level}: '{heading.text}'")
                return HeadingInfo(level=corrected_level, text=heading.text, page=heading.page)
            
            return heading
            
        except Exception as e:
            logger.error(f"Error validating single heading: {e}")
            return heading
    
    def _is_valid_level_progression(self, last_level: str, current_level: str) -> bool:
        """
        Check if the progression from last_level to current_level is valid.
        
        Args:
            last_level: Previous heading level
            current_level: Current heading level
            
        Returns:
            True if progression is valid
        """
        level_order = {"H1": 1, "H2": 2, "H3": 3}
        
        last_num = level_order.get(last_level, 1)
        current_num = level_order.get(current_level, 1)
        
        # Valid progressions:
        # - Same level (H1 → H1, H2 → H2, H3 → H3)
        # - One level deeper (H1 → H2, H2 → H3)
        # - Any level back to H1 (H2 → H1, H3 → H1)
        # - H3 back to H2 (H3 → H2)
        
        if current_num == last_num:  # Same level
            return True
        elif current_num == last_num + 1:  # One level deeper
            return True
        elif current_level == "H1":  # Back to top level
            return True
        elif last_level == "H3" and current_level == "H2":  # H3 back to H2
            return True
        
        return False
    
    def _correct_heading_level(self, last_level: str, current_level: str, level_stack: List[str]) -> str:
        """
        Correct an invalid heading level based on context.
        
        Args:
            last_level: Previous heading level
            current_level: Current (invalid) heading level
            level_stack: Current hierarchy stack
            
        Returns:
            Corrected heading level
        """
        level_order = {"H1": 1, "H2": 2, "H3": 3}
        
        last_num = level_order.get(last_level, 1)
        current_num = level_order.get(current_level, 1)
        
        # If jumping too many levels (e.g., H1 → H3), use intermediate level
        if current_num > last_num + 1:
            corrected_num = last_num + 1
            return f"H{corrected_num}"
        
        # If going backwards more than allowed, find appropriate level
        if current_num < last_num and current_level != "H1":
            # Look at the stack to find appropriate level
            if len(level_stack) >= 2 and current_level == "H2":
                # If we have H1 → H2 → H3 and now want H2, that's valid
                return current_level
            else:
                # Default to one level up from current
                return "H1" if last_level in ["H2", "H3"] else current_level
        
        return current_level
    
    def _update_level_stack(self, level_stack: List[str], new_level: str) -> None:
        """
        Update the hierarchy stack with the new heading level.
        
        Args:
            level_stack: Current hierarchy stack to update
            new_level: New heading level to add
        """
        level_order = {"H1": 1, "H2": 2, "H3": 3}
        new_num = level_order.get(new_level, 1)
        
        # Remove levels deeper than or equal to the new level
        while level_stack:
            last_level = level_stack[-1]
            last_num = level_order.get(last_level, 1)
            
            if last_num >= new_num:
                level_stack.pop()
            else:
                break
        
        # Add the new level
        level_stack.append(new_level)
    
    def calculate_heading_confidence_score(self, headings: List[HeadingInfo], font_candidates: List[HeadingCandidate] = None) -> List[HeadingInfo]:
        """
        Calculate confidence scores for headings and filter low-confidence ones.
        
        Args:
            headings: List of HeadingInfo objects
            font_candidates: Optional list of original font candidates for confidence reference
            
        Returns:
            List of HeadingInfo objects with high confidence
        """
        try:
            if not headings:
                return []
            
            # Create a mapping from text to original confidence if available
            confidence_map = {}
            if font_candidates:
                for candidate in font_candidates:
                    confidence_map[candidate.clean_text()] = candidate.confidence_score
            
            high_confidence_headings = []
            
            for heading in headings:
                # Get original confidence or calculate based on hierarchy
                original_confidence = confidence_map.get(heading.text, 0.5)
                
                # Calculate hierarchy-based confidence
                hierarchy_confidence = self._calculate_hierarchy_confidence(heading, headings)
                
                # Combine confidences (weighted average)
                combined_confidence = (original_confidence * 0.7) + (hierarchy_confidence * 0.3)
                
                # Only keep headings with reasonable confidence
                if combined_confidence >= 0.4:
                    high_confidence_headings.append(heading)
                else:
                    logger.debug(f"Filtered out low-confidence heading: '{heading.text}' (confidence: {combined_confidence:.2f})")
            
            logger.debug(f"Kept {len(high_confidence_headings)} high-confidence headings out of {len(headings)}")
            return high_confidence_headings
            
        except Exception as e:
            logger.error(f"Error calculating heading confidence scores: {e}")
            return headings
    
    def _calculate_hierarchy_confidence(self, heading: HeadingInfo, all_headings: List[HeadingInfo]) -> float:
        """
        Calculate confidence based on heading's position in hierarchy.
        
        Args:
            heading: HeadingInfo object to evaluate
            all_headings: All headings for context
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence
        
        # Count headings of each level
        level_counts = {"H1": 0, "H2": 0, "H3": 0}
        for h in all_headings:
            level_counts[h.level] = level_counts.get(h.level, 0) + 1
        
        # Bonus for balanced hierarchy
        total_headings = len(all_headings)
        if total_headings > 0:
            # H1 should be less frequent than H2/H3
            if heading.level == "H1" and level_counts["H1"] / total_headings <= 0.3:
                confidence += 0.2
            # H2 should be moderately frequent
            elif heading.level == "H2" and 0.2 <= level_counts["H2"] / total_headings <= 0.6:
                confidence += 0.1
            # H3 can be frequent
            elif heading.level == "H3":
                confidence += 0.05
        
        # Bonus for reasonable text length
        text_length = len(heading.text)
        if 5 <= text_length <= 100:
            confidence += 0.1
        
        # Penalty for very short or very long text
        if text_length < 3 or text_length > 150:
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def create_fallback_headings(self, doc: fitz.Document) -> List[HeadingInfo]:
        """
        Create fallback headings for documents with inconsistent font sizing.
        Uses simple heuristics when font-based detection fails.
        
        Args:
            doc: PyMuPDF Document object
            
        Returns:
            List of fallback HeadingInfo objects
        """
        try:
            if doc is None or doc.is_closed:
                logger.debug("Cannot create fallback headings from invalid document")
                return []
            
            fallback_headings = []
            page_count = self.get_page_count(doc)
            
            if page_count == 0:
                return []
            
            logger.debug("Creating fallback headings using simple heuristics")
            
            for page_num in range(min(page_count, 5)):  # Only check first 5 pages
                try:
                    page = doc[page_num]
                    page_headings = self._extract_fallback_page_headings(page, page_num + 1)
                    fallback_headings.extend(page_headings)
                    
                except Exception as e:
                    logger.error(f"Error extracting fallback headings from page {page_num + 1}: {e}")
                    continue
            
            # Validate and clean up fallback headings
            if fallback_headings:
                fallback_headings = self.validate_heading_hierarchy(fallback_headings)
            
            logger.debug(f"Created {len(fallback_headings)} fallback headings")
            return fallback_headings
            
        except Exception as e:
            logger.error(f"Error creating fallback headings: {e}")
            return []
    
    def _extract_fallback_page_headings(self, page: fitz.Page, page_number: int) -> List[HeadingInfo]:
        """
        Extract fallback headings from a page using simple heuristics.
        
        Args:
            page: PyMuPDF Page object
            page_number: 1-based page number
            
        Returns:
            List of HeadingInfo objects from simple heuristics
        """
        try:
            headings = []
            
            # Get text blocks
            text_blocks = page.get_text("blocks")
            
            for block in text_blocks:
                if len(block) < 5:  # Skip blocks without enough data
                    continue
                
                text = block[4].strip()  # Text content is at index 4
                
                if not text or len(text) < 3:
                    continue
                
                # Simple heuristics for headings
                is_potential_heading = False
                heading_level = "H3"  # Default to H3
                
                # Check for numbered sections (1., 1.1, etc.)
                if re.match(r'^\d+\.', text):
                    is_potential_heading = True
                    heading_level = "H1" if re.match(r'^\d+\.$', text.split()[0]) else "H2"
                
                # Check for short lines that might be headings
                elif len(text) <= 80 and '\n' not in text:
                    # Check if it looks like a heading
                    if (text.isupper() or text.istitle() or 
                        any(word in text.lower() for word in ['chapter', 'section', 'part', 'introduction', 'conclusion'])):
                        is_potential_heading = True
                        
                        # Determine level based on content
                        if any(word in text.lower() for word in ['chapter', 'part']):
                            heading_level = "H1"
                        elif any(word in text.lower() for word in ['section', 'subsection']):
                            heading_level = "H2"
                
                if is_potential_heading:
                    # Clean the text
                    cleaned_text = self._clean_heading_text(text)
                    
                    if cleaned_text:
                        heading = HeadingInfo(
                            level=heading_level,
                            text=cleaned_text,
                            page=page_number
                        )
                        
                        if heading.validate():
                            headings.append(heading)
            
            return headings
            
        except Exception as e:
            logger.error(f"Error extracting fallback headings from page {page_number}: {e}")
            return []

    def close_document(self, doc: fitz.Document) -> None:
        """
        Safely close a PDF document.
        
        Args:
            doc: PyMuPDF Document object to close
        """
        try:
            if doc is not None and not doc.is_closed:
                doc.close()
                logger.debug("Document closed successfully")
        except Exception as e:
            logger.error(f"Error closing document: {e}")