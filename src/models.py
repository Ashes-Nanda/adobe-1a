"""
Data models for PDF Outline Extractor

This module contains the data classes and models used throughout
the PDF processing pipeline.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import re


@dataclass
class TextBlock:
    """Represents a block of text extracted from a PDF with formatting information."""
    text: str
    font_name: str
    font_size: float
    is_bold: bool
    is_italic: bool
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    page_number: int
    
    def validate(self) -> bool:
        """Validate the TextBlock data."""
        if not isinstance(self.text, str):
            return False
        if not isinstance(self.font_name, str):
            return False
        if not isinstance(self.font_size, (int, float)) or self.font_size <= 0:
            return False
        if not isinstance(self.is_bold, bool):
            return False
        if not isinstance(self.is_italic, bool):
            return False
        if not isinstance(self.bbox, tuple) or len(self.bbox) != 4:
            return False
        if not all(isinstance(coord, (int, float)) for coord in self.bbox):
            return False
        if not isinstance(self.page_number, int) or self.page_number < 1:
            return False
        return True
    
    def clean_text(self) -> str:
        """Return cleaned text with trimmed whitespace."""
        return self.text.strip()


@dataclass
class HeadingCandidate:
    """Represents a potential heading identified during text analysis."""
    text: str
    font_size: float
    font_weight: str
    position: Tuple[float, float]
    page_number: int
    confidence_score: float
    
    def validate(self) -> bool:
        """Validate the HeadingCandidate data."""
        if not isinstance(self.text, str) or not self.text.strip():
            return False
        if not isinstance(self.font_size, (int, float)) or self.font_size <= 0:
            return False
        if not isinstance(self.font_weight, str):
            return False
        if not isinstance(self.position, tuple) or len(self.position) != 2:
            return False
        if not all(isinstance(coord, (int, float)) for coord in self.position):
            return False
        if not isinstance(self.page_number, int) or self.page_number < 1:
            return False
        if not isinstance(self.confidence_score, (int, float)) or not (0.0 <= self.confidence_score <= 1.0):
            return False
        return True
    
    def clean_text(self) -> str:
        """Return cleaned text with trimmed whitespace and normalized spacing."""
        return re.sub(r'\s+', ' ', self.text.strip())


@dataclass
class HeadingInfo:
    """Represents a confirmed heading with its hierarchical level."""
    level: str  # "H1", "H2", "H3"
    text: str
    page: int
    
    def validate(self) -> bool:
        """Validate the HeadingInfo data."""
        valid_levels = {"H1", "H2", "H3"}
        if not isinstance(self.level, str) or self.level not in valid_levels:
            return False
        if not isinstance(self.text, str) or not self.text.strip():
            return False
        if not isinstance(self.page, int) or self.page < 1:
            return False
        return True
    
    def clean_text(self) -> str:
        """Return cleaned text with trimmed whitespace and normalized spacing."""
        return re.sub(r'\s+', ' ', self.text.strip())
    
    def to_dict(self) -> dict:
        """Convert to dictionary format for JSON output."""
        return {
            "level": self.level,
            "text": self.clean_text(),
            "page": self.page
        }