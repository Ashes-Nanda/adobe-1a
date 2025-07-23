"""
JSON Schema Validator for PDF Outline Extractor

This module provides validation functionality to ensure output JSON
conforms to the required schema format.
"""

import json
import jsonschema
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.models import HeadingInfo


class JSONValidator:
    """Validates JSON output against the required schema."""
    
    def __init__(self, schema_path: Optional[Path] = None):
        """
        Initialize the validator with schema.
        
        Args:
            schema_path: Path to the JSON schema file. If None, uses built-in schema.
        """
        if schema_path and schema_path.exists():
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.schema = json.load(f)
        else:
            # Use a more flexible schema that allows any number of outline items
            self.schema = {
                "$schema": "http://json-schema.org/draft-04/schema#",
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string"
                    },
                    "outline": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "level": {
                                    "type": "string",
                                    "enum": ["H1", "H2", "H3"]
                                },
                                "text": {
                                    "type": "string"
                                },
                                "page": {
                                    "type": "integer",
                                    "minimum": 0
                                }
                            },
                            "required": ["level", "text", "page"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["title", "outline"],
                "additionalProperties": False
            }
    
    def validate_json_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate JSON data against the schema.
        
        Args:
            data: Dictionary containing the JSON data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            jsonschema.validate(data, self.schema)
            return True
        except jsonschema.ValidationError:
            return False
        except jsonschema.SchemaError:
            return False
    
    def validate_json_string(self, json_string: str) -> bool:
        """
        Validate JSON string against the schema.
        
        Args:
            json_string: JSON string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            data = json.loads(json_string)
            return self.validate_json_data(data)
        except json.JSONDecodeError:
            return False
    
    def validate_json_file(self, file_path: Path) -> bool:
        """
        Validate JSON file against the schema.
        
        Args:
            file_path: Path to JSON file to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self.validate_json_data(data)
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            return False
    
    def get_validation_errors(self, data: Dict[str, Any]) -> List[str]:
        """
        Get detailed validation errors for JSON data.
        
        Args:
            data: Dictionary containing the JSON data to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        try:
            jsonschema.validate(data, self.schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Validation error: {e.message}")
            if e.path:
                errors.append(f"Error path: {' -> '.join(str(p) for p in e.path)}")
        except jsonschema.SchemaError as e:
            errors.append(f"Schema error: {e.message}")
        
        return errors
    
    def ensure_output_format_compliance(self, title: str, headings: List[HeadingInfo]) -> Dict[str, Any]:
        """
        Ensure the output data complies with the required format.
        
        Args:
            title: Document title (can be empty string)
            headings: List of HeadingInfo objects
            
        Returns:
            Dictionary in the correct format for JSON output
        """
        # Clean and validate title
        clean_title = title.strip() if title else ""
        
        # Convert headings to dictionary format and validate
        outline = []
        for heading in headings:
            if heading.validate():
                outline.append(heading.to_dict())
        
        # Create the output structure
        output_data = {
            "title": clean_title,
            "outline": outline
        }
        
        # Validate the output
        if not self.validate_json_data(output_data):
            # If validation fails, return minimal valid structure
            return {"title": "", "outline": []}
        
        return output_data
    
    def create_fallback_output(self, filename: str = "") -> Dict[str, Any]:
        """
        Create a fallback output structure for error cases.
        
        Args:
            filename: Optional filename for context
            
        Returns:
            Valid empty JSON structure
        """
        return {"title": "", "outline": []}
    
    def validate_heading_level(self, level: str) -> bool:
        """
        Validate that a heading level is one of the allowed values.
        
        Args:
            level: Heading level string to validate
            
        Returns:
            True if valid heading level, False otherwise
        """
        return level in ["H1", "H2", "H3"]
    
    def validate_page_number(self, page: int) -> bool:
        """
        Validate that a page number is valid (non-negative integer).
        
        Args:
            page: Page number to validate
            
        Returns:
            True if valid page number, False otherwise
        """
        return isinstance(page, int) and page >= 1
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text content for JSON output.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace and normalize
        import re
        sanitized = re.sub(r'\s+', ' ', text.strip())
        
        # Remove any null characters or other problematic characters
        sanitized = sanitized.replace('\x00', '').replace('\r', ' ').replace('\n', ' ')
        
        return sanitized