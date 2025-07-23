"""
JSON Generator for PDF Outline Extractor

This module handles the generation of JSON output files from extracted
PDF outline data, ensuring compliance with the required schema.
"""

import json
import re
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import jsonschema
from .models import HeadingInfo


class JSONGenerator:
    """Handles JSON output generation and validation for PDF outline data."""
    
    def __init__(self, schema_path: Optional[Path] = None):
        """
        Initialize JSONGenerator with optional schema validation.
        
        Args:
            schema_path: Path to JSON schema file for validation
        """
        self.schema = None
        if schema_path and schema_path.exists():
            try:
                with open(schema_path, 'r', encoding='utf-8') as f:
                    self.schema = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                # Schema loading failed, continue without validation
                self.schema = None
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize and clean text for JSON output.
        
        Args:
            text: Raw text to sanitize
            
        Returns:
            Cleaned text with normalized whitespace
        """
        if not isinstance(text, str):
            return ""
        
        # Strip leading/trailing whitespace
        cleaned = text.strip()
        
        # Normalize internal whitespace (multiple spaces/tabs/newlines to single space)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove control characters except common ones
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
        
        return cleaned
    
    def validate_page_number(self, page: Any) -> int:
        """
        Validate and normalize page number to 1-based integer.
        
        Args:
            page: Page number to validate
            
        Returns:
            Valid 1-based page number
            
        Raises:
            ValueError: If page number is invalid
        """
        if not isinstance(page, (int, float)):
            raise ValueError(f"Page number must be numeric, got {type(page)}")
        
        page_int = int(page)
        if page_int < 1:
            raise ValueError(f"Page number must be >= 1, got {page_int}")
        
        return page_int
    
    def validate_heading_level(self, level: str) -> str:
        """
        Validate heading level format.
        
        Args:
            level: Heading level to validate
            
        Returns:
            Valid heading level
            
        Raises:
            ValueError: If heading level is invalid
        """
        valid_levels = {"H1", "H2", "H3"}
        if not isinstance(level, str) or level not in valid_levels:
            raise ValueError(f"Heading level must be one of {valid_levels}, got {level}")
        
        return level
    
    def format_outline_data(self, title: str, headings: List[HeadingInfo]) -> Dict[str, Any]:
        """
        Format extracted outline data into required JSON structure.
        
        Args:
            title: Document title (can be empty string)
            headings: List of HeadingInfo objects
            
        Returns:
            Dictionary in required JSON format
            
        Raises:
            ValueError: If data validation fails
        """
        # Sanitize title
        clean_title = self.sanitize_text(title) if title else ""
        
        # Process headings
        outline = []
        for heading in headings:
            if not isinstance(heading, HeadingInfo):
                raise ValueError(f"Expected HeadingInfo object, got {type(heading)}")
            
            # Validate heading data
            if not heading.validate():
                raise ValueError(f"Invalid HeadingInfo: {heading}")
            
            # Create outline entry
            outline_entry = {
                "level": self.validate_heading_level(heading.level),
                "text": self.sanitize_text(heading.text),
                "page": self.validate_page_number(heading.page)
            }
            outline.append(outline_entry)
        
        # Create final JSON structure
        json_data = {
            "title": clean_title,
            "outline": outline
        }
        
        return json_data
    
    def validate_schema_compliance(self, data: Dict[str, Any]) -> bool:
        """
        Validate JSON data against schema if available.
        
        Args:
            data: JSON data to validate
            
        Returns:
            True if valid or no schema available, False otherwise
        """
        if not self.schema:
            # No schema available, perform basic validation
            return self._basic_validation(data)
        
        try:
            jsonschema.validate(data, self.schema)
            return True
        except jsonschema.ValidationError:
            return False
    
    def _basic_validation(self, data: Dict[str, Any]) -> bool:
        """
        Perform basic validation when schema is not available.
        
        Args:
            data: JSON data to validate
            
        Returns:
            True if basic validation passes
        """
        # Check required fields
        if not isinstance(data, dict):
            return False
        
        if "title" not in data or "outline" not in data:
            return False
        
        # Validate title
        if not isinstance(data["title"], str):
            return False
        
        # Validate outline
        if not isinstance(data["outline"], list):
            return False
        
        # Validate outline entries
        for entry in data["outline"]:
            if not isinstance(entry, dict):
                return False
            
            required_fields = {"level", "text", "page"}
            if not all(field in entry for field in required_fields):
                return False
            
            if not isinstance(entry["level"], str):
                return False
            
            if not isinstance(entry["text"], str):
                return False
            
            if not isinstance(entry["page"], int) or entry["page"] < 1:
                return False
        
        return True
    
    def generate_output_filename(self, input_filename: str) -> str:
        """
        Generate output JSON filename from input PDF filename.
        
        Args:
            input_filename: Input PDF filename (e.g., "document.pdf")
            
        Returns:
            Output JSON filename (e.g., "document.json")
        """
        if not isinstance(input_filename, str):
            raise ValueError("Input filename must be a string")
        
        if not input_filename.strip():
            raise ValueError("Invalid input filename")
        
        # Remove path components and get base filename
        base_name = Path(input_filename).stem
        
        # Ensure we have a valid filename (not empty and not just dots)
        if not base_name or base_name.startswith('.'):
            raise ValueError("Invalid input filename")
        
        return f"{base_name}.json"
    
    def write_json_file(self, data: Dict[str, Any], output_path: Path) -> None:
        """
        Write JSON data to file with proper formatting.
        
        Args:
            data: JSON data to write
            output_path: Path where to write the JSON file
            
        Raises:
            ValueError: If data validation fails
            IOError: If file writing fails
        """
        # Validate data before writing
        if not self.validate_schema_compliance(data):
            raise ValueError("JSON data does not comply with required schema")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except IOError as e:
            raise IOError(f"Failed to write JSON file {output_path}: {e}")
    
    def create_empty_output(self, title: str = "") -> Dict[str, Any]:
        """
        Create empty but valid JSON output for error cases.
        
        Args:
            title: Optional title to include
            
        Returns:
            Valid empty JSON structure
        """
        return {
            "title": self.sanitize_text(title) if title else "",
            "outline": []
        }
    
    def check_output_directory_permissions(self, output_dir: Path) -> None:
        """
        Check if output directory has proper write permissions.
        
        Args:
            output_dir: Directory to check
            
        Raises:
            PermissionError: If directory is not writable
            OSError: If directory cannot be accessed
        """
        try:
            # Ensure directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions by creating a temporary file
            test_file = output_dir / ".write_test"
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                test_file.unlink()  # Clean up test file
            except (PermissionError, OSError) as e:
                raise PermissionError(f"Output directory {output_dir} is not writable: {e}")
                
        except OSError as e:
            raise OSError(f"Cannot access output directory {output_dir}: {e}")
    
    def check_disk_space(self, output_dir: Path, required_mb: float = 10.0) -> None:
        """
        Check if there's sufficient disk space for output files.
        
        Args:
            output_dir: Directory to check
            required_mb: Required space in megabytes
            
        Raises:
            OSError: If insufficient disk space
        """
        try:
            # Get disk usage statistics
            stat = shutil.disk_usage(output_dir)
            free_mb = stat.free / (1024 * 1024)  # Convert bytes to MB
            
            if free_mb < required_mb:
                raise OSError(f"Insufficient disk space. Required: {required_mb}MB, Available: {free_mb:.1f}MB")
                
        except (AttributeError, NotImplementedError):
            # If disk_usage is not available on this platform, skip the check
            pass
    
    def process_pdf_to_json(self, pdf_path: Path, output_dir: Path, 
                           title: str, headings: List[HeadingInfo]) -> Path:
        """
        Complete workflow to process PDF data and write JSON output.
        
        Args:
            pdf_path: Path to the source PDF file
            output_dir: Directory where JSON should be written
            title: Extracted document title
            headings: List of extracted headings
            
        Returns:
            Path to the created JSON file
            
        Raises:
            ValueError: If input data is invalid
            PermissionError: If output directory is not writable
            OSError: If file operations fail
        """
        # Validate inputs
        if not isinstance(pdf_path, Path):
            raise ValueError("pdf_path must be a Path object")
        
        if not isinstance(output_dir, Path):
            raise ValueError("output_dir must be a Path object")
        
        # Check output directory permissions and disk space
        self.check_output_directory_permissions(output_dir)
        self.check_disk_space(output_dir)
        
        # Generate output filename
        output_filename = self.generate_output_filename(pdf_path.name)
        output_path = output_dir / output_filename
        
        # Format and validate data
        json_data = self.format_outline_data(title, headings)
        
        # Write JSON file
        self.write_json_file(json_data, output_path)
        
        return output_path
    
    def process_pdf_to_json_safe(self, pdf_path: Path, output_dir: Path, 
                                title: str = "", headings: Optional[List[HeadingInfo]] = None) -> Path:
        """
        Safe version of process_pdf_to_json that handles errors gracefully.
        
        Args:
            pdf_path: Path to the source PDF file
            output_dir: Directory where JSON should be written
            title: Extracted document title (optional)
            headings: List of extracted headings (optional)
            
        Returns:
            Path to the created JSON file
            
        Note:
            This method will create an empty but valid JSON file if processing fails
        """
        if headings is None:
            headings = []
        
        try:
            return self.process_pdf_to_json(pdf_path, output_dir, title, headings)
        except Exception:
            # If processing fails, create empty output
            try:
                output_filename = self.generate_output_filename(pdf_path.name)
                output_path = output_dir / output_filename
                empty_data = self.create_empty_output()
                self.write_json_file(empty_data, output_path)
                return output_path
            except Exception:
                # If even empty output fails, re-raise the original error
                raise
    
    def validate_processing_results(self, pdf_files: List[Path], output_dir: Path) -> Dict[str, Any]:
        """
        Validate that all processed PDFs have corresponding JSON outputs.
        
        Args:
            pdf_files: List of PDF files that were processed
            output_dir: Directory where JSON files should be located
            
        Returns:
            Dictionary with validation results containing:
            - 'success': bool indicating if all files have outputs
            - 'missing_outputs': list of PDF files without corresponding JSON
            - 'existing_outputs': list of JSON files that exist
            - 'total_pdfs': total number of PDF files
            - 'total_outputs': total number of JSON outputs found
        """
        missing_outputs = []
        existing_outputs = []
        
        for pdf_path in pdf_files:
            try:
                output_filename = self.generate_output_filename(pdf_path.name)
                output_path = output_dir / output_filename
                
                if output_path.exists() and output_path.is_file():
                    existing_outputs.append(output_path)
                else:
                    missing_outputs.append(pdf_path)
            except Exception:
                # If filename generation fails, consider it missing
                missing_outputs.append(pdf_path)
        
        return {
            'success': len(missing_outputs) == 0,
            'missing_outputs': missing_outputs,
            'existing_outputs': existing_outputs,
            'total_pdfs': len(pdf_files),
            'total_outputs': len(existing_outputs)
        }
    
    def cleanup_output_directory(self, output_dir: Path, keep_files: Optional[Set[str]] = None) -> None:
        """
        Clean up output directory, optionally keeping specified files.
        
        Args:
            output_dir: Directory to clean up
            keep_files: Set of filenames to keep (optional)
            
        Raises:
            PermissionError: If files cannot be deleted
            OSError: If directory operations fail
        """
        if not output_dir.exists():
            return
        
        if keep_files is None:
            keep_files = set()
        
        try:
            for file_path in output_dir.iterdir():
                if file_path.is_file() and file_path.name not in keep_files:
                    if file_path.suffix.lower() == '.json':
                        file_path.unlink()
        except (PermissionError, OSError) as e:
            raise OSError(f"Failed to cleanup output directory {output_dir}: {e}")
    
    def get_output_summary(self, output_dir: Path) -> Dict[str, Any]:
        """
        Get summary information about output directory contents.
        
        Args:
            output_dir: Directory to analyze
            
        Returns:
            Dictionary with summary information:
            - 'total_files': total number of files
            - 'json_files': number of JSON files
            - 'total_size_mb': total size in megabytes
            - 'files': list of file information
        """
        if not output_dir.exists():
            return {
                'total_files': 0,
                'json_files': 0,
                'total_size_mb': 0.0,
                'files': []
            }
        
        files_info = []
        total_size = 0
        json_count = 0
        
        try:
            for file_path in output_dir.iterdir():
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    
                    file_info = {
                        'name': file_path.name,
                        'size_bytes': file_size,
                        'is_json': file_path.suffix.lower() == '.json'
                    }
                    
                    if file_info['is_json']:
                        json_count += 1
                    
                    files_info.append(file_info)
        except OSError:
            # If we can't read directory, return empty summary
            pass
        
        return {
            'total_files': len(files_info),
            'json_files': json_count,
            'total_size_mb': total_size / (1024 * 1024),
            'files': files_info
        }