#!/usr/bin/env python3
"""
PDF Outline Extractor - Main Application Entry Point

This is the main entry point for the PDF Outline Extractor application.
It orchestrates the complete PDF processing workflow to extract structured
outlines from PDF documents and generate JSON outputs.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import gc

# Import application modules
from src.config import setup_logging, DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR
from src.file_scanner import FileScanner
from src.outline_extractor import OutlineExtractor
from src.json_generator import JSONGenerator
from src.error_handler import ErrorHandler

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


class PDFOutlineProcessor:
    """Main processor that orchestrates the complete PDF processing workflow."""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        """
        Initialize the PDF processor with input and output directories.
        
        Args:
            input_dir: Directory containing PDF files to process
            output_dir: Directory where JSON outputs will be written
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Initialize components
        self.file_scanner = FileScanner()
        self.outline_extractor = OutlineExtractor()
        self.json_generator = JSONGenerator()
        self.error_handler = ErrorHandler()
        
        # Processing statistics
        self.total_files = 0
        self.successful_files = 0
        self.failed_files = 0
        self.start_time = None
        
        # Thread safety for concurrent processing
        self._stats_lock = threading.Lock()
        
        # Performance optimization settings
        self.max_workers = 2  # Will be set dynamically based on file count
    
    def process_all_pdfs(self) -> bool:
        """
        Process all PDF files in the input directory with concurrent processing optimization.
        
        Returns:
            True if processing completed successfully, False otherwise
        """
        try:
            logger.info("Starting batch PDF processing with performance optimizations")
            self.start_time = time.time()
            
            # Start error handler monitoring
            self.error_handler.start_monitoring()
            
            # Scan for PDF files
            pdf_files = self.file_scanner.scan_input_directory(self.input_dir)
            self.total_files = len(pdf_files)
            
            if self.total_files == 0:
                logger.warning(f"No PDF files found in {self.input_dir}")
                return True
            
            logger.info(f"Found {self.total_files} PDF files to process")
            
            # Ensure output directory exists and is writable
            self._prepare_output_directory()
            
            # Use concurrent processing for multiple files
            if self.total_files > 1:
                success = self._process_pdfs_concurrently(pdf_files)
            else:
                # Single file - process normally
                success = self._process_single_pdf(pdf_files[0])
                if success:
                    self.successful_files = 1
                else:
                    self.failed_files = 1
            
            # Validate processing results
            self._validate_processing_results(pdf_files)
            
            # Log final statistics
            self._log_final_statistics()
            
            return self.failed_files == 0
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return False
    
    def _process_single_pdf(self, pdf_path: Path) -> bool:
        """
        Process a single PDF file with error handling and resource management.
        
        Args:
            pdf_path: Path to the PDF file to process
            
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Use error handler for safe processing
            def process_pdf(file_path):
                # Extract title and outline
                title, headings = self.outline_extractor.extract_title_and_outline(file_path)
                
                # Generate JSON output
                output_path = self.json_generator.process_pdf_to_json_safe(
                    file_path, self.output_dir, title, headings
                )
                
                logger.debug(f"Generated output: {output_path}")
                return {"title": title, "outline": [h.to_dict() for h in headings]}
            
            # Process with comprehensive error handling and monitoring
            result = self.error_handler.monitor_processing_performance(
                pdf_path, process_pdf
            )
            
            if result:
                logger.debug(f"Successfully processed {pdf_path.name}")
                return True
            else:
                logger.warning(f"Processing returned empty result for {pdf_path.name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            
            # Create fallback output for failed processing
            try:
                self.json_generator.process_pdf_to_json_safe(
                    pdf_path, self.output_dir, "", []
                )
                logger.info(f"Created fallback output for {pdf_path.name}")
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback output for {pdf_path.name}: {fallback_error}")
            
            return False
    
    def _process_pdfs_concurrently(self, pdf_files: list) -> bool:
        """
        Process multiple PDF files concurrently using ThreadPoolExecutor.
        
        Args:
            pdf_files: List of PDF file paths to process
            
        Returns:
            True if all processing completed, False if any failures
        """
        try:
            # Determine optimal number of workers based on file count and system resources
            self.max_workers = min(4, max(1, len(pdf_files) // 2))
            logger.info(f"Using {self.max_workers} concurrent workers for {len(pdf_files)} files")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all PDF processing tasks
                future_to_pdf = {
                    executor.submit(self._process_single_pdf_optimized, pdf_path): pdf_path 
                    for pdf_path in pdf_files
                }
                
                # Process completed tasks as they finish
                completed_count = 0
                for future in as_completed(future_to_pdf):
                    pdf_path = future_to_pdf[future]
                    completed_count += 1
                    
                    try:
                        success = future.result()
                        
                        # Thread-safe statistics update
                        with self._stats_lock:
                            if success:
                                self.successful_files += 1
                            else:
                                self.failed_files += 1
                        
                        # Log progress periodically
                        if completed_count % 5 == 0 or completed_count == len(pdf_files):
                            self._log_progress(completed_count)
                        
                        # Force garbage collection periodically to manage memory
                        if completed_count % 10 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        logger.error(f"Error processing {pdf_path.name}: {e}")
                        with self._stats_lock:
                            self.failed_files += 1
            
            logger.info(f"Concurrent processing completed: {self.successful_files} successful, {self.failed_files} failed")
            return True
            
        except Exception as e:
            logger.error(f"Concurrent processing failed: {e}")
            return False
    
    def _process_single_pdf_optimized(self, pdf_path: Path) -> bool:
        """
        Optimized version of single PDF processing with memory management and early termination.
        
        Args:
            pdf_path: Path to the PDF file to process
            
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            logger.debug(f"Starting optimized processing of {pdf_path.name}")
            
            # Use error handler for safe processing with optimizations
            def process_pdf_optimized(file_path):
                # Check for early termination opportunities
                doc = self.outline_extractor.pdf_processor.open_pdf(file_path)
                if not doc:
                    return None
                
                try:
                    # Early termination: if document has embedded outline, use it directly
                    if self.outline_extractor.pdf_processor.has_embedded_outline(doc):
                        logger.debug(f"Using embedded outline for fast processing: {file_path.name}")
                        title = self.outline_extractor.pdf_processor.extract_title(doc)
                        headings = self.outline_extractor.pdf_processor.extract_embedded_outline(doc)
                        return {"title": title, "outline": [h.to_dict() for h in headings]}
                    
                    # Fallback to full extraction with memory-efficient processing
                    title, headings = self.outline_extractor.extract_title_and_outline_optimized(file_path)
                    return {"title": title, "outline": [h.to_dict() for h in headings]}
                    
                finally:
                    # Ensure document is closed to free memory
                    if doc and not doc.is_closed:
                        doc.close()
                    # Force garbage collection for this thread
                    gc.collect()
            
            # Process with comprehensive error handling and monitoring
            result = self.error_handler.monitor_processing_performance(
                pdf_path, process_pdf_optimized
            )
            
            if result:
                # Generate JSON output
                self.json_generator.process_pdf_to_json_safe(
                    pdf_path, self.output_dir, result["title"], 
                    [self.outline_extractor._dict_to_heading_info(h) for h in result["outline"]]
                )
                logger.debug(f"Successfully processed {pdf_path.name}")
                return True
            else:
                logger.warning(f"Processing returned empty result for {pdf_path.name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            
            # Create fallback output for failed processing
            try:
                self.json_generator.process_pdf_to_json_safe(
                    pdf_path, self.output_dir, "", []
                )
                logger.info(f"Created fallback output for {pdf_path.name}")
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback output for {pdf_path.name}: {fallback_error}")
            
            return False
    
    def _prepare_output_directory(self):
        """Prepare and validate the output directory."""
        try:
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check permissions and disk space
            self.json_generator.check_output_directory_permissions(self.output_dir)
            self.json_generator.check_disk_space(self.output_dir, required_mb=50.0)
            
            logger.debug(f"Output directory prepared: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to prepare output directory: {e}")
            raise
    
    def _validate_processing_results(self, pdf_files: list):
        """Validate that all PDF files have corresponding JSON outputs."""
        try:
            validation_results = self.json_generator.validate_processing_results(
                pdf_files, self.output_dir
            )
            
            if validation_results['success']:
                logger.info("All PDF files have corresponding JSON outputs")
            else:
                missing_count = len(validation_results['missing_outputs'])
                logger.warning(f"{missing_count} PDF files are missing JSON outputs")
                
                for missing_pdf in validation_results['missing_outputs']:
                    logger.warning(f"Missing output for: {missing_pdf.name}")
            
        except Exception as e:
            logger.error(f"Failed to validate processing results: {e}")
    
    def _log_progress(self, current_file: int):
        """Log processing progress with performance metrics."""
        if self.start_time is None:
            return
            
        elapsed_time = time.time() - self.start_time
        files_per_second = current_file / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(
            f"Progress: {current_file}/{self.total_files} files "
            f"({current_file/self.total_files*100:.1f}%) - "
            f"Success: {self.successful_files}, Failed: {self.failed_files} - "
            f"Rate: {files_per_second:.2f} files/sec"
        )
        
        # Log performance metrics
        self.error_handler.log_performance_metrics(f"Progress update: {current_file}/{self.total_files}")
    
    def _log_final_statistics(self):
        """Log final processing statistics and performance metrics."""
        elapsed_time = time.time() - self.start_time
        
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total files processed: {self.total_files}")
        logger.info(f"Successful: {self.successful_files}")
        logger.info(f"Failed: {self.failed_files}")
        logger.info(f"Success rate: {self.successful_files/max(self.total_files,1)*100:.1f}%")
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        logger.info(f"Average time per file: {elapsed_time/max(self.total_files,1):.2f} seconds")
        
        # Log detailed performance metrics
        performance_metrics = self.error_handler.get_detailed_performance_metrics()
        logger.info(f"Peak memory usage: {performance_metrics['session_metrics']['peak_memory_mb']:.1f} MB")
        
        # Log error summary
        error_summary = self.error_handler.get_error_summary()
        if error_summary['total_errors'] > 0:
            logger.warning(f"Total errors encountered: {error_summary['total_errors']}")
            for category, count in error_summary['error_categories'].items():
                logger.warning(f"  {category}: {count}")
        
        # Log output directory summary
        output_summary = self.json_generator.get_output_summary(self.output_dir)
        logger.info(f"Output files created: {output_summary['json_files']}")
        logger.info(f"Total output size: {output_summary['total_size_mb']:.2f} MB")
        logger.info("=" * 60)


def parse_arguments():
    """
    Parse command-line arguments for the PDF Outline Extractor.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="PDF Outline Extractor - Extract structured outlines from PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default Docker paths
  python main.py
  
  # Use custom input/output directories
  python main.py --input /custom/input --output /custom/output
  
  # Enable debug logging
  python main.py --verbose
  
  # Show version information
  python main.py --version

Docker Usage:
  docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output/repoidentifier/:/app/output --network none <reponame.someidentifier>
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory containing PDF files (default: {DEFAULT_INPUT_DIR})"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for JSON files (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true", 
        help="Suppress all output except errors"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="PDF Outline Extractor v1.0.0 - Adobe India Hackathon Challenge 1a"
    )
    
    parser.add_argument(
        "--docker-mode",
        action="store_true",
        help="Enable Docker-specific optimizations and logging"
    )
    
    return parser.parse_args()


def setup_logging_from_args(args):
    """
    Configure logging based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
    """
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    # Reconfigure logging with new level
    setup_logging(log_level)
    
    # Update logger reference and set level for all relevant loggers
    global logger
    logger = logging.getLogger(__name__)
    
    # Set level for all application loggers
    for logger_name in ['__main__', 'main', 'src.file_scanner', 'src.outline_extractor', 
                       'src.json_generator', 'src.error_handler', 'src.pdf_processor']:
        logging.getLogger(logger_name).setLevel(log_level)


def validate_directories(input_dir: Path, output_dir: Path, docker_mode: bool = False):
    """
    Validate input and output directories with Docker-specific checks.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        docker_mode: Whether running in Docker mode
        
    Raises:
        SystemExit: If validation fails
    """
    try:
        # Validate input directory
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            if docker_mode:
                logger.error("Ensure the input directory is properly mounted to /app/input")
            sys.exit(2)
        
        if not input_dir.is_dir():
            logger.error(f"Input path is not a directory: {input_dir}")
            sys.exit(2)
        
        # Check input directory permissions
        import os
        if not os.access(input_dir, os.R_OK):
            logger.error(f"Input directory is not readable: {input_dir}")
            if docker_mode:
                logger.error("Check Docker volume mount permissions for /app/input")
            sys.exit(2)
        
        # Validate/create output directory
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.error(f"Cannot create output directory: {output_dir}")
            if docker_mode:
                logger.error("Check Docker volume mount permissions for /app/output")
            sys.exit(2)
        
        if not output_dir.is_dir():
            logger.error(f"Output path exists but is not a directory: {output_dir}")
            sys.exit(2)
        
        # Check output directory write permissions
        try:
            test_file = output_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
        except (PermissionError, OSError):
            logger.error(f"Output directory is not writable: {output_dir}")
            if docker_mode:
                logger.error("Check Docker volume mount permissions for /app/output")
            sys.exit(2)
        
        logger.debug(f"Directory validation successful - Input: {input_dir}, Output: {output_dir}")
        
    except Exception as e:
        logger.error(f"Directory validation failed: {e}")
        sys.exit(2)


def log_startup_info(args, input_dir: Path, output_dir: Path):
    """
    Log startup information and configuration.
    
    Args:
        args: Parsed command-line arguments
        input_dir: Input directory path
        output_dir: Output directory path
    """
    logger.info("=" * 60)
    logger.info("PDF OUTLINE EXTRACTOR STARTING")
    logger.info("=" * 60)
    logger.info(f"Version: 1.0.0")
    logger.info(f"Docker mode: {'Enabled' if args.docker_mode else 'Disabled'}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log level: {logging.getLevelName(logger.getEffectiveLevel())}")
    
    # Log Docker-specific information
    if args.docker_mode:
        logger.info("Running in Docker container mode")
        logger.info("Expected Docker command:")
        logger.info("  docker run --rm -v $(pwd)/input:/app/input:ro \\")
        logger.info("             -v $(pwd)/output/repoidentifier/:/app/output \\")
        logger.info("             --network none <reponame.someidentifier>")
    
    # Log system information
    import platform
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info("=" * 60)


def log_shutdown_info(success: bool, exit_code: int):
    """
    Log shutdown information and final status.
    
    Args:
        success: Whether processing was successful
        exit_code: Exit code to be returned
    """
    logger.info("=" * 60)
    logger.info("PDF OUTLINE EXTRACTOR SHUTDOWN")
    logger.info("=" * 60)
    logger.info(f"Status: {'SUCCESS' if success else 'FAILED'}")
    logger.info(f"Exit code: {exit_code}")
    
    if exit_code == 0:
        logger.info("All PDF files processed successfully")
    elif exit_code == 1:
        logger.error("Processing completed with errors")
    elif exit_code == 2:
        logger.error("Configuration or setup error")
    elif exit_code == 130:
        logger.info("Processing interrupted by user")
    else:
        logger.error(f"Unexpected exit code: {exit_code}")
    
    logger.info("=" * 60)


def main():
    """Main application entry point with command-line interface and Docker support."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Configure logging based on arguments
    setup_logging_from_args(args)
    
    # Get input and output directories
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    
    # Log startup information
    log_startup_info(args, input_dir, output_dir)
    
    try:
        # Validate directories with Docker-specific checks
        validate_directories(input_dir, output_dir, args.docker_mode)
        
        # Initialize and run the processor
        processor = PDFOutlineProcessor(input_dir, output_dir)
        success = processor.process_all_pdfs()
        
        # Determine exit code
        if success:
            exit_code = 0
        else:
            exit_code = 1
        
        # Log shutdown information
        log_shutdown_info(success, exit_code)
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user (Ctrl+C)")
        log_shutdown_info(False, 130)
        sys.exit(130)
    except SystemExit:
        # Re-raise SystemExit to preserve exit codes
        raise
    except Exception as e:
        logger.error(f"Application failed with unexpected error: {e}")
        logger.debug("Exception details:", exc_info=True)
        log_shutdown_info(False, 1)
        sys.exit(1)


if __name__ == "__main__":
    main()